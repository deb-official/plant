#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: topics.py
Description: This script demonstrates how to use argparse to take inputs from the command line. It performs basic operations based on the provided arguments.
Author: Your Name
Created Date: YYYY-MM-DD
Last Modified Date: YYYY-MM-DD
Version: 1.0
Usage: python topics.py --sum 10 20
"""

# Import standard libraries
import os
import ast
import sys
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
import threading
from itertools import islice
import re
from copy import deepcopy

# Import xcube specific libraries
from xcube.imports import *
from xcube.data.all import *
from xcube.text.topicmodel import *

# Import specific libraries
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk 
from nltk.tokenize import sent_tokenize, word_tokenize
import openai
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.representation import OpenAI
from bertopic.representation._utils import (retry_with_exponential_backoff, truncate_document)
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import pipeline
# import warnings
# warnings.filterwarnings("ignore")
from transformers import logging as transformers_logging
# Set Hugging Face logging level to ERROR to suppress info/warning messages
transformers_logging.set_verbosity_error()

# Import external libraries
from tqdm import tqdm
from contextlib import contextmanager
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from gpl.toolkit import (
    qgen,
    NegativeMiner,
    PseudoLabeler,
    load_sbert,
    GenerativePseudoLabelingDataset,
    MarginDistillationLoss,
    evaluate
)

# Constants
openai_API_KEY=os.getenv('OPENAI_API_KEY', None)
base_dir = None

def chunked_generator(source_generator, chunk_size):
    """
    Yields items from a generator in chunks of a given size.
    
    Args:
        source_generator (generator): The generator to chunk.
        chunk_size (int): The number of items in each chunk.
    
    Yields:
        list: A list of items from the generator, with length equal to chunk_size or less.
    """
    chunk = []  # Initialize the chunk list
    for item in source_generator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []  # Reset the chunk list
    if chunk:  # Yield the last chunk if it contains any items
        yield chunk

def first_k_pairs(original_dict, k):
    """
    Returns a new dictionary containing the first k key-value pairs from the original dictionary,
    respecting the order of insertion.

    Parameters:
    - original_dict (dict): The original dictionary from which to extract key-value pairs.
    - k (int): The number of key-value pairs to extract.

    Returns:
    - dict: A new dictionary containing the first k key-value pairs.
    """
    # Use itertools.islice to efficiently slice the first k items from the dictionary's items iterator
    return dict(islice(original_dict.items(), k))

def extract_dict_list_from_string(s):
    """
    Extracts a list of Python dictionaries written in a string using regex.

    Args:
    s (str): The string containing the list of Python dictionary.

    Returns:
    list: The extracted list of dictionaries or None if no dictionary is found.
    """
    # Regex pattern to capture contents between { and } within code blocks marked by either 'python' or 'json'
    # pattern = r"```(?:python|json)\s*\n({.*?})\n```"  # non-greedy matching inside triple backticks
    pattern = r"```(?:python|json)\s*\n(\[\s*(?:{.*?}\s*,*\s*)*\])\n```"


    match = re.search(pattern, s, re.DOTALL)  # DOTALL to make '.' match newline characters

    if match:
        # Extract the dictionary string
        dict_str = match.group(1)
        # Convert the dictionary string to an actual dictionary
        try:
            result_dict = ast.literal_eval(dict_str)
            return result_dict
        except ValueError as e:
            print(f"Error parsing string to dict: {e}")
            return None
    else:
        return None

from concurrent.futures import ThreadPoolExecutor, as_completed
class CustomOpenAI(OpenAI):

    INSTRUCTION = """
    Based on the information above, extract top 5 (in decreasing ranking) most similar products in a dictionary format.
    """

    SYSTEM_MSG = """
        ### Output:
        ```python
        {
            '6305613516': "Sarah, Plain and Tall: Winter's End [VHS] (1999)",
            'B001BN4WJU': 'The Sarah Silverman Program: Season Two, Vol. One (2008)',
            'B002XKKX7A': 'Sarah Silverman Program: Season Two, Vol. 2 (2010)',
            'B0096W4702': 'The Sarah Silverman Program: Season Three (2007)',
            'B000BVXQWM': 'Songs In The Style of Sarah Brightman, Vol. 2'
        }
        ```
    """

    def _process_documents(self, documents, topic_model, topk_lbs_cutoff, doc_length, tokenizer,
                           labels_col='predicted_labels',
                           scores_col='predicted_scores'):

        """
        Private generator to process documents and yield the results for each document.

        Args:
        documents (DataFrame): DataFrame containing the document data.
        topk_lbs_cutoff (int): Number of top labels and scores to consider.

        Yields:
        dict: A dictionary containing the processed results for each document.
        """
        for index, row in documents.iterrows():
            title = row['title']
            content = row['content']
            predicted_scores = first_k_pairs(row[scores_col], k=topk_lbs_cutoff)
            predicted_labels = first_k_pairs(row[labels_col], k=topk_lbs_cutoff)
            description = str(title) + "\n" + str(content)
            description = truncate_document(topic_model, doc_length, tokenizer, description)
            qsummary = row.get('qsummary', None) if 'qsummary' in documents.columns and pd.notna(row.get('qsummary')) else None
            prompt = self._create_reranking_prompt(index, description, predicted_labels)

            yield {
                'index': index,
                'title': title,
                'content': content,
                'predicted_scores': predicted_scores,
                'predicted_labels': predicted_labels,
                'description': description,
                'qsummary': qsummary,
                'prompt': prompt
            }

    def _create_api_request_msg(self, doc, k=5):
        # Unpack doc values
        index, title, content, predicted_scores, predicted_labels, description, qsummary, prompt = doc.values()
        prompt += CustomOpenAI.INSTRUCTION
        
        messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant. Based on an Amazon product description your job is to pick top {k} (in decreasing ranking) similar products from a given dictionary of product ids and their descriptions. Give the answer in the following example format:\n{CustomOpenAI.SYSTEM_MSG}"
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        # This is the new prompting style
        messages = self._generate_ranking_messages(index, description, qsummary, predicted_labels)

        return index, messages
    
    def _create_api_request_summary(self, doc):
        # prompt = f"Summarize the following wikipedia text in a concise and clear manner such that I can use the summary to find titles of related articles:\n\n{doc}"
        prompt = f"{doc}"

        messages = [
        {
            "role": "system",
            "content": f"The following is a Wikipedia article. Based on the description generate a summary of 20 words that can be used to search the titles of related Wikipedia articles."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        return messages

    def _generate_chatgpt_messages(self, input_text: str, related_titles: list):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI language model that converts descriptive text into a question where the answer corresponds "
                    "to the 'related' article title(s). Your goal is to generate a natural-sounding question that leads directly "
                    "to the related topic(s). The question should be clear, concise, and framed in a way that makes the related topic the best possible answer."
                )
            },
            {
                "role": "user",
                "content": f"""Given the following input:

            ### **Input:**
            "text": "{input_text}"
            "related": {related_titles}

            ### **Instructions:**
            1. Generate a **single question** that naturally leads to the related article(s) as the answer.
            2. Ensure the question is phrased in a **neutral and informative** manner.
            3. Do **not** include the related article title directly in the question, but rather **imply it**.
            4. If multiple related topics exist, craft the question so that it encompasses all of them.

            ### **Examples:**

            #### **Example 1**
            **Input:**
            "text": "Moorish oven\nA Moorish oven ('hornos morunos' in Spanish) is a traditional clay wood-fired oven. It is commonly used in Morocco and environs for the manufacture of plaster. During the Moorish period, the plaster produced was used to build walls in Madrid, leading to the construction of over a hundred Moorish ovens. These ovens were in operation until the mid-century."
            "related": ["Wood-fired oven"]

            **Output:**
            "What type of traditional clay wood-fired oven was historically used in Morocco and Spain for plaster production and construction?"

            ---

            #### **Example 2**
            **Input:**
            "text": "Expulsion from the United States Congress\nExpulsion is the most serious form of disciplinary action that can be taken against a Member of Congress. Article I, Section 5 of the United States Constitution allows each House to expel a member with a two-thirds vote. The House Ethics Committee or the Senate Ethics Committee investigates misconduct before a vote on expulsion. Censure is a less severe form of disciplinary action that does not remove a member from office."
            "related": ["List of United States Representatives expelled, censured, or reprimanded", "List of United States senators expelled or censured"]

            **Output:**
            "What historical records document disciplinary actions, including expulsions and censures, taken against members of the U.S. Congress?"

            ---

            #### **Example 3**
            **Input:**
            "text": "Convoy HX 212\nConvoy HX 212 was a World War II transatlantic convoy of merchant ships traveling from Halifax to Liverpool. It was escorted by the Mid-Ocean Escort Force Group A-3 and was attacked by German U-boats. The convoy system was essential in protecting merchant ships from U-boat attacks during the Battle of the Atlantic."
            "related": ["Convoy Battles of World War II"]

            **Output:**
            "What series of naval engagements during World War II involved merchant ship convoys and U-boat attacks, similar to Convoy HX 212?"

            ---

            Now, generate a question based on the following input:

            "text": "{input_text}"
            "related": {related_titles}
            """
            }
        ]
        return messages

    def _formatted_predicted_labels(self, predicted_labels: str):
        return "\n".join([f"{i+1}> '{key}': '{value}'," for i, (key, value) in enumerate(predicted_labels.items())])

    def _generate_ranking_messages(self, index: int, description: str, qsummary:str, predicted_labels: dict, k:int = 5):
        """
        Generate messages for ranking Wikipedia article titles based on a description and predicted labels.
        
        Args:
            description (str): A textual description.
            predicted_labels (dict): A dictionary of predicted labels.
            
        Returns:
            list: A list of message dictionaries to be used in a ranking task.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI model that ranks the most relevant Wikipedia article titles based on a given question. "
                    f"Your goal is to analyze the question, compare it with the given predicted labels, and select the top {k} most similar titles. "
                    "The ranking should be in decreasing order of relevance."
                )
            },
            {
                "role": "user",
                "content": f"""Given the following input:

                ### **Item Index:**
                "{index}"

                ### **Question:**
                "{qsummary if qsummary is not None else description}"

                ### **Predicted Labels:**
                {self._formatted_predicted_labels(predicted_labels)}

                ### **Instructions:**
                1. Extract **the top {k} most relevant titles** from the predicted labels list in **python dictionary format (example provided below)**.
                2. Make sure to **always give top {k}, not less**.
                3. Rank them in **decreasing order of relevance** (most relevant first).
                4. The relevance should be based on **semantic similarity to the question** and your **knowledge about Wikipedia articles**.
                5. **Return the output as a dictionary**, where:
                - The **key** is the `label_uid` from the predicted labels. Make sure the `label_uid` is the one form the predicted labels list. **No Hallucination**.
                - The **value** is the corresponding `label_title`. Make sure the `label_title` is the one corresponding to the `label_uid`. **No Hallucination**.

                ### **Example Question:**
                What archaeological site along Hadrian's Wall features a milecastle and associated turrets, and is part of a significant landscape designated for its ecological importance?

                ### **Expected Output Format (Example)**
                ```python
                {{
                    "label_uid_129733": "Allolee_to_Walltown_SSSI",
                    "label_uid_45362": "Hadrian's Wall",
                    "label_uid_156320": "Trajan's Wall",
                    "label_uid_90388": "Chesters Hill Fort",
                    "label_uid_73808": "Ridge turret",
                }}
                ```
                """ 
            }, 
        ] 
        return messages

    @staticmethod
    def write_api_requests_to_multiple_files(api_requests, api_fname, max_records_per_file=10000):
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(api_requests)

        # Calculate the number of files needed
        num_files = np.ceil(len(df) / max_records_per_file).astype(int)

        # Split the DataFrame into chunks
        chunks = np.array_split(df, num_files)

        # Loop through the chunks and write each to a separate file, named based on api_fname
        for i, chunk in enumerate(chunks):
            # Construct file name with sequential numbering
            filename = f"{api_fname.parent/api_fname.name.split('.')[0]}_part{i+1}.jsonl"
            chunk.to_json(filename, orient='records', lines=True)
            print(f"The API requests have been written to the file {filename}")

    def create_jsonl_file_batching(self, api_fname, topic_model, 
                documents:pd.DataFrame, topk_lbs_cutoff=100, 
                prompt_rerank_predicted_labels_col='predicted_labels',
                prompt_rerank_predicted_scores_col='predicted_scores',
                max_records_per_file=10000,
                max_doc_length_chatgpt_ranking=None):
        doc_generator = self._process_documents(documents, topic_model, topk_lbs_cutoff, max_doc_length_chatgpt_ranking, tokenizer='whitespace',
                                                labels_col=prompt_rerank_predicted_labels_col,
                                                scores_col=prompt_rerank_predicted_scores_col)
        api_requests = []  # This will store all data to be written to JSONL
        for doc in tqdm(doc_generator, total=len(documents), desc="Writing Messages for API request"):
            index, messages = self._create_api_request_msg(doc)
            api_requests.append({
                "metadata": {
                    "row_id": f"request-{index}"
                },
                # "method": "POST",
                # "url": "/v1/chat/completions",
                # "body": {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8000,
                # "custom_id": f"request-{index}"
                # }
            })
        self.write_api_requests_to_multiple_files(api_requests, api_fname, max_records_per_file=max_records_per_file)
    
    def create_jsonl_file_summarization(self, api_fname, documents):
        api_requests = []
        for row in documents.itertuples(index=False):
            # messages = self._create_api_request_summary(row.text)
            # import pdb; pdb.set_trace()
            messages = self._generate_chatgpt_messages(row.text, L(row.labels.keys()))
            api_requests.append({
                "metadata": {
                    "_id": f"{row._0}"
                },
                "model": self.model,
                "messages": messages,
                "max_tokens": 8000,
            })
        self.write_api_requests_to_multiple_files(api_requests, api_fname, max_records_per_file=20000)

    def _create_reranking_prompt(self, index, description, labels):
        PROMPT = """
        This is a product description on Amazon.com for [ITEM_IDX]: 
        [DESCRIPTION] 
        The following is a dictionary with product_id as key and product_description as value:
        [LABELS]
        --------------------------
        """ 
        prompt = PROMPT.replace("[LABELS]", str(labels)).replace("[DESCRIPTION]", description).replace("[ITEM_IDX]", f"ITEM {index}")
        return prompt
        
def chat_completions_with_backoff(client, **kwargs):
    return retry_with_exponential_backoff(
        client.chat.completions.create,
        errors=(openai.RateLimitError,),
    )(**kwargs)

@contextmanager
def temp_attribute(obj, attr_name, new_value):
    # Save the original value
    original_value = getattr(obj, attr_name)
    
    # Set the new value
    setattr(obj, attr_name, new_value)
    
    try:
        yield
    finally:
        # Restore the original value
        setattr(obj, attr_name, original_value)

def simulate_progress(duration):
    """ Simulates a progress bar for the given duration """
    with tqdm(total=100, desc="Performing Search") as pbar:
        for i in range(100):
            time.sleep(duration / 100)  # sleep in small increments to fill up the bar
            pbar.update(1)

# Function definitions
def get_dataset(name, split=None, split_sz=None):
    """
    Process data based on command-line arguments.
    
    Parameters:
    - args (Namespace): Parsed command-line arguments.
    
    Returns:
    - None
    """
    dset = load_dataset(name, split=split if split is not None else 'train')
    splt_sz = split_sz if split_sz is not None else len(dset)
    titles = dset['title'][:splt_sz]
    contents = dset['content'][:splt_sz]
    docs = [f'{t} \n {c}' for t,c in  zip(titles, contents)]
    # docs = [sent_tokenize(doc) for doc in docs]
    # docs = [sentence for doc in docs for sentence in doc]
    return docs

def topicfy_data(file_path, topic_model, topicfied_file_name, doc_embeddings, base_dir="../tmp", split_sz=None):
    """
    Processes a file to assign topics to each data entry, saves the topicfied data to a new file, 
    and performs similarity calculations between provided query and corpus embeddings.

    Parameters
    ----------
    file_path : str
        The path to the file containing the data to be topicfied. This file should be in a JSON format
        where each line is a separate JSON object.
    topic_model : object
        A model object that has a `find_topics` method which accepts text data and returns topics.
    topicfied_file_name : str
        The name of the file to save the processed data to. This file will be stored in the directory
        specified by `base_dir`.
    base_dir : str, optional
        The base directory where the `topicfied_file_name` will be saved. Default is "../tmp".

    Returns
    -------
    None
        This function does not return a value. It saves the topicfied data to a file and may output logs or 
        other information during processing.

    Notes
    -----
    The function assumes that `topic_model.find_topics` can handle the scale of data provided and
    that the embeddings are pre-computed and properly aligned with the data in `file_path`.

    Example
    -------
    >>> topic_model = SomeTopicModel()
    >>> topicfy_data("data.json", topic_model, "processed_data.json")
    """

    base_dir = Path(base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    df_topics = topic_model.get_topic_info()
    df_topics.to_csv(base_dir/'topic_info.csv', index=False)
    logging.info(f"The detailed topics information are written to the file to {base_dir/'topic_info.csv'}.")
    
    topics_map = get_topicsmap(df_topics)
    topics_file = base_dir/'topics.csv'
    df_topic_dict = pd.DataFrame(list(topics_map.items()), columns=['Topic ID', 'Topic Description'])
    df_topic_dict.to_csv(topics_file, index=False)
    logging.info(f"There are {len(topics_map)} topics. The topics are written to the file {topics_file}.")

    # open the train file and cluster the data point based on the topic model
    logging.info(f"Opening the train file with split size {slice(0, split_sz)}.")
    df_train = pd.read_json(file_path, lines=True).iloc[slice(0, split_sz)]
    docs = (df_train['title'] + " \n " + df_train['content']).values.tolist()  
    with temp_attribute(topic_model, 'hdbscan_model', BaseCluster()): # hack to use cosine similarity and efficient transformation
        df_train['topics'], _ = topic_model.transform(docs, doc_embeddings)
    df_train['topic_description'] = df_train['topics'].map(topics_map)
    df_train['split'] = 'train'
    df_train.to_csv(base_dir/topicfied_file_name, index=False)
    logging.info(f"The topicfied data is written to the file {base_dir/topicfied_file_name}.")
    try: 
        outlier_count = df_train['topics'].value_counts()[-1]
        logging.info(f"There are {outlier_count} number of outliers, out of total {len(df_train)} many documents.")
        logging.info(f"Outlier Percentage: {outlier_count/ len(df_train)}") 
    except KeyError: 
        logging.info(f"There are no outliers in the training set.")

def get_labels(labels_file):
    # First, count the total number of lines
    with open(labels_file, 'r') as infile:
        total_lines = sum(1 for line in infile)

    # Create an empty dictionary to store label descriptions
    labels = []
    # Load the lbl.json file
    with open(labels_file, 'r') as infile:
        for line in tqdm(infile, unit='records', total=total_lines, desc="Reading Labels"):
            record = json.loads(line)
            uid = record['uid']
            labels.append(record['title'])  # Change this to whatever description field you want to extract

    return labels

def get_label_uid_to_title_dict(labels_file):
    df = pd.read_json(labels_file, lines=True)
    uid_to_title_dict = df.set_index('uid')['title'].to_dict()
    return uid_to_title_dict

def get_label_idx_to_uid_dict(labels_file):
    df = pd.read_json(labels_file, lines=True)
    idx_to_uid_dict = df['uid'].to_dict()
    return idx_to_uid_dict

# A custom function to merge dictionaries
def merge_label_dicts(series):
    result = {}
    for d in series:
        result.update(d)
    return result  # Return the merged dictionary

def move_embeddings_to_device(doc_embeddings, label_embeddings, device="cpu"):
    """
    Moves embeddings between numpy arrays and PyTorch tensors and places them on the specified device.

    Args:
        doc_embeddings (numpy.ndarray or torch.Tensor): Document embeddings.
        label_embeddings (numpy.ndarray or torch.Tensor): Label embeddings.
        device (str): Target device, either "cpu" or "cuda".

    Returns:
        tuple: A tuple containing the embeddings on the specified device.
    """
    if device not in ["cpu", "cuda"]:
        raise ValueError("Invalid device. Choose either 'cpu' or 'cuda'.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system. Please select 'cpu'.")

    logging.info(f"Moving embeddings to {device}...")

    def convert_to_device(embeddings):
        # If already a tensor, move it to the target device
        if isinstance(embeddings, torch.Tensor):
            return embeddings.to(device)
        # If numpy array, convert to tensor and move to the target device
        elif isinstance(embeddings, np.ndarray):
            return torch.tensor(embeddings).to(device)
        else:
            raise TypeError("Embeddings must be either numpy.ndarray or torch.Tensor.")

    def convert_to_numpy(embeddings):
        # Convert tensor to numpy if it's on a device
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu().numpy()
        elif isinstance(embeddings, np.ndarray):
            return embeddings
        else:
            raise TypeError("Embeddings must be either numpy.ndarray or torch.Tensor.")

    if device == "cpu":
        # Ensure the embeddings are converted to numpy arrays if on CPU
        doc_embeddings_device = convert_to_numpy(doc_embeddings)
        label_embeddings_device = convert_to_numpy(label_embeddings)
    else:
        # Move or convert the embeddings to the specified device
        doc_embeddings_device = convert_to_device(doc_embeddings)
        label_embeddings_device = convert_to_device(label_embeddings)

    logging.info(f"Embeddings successfully moved to {device}.")
    return doc_embeddings_device, label_embeddings_device

def label_data(topic_model, clustered_file, labelfile, doc_embeddings, label_embeddings, top_k=10):
    """
    Apply a topic modeling algorithm to the data in the specified file to label it with topics.

    Parameters:
    -----------
    topic_model : object
        An instance of a topic modeling class, equipped with methods to fit data and predict topics.
        This model should be already trained or capable of training on the fly.
    
    clustered_file : str
        The path to the CSV file containing data to be labeled. The file should be readable and
        formatted correctly according to the expectations of the `topic_model`.

    Returns:
    --------
    DataFrame
        A Pandas DataFrame containing the original data along with a new column 'Topic' which
        has the topic labels assigned to each entry based on the topic model.

    Raises:
    -------
    FileNotFoundError
        If the `clustered_file` does not exist at the specified path.

    ValueError
        If the file is not in the expected format or if there are issues with the data that
        prevent the topic model from processing it.

    Example:
    --------
    >>> topic_model = BERTopic()
    >>> labeled_df = label_data(topic_model, 'data/clustered_data.csv')
    >>> print(labeled_df.head())
    """
    # Load the topicfied data
    try:
        logging.info(f"Opening the clustered file {clustered_file}")
        data = pd.read_csv(clustered_file, usecols=['uid','title', 'content', 'target_ind', 'target_rel', 'topics', 'topic_description', 'split'])
        logging.info(f"The number of documents needed to be pseudo-labelled is {len(data)}.")
        # docs = list(map( str, (data['title'] + " \n " + data['content']).values.tolist() ))
        label_uid_to_title = get_label_uid_to_title_dict(labelfile)
        # label_idx_to_uid = {idx: uid for idx, uid in enumerate(label_uid_to_title)}
        label_idx_to_uid = get_label_idx_to_uid_dict(labelfile) # use this if sometimes label uids are duplicate
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {clustered_file} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or not in expected CSV format.")
    
    # Apply the topic model to the data
    try:
        labels_instruction = "Represent the Amazon product title for retrieval: "
        query_instruction = "Represent the Amazon product description for retrieving similar product titles: "

        # labels_embeddings = model.encode(labels, prompt=labels_instruction)
        # label_embeddings = topic_model._extract_embeddings(labels, images=None, method="document", verbose=True)

        # query_embeddings = model.encode(docs, prompt=query_instruction)
        # query_embeddings = model.encode(docs[:2000], batch_size=128, prompt=query_instruction)
        # query_embeddings = topic_model._extract_embeddings(docs, images=None, method="document", verbose=True)

        # Start the simulated progress bar in a separate thread
        duration_estimate = 230 # Estimated time for `semantic_search` to complete
        thread = threading.Thread(target=simulate_progress, args=(duration_estimate,))
        # thread.start()  

        start_time = time.time()  # Start timing
        # Perform the semantic search while the progress bar runs
        doc_embeddings, label_embeddings = move_embeddings_to_device(doc_embeddings, label_embeddings, device='cuda') # move to GPU to speed semantic search
        logging.info("Performing Semantic search...")
        hits = util.semantic_search(doc_embeddings, label_embeddings, top_k=top_k)
        end_time = time.time()  # End timing
        print(f"Semantic search completed in {end_time - start_time:.2f} seconds")

        # Wait for the simulated progress to complete if the search finishes first
        # thread.join()

        data['predicted_labels'] = [ {label_uid: label_uid_to_title[label_uid] for label_uid in map(lambda d: label_idx_to_uid[d['corpus_id']], hit)} for hit in hits ]
        # Convert string representations of dictionaries to actual dictionaries
        # data['predicted_labels'] = data['predicted_labels'].apply(ast.literal_eval)

        ## deb making changes: topicfying labels (later move this to topicfy_data)
        topicfied_labelfile = pd.read_json(labelfile, lines=True)
        # Getting the topics of the labels suing the topic_model
        doc_embeddings, label_embeddings = move_embeddings_to_device(doc_embeddings, label_embeddings, device='cpu') # move back to CPU to do topicfy labels
        topicfied_labelfile['label_topics'], label_probs = topic_model.transform(topicfied_labelfile.title.values, label_embeddings)
        topicfied_labelfile['topic_description'] = topicfied_labelfile['label_topics'].map(topic_model.topic_labels_)
        topicfied_labelfile['label_dict'] = topicfied_labelfile.apply(lambda row: {f"{row['uid']}": row['title']}, axis=1)
        topicfied_label_fname = clustered_file.parent/'topicfied_labels.csv'
        topicfied_labelfile.to_csv(topicfied_label_fname, index=False)
        logging.info(f"The topicfied labels are written to the file {topicfied_label_fname}.")

        # Group the topicfied_labels by 'topics' and aggregate
        aggregated_labels = topicfied_labelfile.groupby('label_topics').agg({
            'label_dict': merge_label_dicts,  # Apply merging function to the predicted_labels
            'uid': 'count'  # Count the number of entries in each group
        }).reset_index().rename(columns={'uid': 'label_set_size', 'label_dict': 'label_set'})
        label_cluster_fname_by_topicfying_labels = clustered_file.parent/clustered_file.name.replace('clustered', 'labels_for_each_cluster_by_topicfying_labels')
        aggregated_labels.to_csv(label_cluster_fname_by_topicfying_labels, index=False)
        logging.info(f"The label set for each topic cluster of the labels is written to the file {label_cluster_fname_by_topicfying_labels}.")
        ## end deb making changes

        # Group data by 'topics' and aggregate
        aggregated_data = data.groupby('topics').agg({
            'predicted_labels': merge_label_dicts,  # Apply merging function to the predicted_labels
            'uid': 'count'  # Count the number of entries in each group
        }).reset_index().rename(columns={'uid': 'count', 'predicted_labels': 'predicted_label_set'})

        # Now calculate the size of each merged dictionary (that is the label set for each group)
        aggregated_data['label_set_size'] = aggregated_data['predicted_label_set'].apply(len)

        ## deb making changes
        aggregated_data = aggregated_data.merge(aggregated_labels, left_on='topics', right_on='label_topics', how='outer')
        # merge keys should be unified
        aggregated_data['topics'] = aggregated_data['topics'].fillna(aggregated_data['label_topics'])
        aggregated_data['label_topics'] = aggregated_data['label_topics'].fillna(aggregated_data['topics'])
        # Fill Missing Values
        aggregated_data['predicted_label_set'] = aggregated_data['predicted_label_set'].fillna(aggregated_data['label_set'])
        aggregated_data['label_set'] = aggregated_data['label_set'].fillna(aggregated_data['predicted_label_set'])
        aggregated_data['label_set_size_x'] = aggregated_data['label_set_size_x'].fillna(aggregated_data['label_set_size_y'])
        aggregated_data['label_set_size_y'] = aggregated_data['label_set_size_y'].fillna(aggregated_data['label_set_size_x'])
        # Create a new column by merging dictionaries from 'predicted_label_set' and 'label_set'
        aggregated_data['label_set_union'] = aggregated_data.apply(lambda row: {**row['predicted_label_set'], **row['label_set']}, axis=1)
        logging.info(f"The number of NaNs in the labels_for_each_topic_cluster is {aggregated_data.isna().any(axis=1).sum()}")
        ## end deb making changes

        labelled_filename = clustered_file.parent/clustered_file.name.replace('clustered', 'clustered_pseudo_labelled')
        label_cluster_filename = clustered_file.parent/clustered_file.name.replace('clustered', 'labels_for_each_cluster')

        data.to_csv(labelled_filename, index=False)
        logging.info(f"The pseudo labelled data is written to the file {labelled_filename}.")

        aggregated_data.to_csv(label_cluster_filename, index=False)
        logging.info(f"The label set for each topic cluster is written to the file {label_cluster_filename}.")

    except Exception as e:
        raise ValueError(f"An error occurred while applying the topic model: {e}")

    return None

def generate_queries_from_corpus(df_queries_corpus, summarizer_model="Falconsai/text_summarization", batch_size=8,  summarize=False, max_content_length=500, training_mode=True):
    """
    Converts a DataFrame with 'uid', 'title', and 'content' columns into a query-based format,
    generating a new DataFrame with '_id', 'text', and 'metadata', with batch processing for summarization.
    
    Args:
        df_queries_corpus (pd.DataFrame): Input DataFrame containing 'uid', 'title', and 'content'.
        summarizer_model (str): Hugging Face model name for summarization.
        batch_size (int): Number of entries to process in a single batch for summarization.
        max_content_length (int): Maximum number of words for content; truncate if exceeded.
        
    Returns:
        pd.DataFrame: Transformed DataFrame with '_id', 'text', and 'metadata'.
    """
    # Fill missing values
    df_queries_corpus[['title', 'content']] = df_queries_corpus[['title', 'content']].fillna("")
    
    # Create a base DataFrame with required columns
    df = df_queries_corpus[['uid', 'title', 'content', 'labels']]
    df['metadata'] = [{}] * len(df)

    # Convert titles and contents to string lists
    grouped_titles = df['title'].astype(str).tolist()
    grouped_contents = [
        " ".join(content.split()[:max_content_length])  # Truncate to max_content_length words
        for content in df['content'].astype(str).tolist()
    ]

    # import pdb; pdb.set_trace()

    # Quick iteration <remove later>
    # logging.info(f"Original length was {len(grouped_contents)}.")
    # df = df.iloc[:100]
    # grouped_titles = grouped_titles[:100]
    # grouped_contents = grouped_contents[:100]
    # logging.info(f"For debugging I will only consider the first {len(grouped_contents)}")
    # Quick iteration <remove later>
    
    # Initialize the summarizer pipeline
    summarizer = pipeline("summarization", model=summarizer_model, device=0)
    
    # Summarize the contents in batches
    summarized_grouped_contents = []
    logging.info(f"Converting the dataset (size: {len(grouped_contents)}) into queries using {summarizer_model}...")
    
    if summarize:
        # Start the timer
        logging.info(f"Starting the summarization...")
        start_time = time.time()
        for i in tqdm(range(0, len(grouped_contents), batch_size), total= np.ceil(len(grouped_contents)/batch_size), desc=f"Summarizing Contents in batches(size: {batch_size})"):
            batch = grouped_contents[i:i + batch_size]
            summaries = summarizer(batch, max_length=200, min_length=30, do_sample=False, batch_size=batch_size)
            summarized_grouped_contents.extend(summaries)
        end_time = time.time()
        time_taken = end_time - start_time
        logging.info(f"Time taken for summarization: {time_taken:.2f} seconds")
    
    # Generate the queries
    if summarize:
        logging.info("Summarization is enabled. Using summarized contents.")
        # queries = [
        #     f"What is a related product for item ***{title}*** whose description is similar to ***{summary['summary_text']}***?"
        #     for title, summary in zip(grouped_titles, summarized_grouped_contents)
        # ]
        queries = [
            f"The description of a Wikipedia article titled ***{title}*** is ***{summary['summary_text']}***. What are some related Wikipedia titles?"
            for title, summary in zip(grouped_titles, summarized_grouped_contents)
        ]
    else:
        logging.info("Summarization is disabled. Using grouped contents.")
        # queries = [
        #     f"What is a related product for item ***{title}*** whose description is similar to ***{content}***?"
        #     for title, content in zip(grouped_titles, grouped_contents)
        # ]
        queries = [
            f"{title}\n{content}"
            for title, content in zip(grouped_titles, grouped_contents)
        ]
    
    # Add the queries to the DataFrame
    df['text'] = queries
    
    # Rename 'uid' to '_id' and retain only necessary columns
    df = df.rename(columns={'uid': '_id'})[['_id', 'text', 'metadata', 'labels']]

    # ChatGPT Summarization API requests
    openai_model = get_openai_model()
    if openai_model:
        api_fname = Path(base_dir) / (f'{"train_" if training_mode else "test_"}'+"summarization_requests.jsonl") 
        openai_model.create_jsonl_file_summarization(api_fname=api_fname, documents=df)
    # ChatGPT Summarization API requests
    
    return df

def create_jsonl(lbs_corpus, df_queries_corpus, df_queries_corpus_test, base, source, num_pos=3, label_frac_sup=0,
                generator_name_or_path="BeIR/query-gen-msmarco-t5-base-v1",
                summarize=False,
                max_content_length=500):
    # Creating corpus.jsonl
    df_lbs = pd.DataFrame({
        "_id": lbs_corpus.keys(),
        "title": [""] * len(lbs_corpus),  # Empty title for each record
        "text": lbs_corpus.values(),
        "metadata": [{}] * len(lbs_corpus)  # Empty metadata for each record
    })
    jsonl_corpus_path = os.path.join(base, 'corpus.jsonl')
    df_lbs.to_json(jsonl_corpus_path, orient='records', lines=True)
    logging.info(f"Corpus created at {jsonl_corpus_path}")

    # Creating qgen-queries.jsonl (for training)
    # df_queries_corpus = df_queries_corpus.sample(167, random_state=8990).reset_index(drop=True)
    jsonl_queries_path = Path(base) / 'qgen-queries.jsonl'
    if jsonl_queries_path.exists():
        creation_time = os.path.getctime(jsonl_queries_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Using the existing train queries at {jsonl_queries_path}. File created on: {creation_date}")
    else:
        logging.info(f"No train queries found at {base}. Creating now...")
        processed_df_train = generate_queries_from_corpus(df_queries_corpus, summarizer_model="sshleifer/distilbart-cnn-12-6", batch_size=8, summarize=summarize, max_content_length=max_content_length, training_mode=True)
        processed_df_train.to_json(jsonl_queries_path, orient='records', lines=True)
        logging.info(f"Train Queries created at {jsonl_queries_path}")

    # Prepare the records for the qgen-qrels/train.tsv file
    try:
        with open(f"{source}/train_uids_{label_frac_sup}.txt", "r") as file:
            train_sup_ids = [line.strip() for line in file.readlines()]
            logging.info(f"Supervision fraction is {label_frac_sup}. Loading supervision training ids from 'train_uids_{label_frac_sup}.txt' in {source}. Will Train with supervision on these {len(train_sup_ids)} training examples.")
    except FileNotFoundError:
        logging.warning(f"Warning: File 'train_uids_{label_frac_sup}.txt' does not exist in {source}. Will Train without supervision.")
        train_sup_ids = []
    
    df_queries_corpus['predicted_labels'] = df_queries_corpus['predicted_labels'].apply(ast.literal_eval)
    records = []
    for index, row in tqdm(df_queries_corpus.iterrows(), total=len(df_queries_corpus), desc="Preparing qgen-qrels"):
        query_id = row['uid']
        corpus_ids = L(row['labels'].keys()) if query_id in train_sup_ids else L() # if we have supervision on this training id
        # Get the num_pos more corpus_ids from predicted_labels
        more_corpus_ids = L(row['predicted_labels'].keys())[:num_pos]
        filtered_more_corpus_ids = [item for item in more_corpus_ids if item not in corpus_ids]
        corpus_ids.extend(filtered_more_corpus_ids) 
        for corpus_id in corpus_ids:
            records.append({
                'query_id': query_id,
                'corpus_id': corpus_id,
                'score': 1
            })
    train_df = pd.DataFrame(records)
    qgen_qrels_dir = base/'qgen-qrels'
    qgen_qrels_dir.mkdir(exist_ok=True, parents=True)
    tsv_file_path = os.path.join(qgen_qrels_dir, 'train.tsv')
    train_df.to_csv(tsv_file_path, sep='\t', index=False)
    logging.info(f"qgen-qrels created at {tsv_file_path}")

    # Creating queries.jsonl (for testing)
    # df_queries_corpus_test = df_queries_corpus_test.sample(157, random_state=8990).reset_index(drop=True)
    jsonl_test_queries_path = Path(base) / 'queries.jsonl'
    if jsonl_test_queries_path.exists():
        creation_time = os.path.getctime(jsonl_test_queries_path)
        creation_date = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Using the existing test queries at {jsonl_queries_path}. File created on: {creation_date}")
    else:
        logging.info(f"No test queries found at {base}. Creating now...")
        processed_df_test = generate_queries_from_corpus(df_queries_corpus_test, summarizer_model="sshleifer/distilbart-cnn-12-6", batch_size=8, summarize=summarize, training_mode=False)
        processed_df_test.to_json(jsonl_test_queries_path, orient='records', lines=True)
        logging.info(f"Test Queries created at {jsonl_test_queries_path}")

    # Prepare the records for the qrels/test.tsv file
    test_records = []
    # df_queries_corpus_test['labels'] = df_queries_corpus_test['labels'].apply(ast.literal_eval)
    for index, row in tqdm(df_queries_corpus_test.iterrows(), total=len(df_queries_corpus_test), desc="Preparing qrels"):
        query_id = row['uid']
        # Get the corpus ids from the `labels` field
        corpus_ids = row['labels'].keys()
        for corpus_id in corpus_ids:
            test_records.append({
                'query_id': query_id,
                'corpus_id': corpus_id,
                'score': 1
            })
    test_df = pd.DataFrame(test_records)
    qrels_dir = base/'qrels'
    qrels_dir.mkdir(exist_ok=True, parents=True)
    test_tsv_file_path = os.path.join(qrels_dir, 'test.tsv')
    test_df.to_csv(test_tsv_file_path, sep='\t', index=False)
    logging.info(f"qrels created at {test_tsv_file_path}")

    # return jsonl_corpus_path, jsonl_test_queries_path, tsv_file_path, json_test_queries_path, test_tsv_file_path

def prepare_dataset_for_margin_mse_loss(pseudo_labelled_trnfile, train_test_file ,label_file, num_pos=3, label_frac_sup=0, summarize=False,
                                        max_content_length=500):
    try:
        logging.info(f"Opening the pseudo-labelled file {pseudo_labelled_trnfile} to prepare train data in BeIR format")
        queries_corpus_train = pd.read_csv(pseudo_labelled_trnfile, usecols=['uid','title', 'content', 'target_ind', 'target_rel', 'topics', 'topic_description', 'split', 'predicted_labels'])
        logging.info(f"The number of documents in the train set that are pseudo-labelled is {len(queries_corpus_train)}.")

        logging.info(f"Opening the file {train_test_file} to read the test examples for preparing the test data in BeIR format.")
        queries_corpus_test = pd.read_csv(train_test_file)
        queries_corpus_test = queries_corpus_test[queries_corpus_test['split'] == 'test'].reset_index(drop=True)
        queries_corpus_test['labels'] = queries_corpus_test['labels'].apply(ast.literal_eval)

        queries_corpus_train_actual = pd.read_csv(train_test_file)
        queries_corpus_train_actual = queries_corpus_train_actual[queries_corpus_train_actual['split'] == 'train'].reset_index(drop=True)

        queries_corpus_train = pd.merge(
            queries_corpus_train,
            queries_corpus_train_actual[['uid', 'labels']],
            on='uid',
            how='left'  # Use 'left' join to keep all rows from df_queries_corpus
        )
        assert not queries_corpus_train['labels'].isna().any() # make we have all the labels
        queries_corpus_train['labels'] = queries_corpus_train['labels'].apply(ast.literal_eval)

        lbs_corpus = get_label_uid_to_title_dict(label_file)
        # label_idx_to_uid = {idx: uid for idx, uid in enumerate(lbs_corpus)}
        dir_base = pseudo_labelled_trnfile.parent
        dir_source = train_test_file.parent
        create_jsonl(lbs_corpus, queries_corpus_train, queries_corpus_test, dir_base, dir_source, num_pos=num_pos, label_frac_sup=label_frac_sup, summarize=summarize, max_content_length=max_content_length)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {pseudo_labelled_trnfile} or {train_test_file} or {label_file} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or not in expected CSV format.")

def get_qrels_results(df, cross_rerank=False):
    if not _is_column_all_dicts(df, col='labels'): df['labels'] = df['labels'].apply(ast.literal_eval)
    qrels = {}
    results={}
    results_reranked={}
    for index, row in df.iterrows():
        uid = row['uid']

        label_ids = row['labels'].keys()
        qrels[uid] = {label_id: 1 for label_id in label_ids}

        label_id_to_score_dict = row['predicted_scores']
        results[uid] = {label_id: score for label_id, score in label_id_to_score_dict.items()}

        if cross_rerank:
            label_id_to_score_dict_reranked = row['predicted_scores_reranked']
            results_reranked[uid] = {label_id: float(score) for label_id, score in label_id_to_score_dict_reranked.items()}
    return qrels, results, results_reranked

def _is_column_all_dicts(dataframe, col):
    """A function to check if all elements in a DataFrame column are dictionaries."""
    if col in dataframe.columns:
        return dataframe[col].apply(lambda x: isinstance(x, dict)).all()
    else:
        raise ValueError(f"Column {col} does not exist in the DataFrame.")

def get_qrels_results_agg(df, colname='agg_predicted_scores'):
    if not _is_column_all_dicts(df, 'labels'): df['labels'] = df['labels'].apply(ast.literal_eval)

    qrels = {}
    results={}
    for index, row in df.iterrows():
        uid = row['uid']

        label_ids = row['labels'].keys()
        qrels[uid] = {label_id: 1 for label_id in label_ids}

        label_id_to_score_dict = row[colname]
        results[uid] = {label_id: score for label_id, score in label_id_to_score_dict.items()}

    return qrels, results

def convert_test_data_to_queries(df_test, 
                                 generator_name_or_path="BeIR/query-gen-msmarco-t5-base-v1",
                                 ques_per_passage=3,
                                 batch_size_generation=32,
                                 qgen_prefix='testqgen'):
    # Creating queries.jsonl (for testing)
    # Replace NaN values in 'title' and 'content' columns with empty strings
    df_test[['title', 'content']] = df_test[['title', 'content']].fillna("")
    df = df_test[['uid', 'title', 'content']].rename(columns={'uid': '_id', 'content': 'text'})

    # remove later 
    # df = df.sample(100, random_state=8990)#.reset_index(drop=True) 
    # remove later 

    test_set_dir = os.path.join(base_dir, 'test_set')
    os.makedirs(test_set_dir, exist_ok=True)
    jsonl_test_queries_path = os.path.join(test_set_dir, 'corpus.jsonl')
    df.to_json(jsonl_test_queries_path, orient='records', lines=True)
    logging.info(f"Test Set in beIR format created at {jsonl_test_queries_path}")

    qgen(
            test_set_dir,
            test_set_dir,
            generator_name_or_path=generator_name_or_path,
            ques_per_passage=ques_per_passage,
            bsz=batch_size_generation,
            qgen_prefix=qgen_prefix,
        )
    logging.info("The test set has been converted into queries at directory {test_set_dir}.")
    corpus, gen_queries, gen_qrels = GenericDataLoader(test_set_dir, prefix=qgen_prefix).load(split="train")

def update_prompt_rerank_cols(df, prompt_rerank_predicted_labels_col, prompt_rerank_predicted_scores_col, label_uids):
    """
    Update the dictionaries in the specified column so that keys are replaced with new generic UIDs,
    using the provided label_uids list. Also, save the mapping in CSV format at base_dir.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the column.
        prompt_rerank_predicted_scores_col (str): Name of the column with dictionaries.
        label_uids (list): List of original label uids used to create the mapping.
        base_dir (str): Directory where the mapping CSV file will be saved.
        
    Returns:
        pd.DataFrame: DataFrame with the updated column.
    """
    # Create mapping from provided label_uids to generic uids.
    mapping = {uid: f"label_uid_{i+1}" for i, uid in enumerate(label_uids)}
    
    # Save mapping to CSV file.
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['old_label_uid', 'new_label_uid'])
    mapping_csv_path = os.path.join(base_dir, 'label_uid_mappings.csv')
    mapping_df.to_csv(mapping_csv_path, index=False)
    
    # Function to update keys in a single dictionary.
    def update_dict(d):
        return {mapping.get(k, k): v for k, v in d.items()}
    
    # Update the DataFrame column.
    df[prompt_rerank_predicted_scores_col] = df[prompt_rerank_predicted_scores_col].apply(update_dict)
    df[prompt_rerank_predicted_labels_col] = df[prompt_rerank_predicted_labels_col].apply(update_dict)
    
    return df, mapping

def make_predcitions(topic_model, train_test_file, label_cluster_file, label_file, topics_file, label_embeddings, top_k=10, prompt_rerank=False, cross_rerank=False,
                     prompt_rerank_topk_lbs_cutoff=100,
                     prompt_rerank_predicted_scores_col="predicted_scores",
                     prompt_rerank_predicted_labels_col='predicted_labels',
                     max_records_per_file = 10000,
                     topic_labels=True,
                     max_doc_length_chatgpt_ranking:int = None):
    # Load the testing data
    try:
        tst_predicted_path = os.path.join(base_dir, 'tst_predicted.csv')
        if os.path.exists(tst_predicted_path): # tst_predicted.csv exists we want to load that up 
            logging.info(f" The file {tst_predicted_path} already exists. We will load it up for making predictions.")
            df_test = pd.read_csv(tst_predicted_path)
        else:
            logging.info(f"Opening the file {train_test_file} to read the test examples for making predictions.")
            df_test = pd.read_csv(train_test_file)
            df_test = df_test[df_test['split'] == 'test'].reset_index(drop=True)

        # remove later 
        # df_test = df_test.sample(500, random_state=8990)#.reset_index(drop=True) 
        # remove later 
        logging.info(f"From the file {train_test_file}, we are testing on {len(df_test)} examples.")
        
        df_label_cluster = pd.read_csv(label_cluster_file)
        label_uid_to_title = get_label_uid_to_title_dict(label_file)

        # previously
            # label_idx_to_uid = {idx: uid for idx, uid in enumerate(label_uid_to_title)}
            # label_uid_to_idx = {uid: idx for idx, uid in enumerate(label_uid_to_title)}

        # now
        label_idx_to_uid = get_label_idx_to_uid_dict(label_file) # use this if sometimes label uids are duplicate
        label_uid_to_idx = {uid:idx for idx, uid in label_idx_to_uid.items()}

        df_topics = pd.read_csv(topics_file)
        if cross_rerank:
            logging.info(f"The bi-encoder will retrieve {top_k} documents. We use a cross-encoder, to re-rank the results list to improve the quality")
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2',)# default_activation_function=torch.nn.Sigmoid())
            summarizer = pipeline("summarization", model="Falconsai/text_summarization")
            # cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-large')# default_activation_function=torch.nn.Sigmoid())
            # cross_encoder = CrossEncoder('cross-encoder/qnli-electra-base',)# default_activation_function=torch.nn.Sigmoid())
        
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {train_test_file} or {label_cluster_file} or {topics_file} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or not in expected CSV format.")

    # Convert DataFrame to dictionary
    topics_map = pd.Series(df_topics['Topic Description'].values,index=df_topics['Topic ID']).to_dict()
    topic_to_label_set = pd.Series(df_label_cluster['label_set_union'].apply(ast.literal_eval).values, index=df_label_cluster['topics']).to_dict()

    # For each data point in the test set get the corresponding topic
    docs = list(map( str, (df_test['title'] + " \n " + df_test['content']).values.tolist() ))
    
    try:
        query_embeddings = load_test_embeddings(base_dir)
        logging.info(f"Test Document Embeddings found at {base_dir}.")
    except FileNotFoundError:
        logging.info(f"Test Document Embeddings not found at {base_dir}. Creating fresh embeddings.")
        query_embeddings = topic_model._extract_embeddings(docs, images=None, method="document", verbose=True)
        save_test_embeddings(query_embeddings, base_dir)
    
    # To make sure we can sure the saved query embeddings even when we are running on a sample of the test set
    query_embeddings_sampled = query_embeddings[df_test.index.to_list()]

    # deb added change to check full pipeline transformation
    # with temp_attribute(topic_model, 'hdbscan_model', BaseCluster()): # hack to use cosine similarity and efficient transformation
    df_test['topics'], _ = topic_model.transform(docs, query_embeddings if len(docs) == len(query_embeddings) else query_embeddings_sampled)
    df_test['topic_description'] = df_test['topics'].map(topics_map)

    # Now groupby 'topics' and for each `topic` group consider the label set from `topic_to_label_set[topic]`
    # and for the documents in that group perform a semantic search for labels over that label set  
    df_test['predicted_labels'] = df_test.apply(lambda x: "", axis=1)
    df_test['predicted_scores'] = df_test.apply(lambda x: "", axis=1)
    df_test['predicted_labels_reranked'] = df_test.apply(lambda x: "", axis=1)
    df_test['predicted_scores_reranked'] = df_test.apply(lambda x: "", axis=1)
    # addition made to read up the chatgpt created queries while training margin mseloss 
    chatgpt_test_queries_path = Path(base_dir)/'queries.jsonl'
    if cross_rerank and chatgpt_test_queries_path.exists():
        df_queries = pd.read_json(Path(base_dir)/"queries.jsonl", lines=True)
        df_test = df_test.merge(df_queries[['_id', 'text']], left_on='uid', right_on='_id', how='left')
        df_test.drop(columns=['_id'], inplace=True)
        assert not df_test['text'].isna().any()
        df_test['margin_mseloss_predicted_scores'] = df_test['margin_mseloss_predicted_scores'].apply(ast.literal_eval)
        df_test['margin_mseloss_predicted_scores'] =  df_test['margin_mseloss_predicted_scores'].apply(
            lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
        )
        df_test['margin_mseloss_predicted_labels'] = df_test['margin_mseloss_predicted_scores'].apply(
            lambda x: {key: label_uid_to_title.get(key, 'Unknown') for key in x.keys()}
        )
    # addition made to read up the chatgpt created queries while training margin mseloss
    grouped = df_test.groupby('topics')

    # import pdb; pdb.set_trace()
    query_embeddings, label_embeddings = move_embeddings_to_device(query_embeddings, label_embeddings, device='cuda') # move to GPU to speed semantic search
    if topic_labels:
        logging.info("Using labels of the clusters in making predictions.")
    else:
        logging.info("Not Using labels of the clusters in making predictions [Vanilla Full Corpus Semantic Search]")
    try:
        for topic, doc_idxs in tqdm(grouped.groups.items(), total=len(grouped), desc="Making Predictions"):
            # import pdb; pdb.set_trace()
            topic_query_embeddings = query_embeddings[doc_idxs] 
            
            if topic_labels and topic_to_label_set.get(topic, False): # when the at least one train set doc had that topic
                relevant_label_idxs = list(map(label_uid_to_idx.get, topic_to_label_set[topic].keys()))
                relevant_label_embeddings = label_embeddings[relevant_label_idxs] # change this for zero-shot
            else: 
                relevant_label_idxs = list(range(len(label_embeddings)))
                relevant_label_embeddings = label_embeddings # when no train set doc had that topic
            
            hits = util.semantic_search(topic_query_embeddings, relevant_label_embeddings, top_k=top_k) # we have one hit for each query
            predicted_labels, predicted_scores = process_hits(hits, label_idx_to_uid, relevant_label_idxs, label_uid_to_title)
            if cross_rerank: 
                # grouped_docs = list(map( str, (df_test.loc[doc_idxs.to_list()]['title'] + " \n " + df_test.loc[doc_idxs.to_list()]['content']).values.tolist() ))
                # grouped_docs = [doc+ " What is a related product description for this item?" for doc in grouped_docs]
                grouped_titles = list(map( str, (df_test.loc[doc_idxs.to_list()]['title']).values.tolist() ))
                grouped_contents = list(map( str, (df_test.loc[doc_idxs.to_list()]['content']).values.tolist() ))

                # Uncomment the following two lines and comment the next line if you want to do summarization
                # summarized_grouped_contents = summarizer(grouped_contents, max_length=200, min_length=30, do_sample=False)
                # grouped_docs = [f"What is a related product for item ***{title}*** whose description is similar to ***{summary['summary_text']}***?" for title, summary in zip(grouped_titles, summarized_grouped_contents)]
                # grouped_docs = [f"What is a related product for item ***{title}*** and description ***{content}***?" for title, content in zip(grouped_titles, grouped_contents)]
                grouped_docs = list(map( str, (df_test.loc[doc_idxs.to_list()]['text']).values.tolist() ))

                # crossrerank_hits(docs, doc_idxs, hits, predicted_labels, cross_encoder)
                margin_predicted_labels = df_test.loc[doc_idxs.to_list()]['margin_mseloss_predicted_labels'].values.tolist()
                margin_predicted_labels = [{lbl_id: lbl_title for lbl_id,lbl_title in itertools.islice(ml.items(), top_k)} for ml in margin_predicted_labels]
                margin_predicted_scores = df_test.loc[doc_idxs.to_list()]['margin_mseloss_predicted_scores'].values.tolist()
                margin_predicted_scores = [{lbl_id: lbl_score for lbl_id,lbl_score in itertools.islice(ml.items(), top_k)} for ml in margin_predicted_scores]

                def margin_hits(labels_list, scores_list):
                    hits = []
                    for labels_dict,scores_dict in zip(labels_list, scores_list):
                        hit = []
                        for lbl in labels_dict:
                            hit.append({'label_id': lbl, 'label_title': labels_dict[lbl], 'margin-score': scores_dict[lbl]})
                        hits.append(hit)
                    return hits
                
                mhits = margin_hits(margin_predicted_labels, margin_predicted_scores) # creates hits for the margin preds

                # crossrerank_hits(grouped_docs, hits, predicted_labels, cross_encoder) # old cross reranking
                # crossrerank_hits(grouped_docs, hits, predicted_labels, margin_predicted_scores, cross_encoder)
                crossrerank_hits(grouped_docs, mhits, margin_predicted_labels, cross_encoder) # adds a key 'cross-score' which is the cross-encoder scores on the margin preds
                # aggregate_ranked_lists(hits, col1='score', col2='cross-score')
                aggregate_ranked_lists(mhits, col1='margin-score', col2='cross-score') # adds a key 'agg-score' which is the sum of 'margin-score' and 'cross-score'
                predicted_labels_reranked, predicted_scores_reranked = process_crosshits(mhits, col='agg-score') # sorts labels and scores by 'agg-score'
                # import pdb; pdb.set_trace()
                # predicted_labels_reranked, predicted_scores_reranked = process_hits(hits, label_idx_to_uid, relevant_label_idxs, label_uid_to_title, cross_rerank=cross_rerank)
               
            # Create a boolean mask where only the indices in doc_idxs are True
            mask = df_test.index.isin(doc_idxs)
            # Use the mask to assign values
            df_test.loc[mask, 'predicted_labels'] = predicted_labels
            df_test.loc[mask, 'predicted_scores'] = predicted_scores
            if cross_rerank:
                df_test.loc[mask, 'predicted_labels_reranked'] = predicted_labels_reranked
                df_test.loc[mask, 'predicted_scores_reranked'] = predicted_scores_reranked
    except Exception as e:
        logging.error(f"Catched exception {e} inside `make_predictions`:")

    # hits = util.semantic_search(query_embeddings, label_embeddings, top_k=20)
    # predicted_labels = []
    # for hit in hits:  # Process each hit separately
    #     hit_labels = {}  # Dictionary to store label UIDs and their titles for this hit
    #     for record in hit:  # Each record corresponds to a document or item in the hit
    #         # Convert corpus_id to label_uid, and then get the label's title
    #         label_uid = label_idx_to_uid[record['corpus_id']]  
    #         label_title = label_uid_to_title[label_uid]
    #         hit_labels[label_uid] = label_title  # Map label UID to its title
    #     predicted_labels.append(hit_labels)  # Add the dictionary of labels for this hit to the list
    # df_test['predicted_labels'] = predicted_labels
    
    predicted_file = label_cluster_file.parent/'tst_predicted.csv'
    # Check if 'text' exists in df_test due to the cross encoding query joining, and if so, drop it
    if 'text' in df_test.columns:
        df_test.drop(columns=['text'], inplace=True)
    df_test.to_csv(predicted_file, index=False)
    logging.info(f"The predicted test data is written to the file {predicted_file}.")

    ## deb add changes
    # perform reranking
    if prompt_rerank:
        openai_model = get_openai_model()
        if openai_model:
            logging.info("Performing a prompt based reranking.")
            api_fname = predicted_file.parent/'api_requests.jsonl'
            if not _is_column_all_dicts(df_test, prompt_rerank_predicted_labels_col): df_test[prompt_rerank_predicted_labels_col] =  df_test[prompt_rerank_predicted_labels_col].apply(ast.literal_eval)
            if not _is_column_all_dicts(df_test, prompt_rerank_predicted_scores_col): df_test[prompt_rerank_predicted_scores_col] = df_test[prompt_rerank_predicted_scores_col].apply(ast.literal_eval)
            logging.info(f"Metric Computation of the column `{prompt_rerank_predicted_scores_col}` that will be used to prepare requests for CHATGPT Re-Ranking:")
            qrels, agg_results = get_qrels_results_agg(df_test, colname=prompt_rerank_predicted_scores_col)
            ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, agg_results, k_values=[1, 3, 5, 10, 20, 30, 40], ignore_identical_ids=True)
            df_test_copy = deepcopy(df_test)
            df_test_copy, label_uid_mapping = update_prompt_rerank_cols(df_test_copy, prompt_rerank_predicted_labels_col, prompt_rerank_predicted_scores_col, list(label_uid_to_title))
            test_summary_path = Path(base_dir)/'queries.jsonl'
            if test_summary_path:
                df_summary = pd.read_json(test_summary_path, lines=True)[['_id', 'text']].rename(columns={'text': 'qsummary'})
                df_test_copy = df_test_copy.merge(df_summary, left_on='uid', right_on='_id', how='left')
                df_test_copy.drop(columns=['_id'], inplace=True)
                nans_in_summary = df_test_copy['qsummary'].isna().sum()
                assert nans_in_summary == 0
            openai_model.create_jsonl_file_batching(api_fname, topic_model, df_test_copy, 
                                                    topk_lbs_cutoff=prompt_rerank_topk_lbs_cutoff, 
                                                    prompt_rerank_predicted_labels_col=prompt_rerank_predicted_labels_col,
                                                    prompt_rerank_predicted_scores_col=prompt_rerank_predicted_scores_col,
                                                    max_records_per_file=max_records_per_file,
                                                    max_doc_length_chatgpt_ranking=max_doc_length_chatgpt_ranking)
    ## end deb add changes

    # import pdb; pdb.set_trace()
    qrels, results, results_reranked = get_qrels_results(df_test, cross_rerank=cross_rerank)
    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=[1, 3, 5, 10], ignore_identical_ids=True)
    # logging.info(f"{ndcg=}, {recall=}, {precision=}")
    if cross_rerank:
        logging.info("***Results after Cross-Reranking the Margin Results. That is margin scores aggregated with cross scores: column name: `predicted_scores_reranked`***")
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results_reranked, k_values=[1, 3, 5, 10], ignore_identical_ids=True)

        col_agg1, col_agg2='predicted_scores_reranked', 'agg_predicted_scores'
        if not _is_column_all_dicts(df_test, col_agg1): df_test[col_agg1] = df_test[col_agg1].apply(ast.literal_eval)
        if not _is_column_all_dicts(df_test, col_agg2): df_test[col_agg2] = df_test[col_agg2].apply(ast.literal_eval)
        df_test['topic_margin_cross_agg_predicted_scores'] = df_test.apply(aggregate_two_scores, axis=1, col_agg1=col_agg1, col_agg2=col_agg2)
        # sorting
        df_test['topic_margin_cross_agg_predicted_scores'] = df_test['topic_margin_cross_agg_predicted_scores'].apply(
            lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
        )
        df_test['topic_margin_cross_agg_predicted_labels'] = df_test['topic_margin_cross_agg_predicted_scores'].apply(
            lambda x: {key: label_uid_to_title.get(key, 'Unknown') for key in x.keys()}
        )
        # sorting
        qrels, agg_results = get_qrels_results_agg(df_test, colname='topic_margin_cross_agg_predicted_scores')
        logging.info("***Results after Cross-Reranking. Aggregation of margin scores + cross-scores (i.e. previous, column_name: `predicted_scores_reranked`) and aggregated margin + TASS (col_name: `agg_predicted_scores`)***")
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, agg_results, k_values=[1, 3, 5, 10, 20, 30, 40], ignore_identical_ids=True)
        df_test.to_csv(predicted_file, index=False)
        logging.info(f"The predicted test data is saved once again to the file {predicted_file}. to include the column `topic_margin_cross_agg_predicted_scores/labels`")

# def crossrerank_hits(docs, doc_idxs, hits, predicted_labels, cross_encoder):
def crossrerank_hits(docs, hits, predicted_labels, cross_encoder):
    cross_inp=[]
    for doc, hit_lbs in zip(docs, predicted_labels):
        for lbl in hit_lbs.values():
            # cross_inp.append([docs[doc_idx], lbl])
            cross_inp.append([doc, lbl])
    cross_scores = cross_encoder.predict(cross_inp, batch_size=128, show_progress_bar=False)
    cross_scores = cross_scores.reshape(len(docs), -1)

    for i,hit in enumerate(hits):
        # Sort results by the cross-encoder scores
        for idx  in range(cross_scores.shape[1]):
            hit[idx]['cross-score'] = cross_scores[i][idx]
            # hit[idx]['margin-score'] = score

def min_max_normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [ (score - min_score) / (max_score - min_score + 1e-10) for score in scores ]

# def aggregate_ranked_lists(hits, w1=0.5, w2=0.5):
#     for hit in hits:
#         score_list = min_max_normalize([record['score'] for record in hit])
#         cross_list = min_max_normalize([record['cross-score'] for record in hit])
#         agg_list = [w1*score + w2*cross_score for score, cross_score in zip(score_list, cross_list)]
#         for i, agg_score in enumerate(agg_list): hit[i]['agg-score'] = agg_score

def aggregate_ranked_lists(hits, w1=0.5, w2=0.5, col1='score', col2='cross-score', agg_col='agg-score'):
    for hit in hits:
        # Determine values for col1: if col1 is None, use zeros for the entire column
        if col1 is None:
            values1 = [0] * len(hit)
        else:
            values1 = [0 if record[col1] is None else record[col1] for record in hit]

        # Determine values for col2: if col2 is None, use zeros for the entire column
        if col2 is None:
            values2 = [0] * len(hit)
        else:
            values2 = [0 if record[col2] is None else record[col2] for record in hit]

        # Normalize the values using min_max_normalize (assumed to be defined elsewhere)
        norm_values1 = min_max_normalize(values1)
        norm_values2 = min_max_normalize(values2)

        # Compute the weighted aggregated score
        agg_list = [w1 * v1 + w2 * v2 for v1, v2 in zip(norm_values1, norm_values2)]
        
        # Assign the aggregated score to each record under the key 'agg-score'
        for i, agg_score in enumerate(agg_list):
            hit[i][agg_col] = agg_score

def process_hits(hits, label_idx_to_uid, relevant_label_idxs, label_uid_to_title, cross_rerank=False):
    """
    Process each hit to extract and map label UIDs, titles, and scores.

    Args:
        hits (list of list of dicts): Each outer list item represents a hit, which is a list of dictionaries.
        label_idx_to_uid (dict): Mapping from index to label UIDs.
        relevant_label_idxs (dict): Mapping from corpus ID to relevant label index.
        label_uid_to_title (dict): Mapping from label UIDs to their titles.

    Returns:
        tuple: Two lists containing the dictionaries of labels and scores for each hit.
    
    Example usage:
    Assuming hits, label_idx_to_uid, relevant_label_idxs, label_uid_to_title are defined:
    predicted_labels, predicted_scores = process_hits(hits, label_idx_to_uid, relevant_label_idxs, label_uid_to_title)
    """
    predicted_labels = []
    predicted_scores = []

    for hit in hits:
        if cross_rerank: 
            hit = sorted(hit, key=lambda x: x['agg-score'], reverse=True) # Sort results by the cross-encoder scores

        hit_labels = {}  # Dictionary to store label UIDs and their titles for this hit
        hit_scores = {}  # Dictionary to store label UIDs and their scores for this hit
        for record in hit:
            try:
                label_uid = label_idx_to_uid[relevant_label_idxs[record['corpus_id']]]
            except KeyError as e:
                logging.error(f"KeyError accessing label index: {e}")
                # continue  # Skip this record or handle as needed
            except IndexError as e:
                # logging.error(f"IndexError accessing label UID: {e}")
                label_uid = label_idx_to_uid[record['corpus_id']]  
                # continue  # Skip this record or handle as needed

            label_title = label_uid_to_title.get(label_uid, "Unknown Label Title")
            hit_labels[label_uid] = label_title  # Map label UID to its title
            hit_scores[label_uid] = record['agg-score'] if cross_rerank else record['score']  # Map label UID to its score

        predicted_labels.append(hit_labels)  # Add the dictionary of labels for this hit to the list
        predicted_scores.append(hit_scores)  # Add the dictionary of scores for this hit to the list

    return predicted_labels, predicted_scores

def process_crosshits(hits, col='margin-score'):
    predicted_labels = []
    predicted_scores = []
    for hit in hits:
        hit = sorted(hit, key=lambda x: x[col], reverse=True) # Sort results by the `col` name passed
        hit_labels = {}  # Dictionary to store label UIDs and their titles for this hit
        hit_scores = {}  # Dictionary to store label UIDs and their scores for this hit
        for record in hit:
            hit_labels[record['label_id']] = record['label_title']
            hit_scores[record['label_id']] = record['agg-score']
        predicted_labels.append(hit_labels)
        predicted_scores.append(hit_scores)
    return predicted_labels, predicted_scores

def get_openai_model(model="gpt-4o-mini"):
    if not openai_API_KEY:
        logging.info("Openai API key not found. Cannot use prompt-based reranking.")
        return None
    client = openai.OpenAI(api_key=openai_API_KEY)
    openai_model = CustomOpenAI(client, model=model, chat=True, delay_in_seconds=10, exponential_backoff=False)
    return openai_model

def save_embeddings(embeddings, lbs, base_dir):
    """
    Saves the document and corpus embeddings into separate files.

    Parameters:
    -----------
    embeddings : torch.Tensor
        A PyTorch tensor containing both document and corpus embeddings.
    lbs : list or torch.Tensor
        A list or tensor containing labels used to split the embeddings.
    base_dir : str
        The directory where the embedding files will be saved.

    Returns:
    --------
    None
    """
    # Calculate embeddings splits
    doc_embeddings = embeddings[:-len(lbs)]
    label_embeddings = embeddings[-len(lbs):]

    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Define the paths for saving embeddings
    doc_embeddings_path = os.path.join(base_dir, 'doc_embeddings.pt')
    label_embeddings_path = os.path.join(base_dir, 'label_embeddings.pt')

    # Save the embeddings to files
    try:
        torch.save(doc_embeddings, doc_embeddings_path)
        torch.save(label_embeddings, label_embeddings_path)
        logging.info(f"Document embeddings saved to {doc_embeddings_path} for future use.")
        logging.info(f"Corpus embeddings saved to {label_embeddings_path} for future use.")
    except Exception as e:
        print(f"Failed to save embeddings: {e}")

def save_test_embeddings(embeddings, base_dir):
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Define the paths for saving embeddings
    test_doc_embeddings_path = os.path.join(base_dir, 'test_doc_embeddings.pt')

    # Save the embeddings to files
    try:
        torch.save(embeddings, test_doc_embeddings_path)
        logging.info(f"Test Document embeddings saved to {test_doc_embeddings_path} for future use.")
        return test_doc_embeddings_path
    except Exception as e:
        print(f"Failed to save embeddings: {e}")

def load_test_embeddings(base_dir):
    # Define the path for loading embeddings
    test_doc_embeddings_path = os.path.join(base_dir, 'test_doc_embeddings.pt')
    # Check if the files exist
    if not os.path.exists(test_doc_embeddings_path):
        raise FileNotFoundError(f"Test Document embeddings file not found at {test_doc_embeddings_path}")
    # Load the embeddings from the file
    try:
        embeddings = torch.load(test_doc_embeddings_path)
        logging.info(f"Test Document embeddings loaded from {test_doc_embeddings_path}.")
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to load embeddings: {e}")

def load_embeddings(base_dir):
    """
    Loads document and corpus embeddings from the specified base directory.

    Parameters:
    -----------
    base_dir : str
        The base directory where the embeddings files ('doc_embeddings.pt' and 'corpus_embeddings.pt') are stored.

    Returns:
    --------
    tuple of torch.Tensor
        A tuple containing two tensors: the document embeddings and the corpus embeddings.

    Raises:
    -------
    FileNotFoundError
        If either of the embeddings files does not exist in the base directory.
    Exception
        If there are issues loading the data from the files.
    """
    # Define the full paths to the embeddings files
    doc_embeddings_path = os.path.join(base_dir, 'doc_embeddings.pt')
    label_embeddings_path = os.path.join(base_dir, 'label_embeddings.pt')

    # Check if the files exist
    if not os.path.exists(doc_embeddings_path):
        raise FileNotFoundError(f"Document embeddings file not found at {doc_embeddings_path}")
    if not os.path.exists(label_embeddings_path):
        raise FileNotFoundError(f"Corpus embeddings file not found at {label_embeddings_path}")

    # Load the embeddings from files
    try:
        doc_embeddings = torch.load(doc_embeddings_path)
        label_embeddings = torch.load(label_embeddings_path)
        logging.info(f"Document embeddings loaded from {doc_embeddings_path}")
        logging.info(f"Corpus embeddings loaded from {label_embeddings_path}")
        return doc_embeddings, label_embeddings
    except Exception as e:
        raise Exception(f"Failed to load embeddings: {e}")

def print_args(args):
    # Prepare to print and log the formatted arguments
    args_dict = vars(args)  # Convert Namespace to dictionary for easy manipulation
    formatted_args = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    
    # Log and print the formatted arguments
    logging.info("Command-line Arguments:")
    logging.info("\n" + formatted_args)
    
    # Also print to console if necessary
    # print("Command-line Arguments:")
    # print(formatted_args)

def setup_logging(args, base_dir):
    # Configure the root logger based on the quiet flag
    if args.quiet:
        logging.basicConfig(level=logging.CRITICAL)  # Suppresses all logging below CRITICAL
    else:
        # Configure logging to include function name and line number
        logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]')

    # Set up file logging globally
    log_base_dir = Path(base_dir)/'logs'
    log_base_dir.mkdir(exist_ok=True, parents=True)
    # Format the current date and time to use in the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f'{args.logfile}_{current_time}.log' if args.logfile else f'default_logfile_{current_time}.log'
    log_filename = log_base_dir/log_filename
    # Create handlers: one for the file and one for the console
    file_handler = logging.FileHandler(log_filename)
    # Set the logging level based on the quiet flag
    file_handler.setLevel(logging.INFO)
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d:%(funcName)s]')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    # If you want to specifically control third-party loggers:
    third_party_logger = logging.getLogger('bertopic')
    third_party_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages from this library
    third_party_logger.addHandler(file_handler)
    # third_party_logger.propagate = False  # Stop propagation if you do not want these logs in the root logger

def get_top_labels_and_scores_old(uid, results, label_uid_to_title, top_k=200):
    # Function to get top 200 labels and scores as a dictionary

    # Get the dictionary of label_id -> score for this uid
    result_dict = results.get(uid, {})

    if result_dict:
        # Sort the labels by score in descending order
        sorted_labels = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top-k labels and their corresponding scores (or fewer if not enough)
        top_labels = sorted_labels[:top_k]
        top_labels = [label_uid for label_uid, _ in top_labels]


        # Extract scores for normalization
        scores = [result_dict[label_uid] for label_uid in top_labels]
        
        # Apply Min-Max normalization to the scores
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = [(score - min_score) / (max_score - min_score) if max_score != min_score else 0 for score in scores]
        
        # Prepare dictionaries for labels and scores
        label_uid_to_title_dict = {label_uid: label_uid_to_title.get(label_uid, "") for label_uid in top_labels}
        label_uid_to_score_dict = {label_uid: normalized_score for label_uid, normalized_score in zip(top_labels, normalized_scores)}
        # label_uid_to_score_dict = {label_uid: score for label_uid, score in top_labels}

        
        return label_uid_to_title_dict, label_uid_to_score_dict
    
    return None, None

def get_top_labels_and_scores(uid, results, label_uid_to_title, top_k=200):
    # Function to get top 200 labels and scores as a dictionary
    # Get the dictionary of label_id -> score for this uid
    result_dict = results.get(uid, {})
    
    if result_dict:
        # Sort the labels by score in descending order
        sorted_labels = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top-k labels and their corresponding scores (or fewer if not enough)
        top_labels = sorted_labels[:top_k]
        
        # Prepare dictionaries for labels and scores
        label_uid_to_title_dict = {label_uid: label_uid_to_title.get(label_uid, "") for label_uid, _ in top_labels}
        label_uid_to_score_dict = {label_uid: score for label_uid, score in top_labels}
        
        return label_uid_to_title_dict, label_uid_to_score_dict
    return None, None

def min_max_normalize_dict(score_dict):
    # Function to apply Min-Max normalization to a dictionary of scores
    # Extract the scores from the dictionary
    scores = list(score_dict.values())
    
    # If all scores are the same, avoid division by zero
    if len(scores) > 1:
        min_score = min(scores)
        max_score = max(scores)
        # Apply min-max normalization
        normalized_scores = {
            key: (score - min_score) / (max_score - min_score) if max_score != min_score else 0
            for key, score in score_dict.items()
        }
    else:
        # If there's only one score, set it to 0 (no variation)
        normalized_scores = {key: 0 for key in score_dict}
    
    return normalized_scores

def aggregate_predicted_scores(row):
    # Function to aggregate the scores after min-max normalization
    predicted_scores = ast.literal_eval(row['predicted_scores'])
    
    # Check if 'margin_mseloss_predicted_scores' is NaN before performing aggregation
    if pd.isna(row['margin_mseloss_predicted_scores']) or row['margin_mseloss_predicted_scores'] is None:
        return min_max_normalize_dict(predicted_scores)  # Return an empty dictionary if 'margin_mseloss_predicted_scores' is NaN
    
    try:
        margin_scores = row['margin_mseloss_predicted_scores']
    except Exception as e:
        print('jj')
        import pdb; pdb.set_trace()
    
    # Normalize both dictionaries
    normalized_predicted_scores = min_max_normalize_dict(predicted_scores)
    normalized_margin_scores = min_max_normalize_dict(margin_scores)
    
    # Union of keys from both dictionaries
    all_keys = set(normalized_predicted_scores.keys()).union(set(normalized_margin_scores.keys()))
    # import pdb; pdb.set_trace()
    
    # Sum the corresponding normalized values
    agg_scores = {}
    for key in all_keys:
        predicted_score = normalized_predicted_scores.get(key, 0)  # Default to 0 if key is not in predicted_scores
        margin_score = normalized_margin_scores.get(key, 0)  # Default to 0 if key is not in margin_scores
        agg_scores[key] = predicted_score + margin_score
    
    return agg_scores

def aggregate_two_scores(row, col_agg1='predicted_scores_reranked', col_agg2='agg_predicted_scores'):
    # Function to aggregate the scores from two columns after min-max normalization
    
    scores1 = row[col_agg1]
    
    if (col_agg2 not in row.index) or (pd.isna(row[col_agg2])) or (row[col_agg2] is None) :
        return min_max_normalize_dict(scores1)
    
    # scores2 = ast.literal_eval(row[col_agg2])
    scores2 = row[col_agg2]

    normalized_scores1 = min_max_normalize_dict(scores1)
    normalized_scores2 = min_max_normalize_dict(scores2)
    all_keys = set(normalized_scores1.keys()).union(set(normalized_scores2.keys()))

    agg_scores = {}
    for key in all_keys:
        score1 = normalized_scores1.get(key, 0)  # Default to 0 if key is not in predicted_scores
        score2 = normalized_scores2.get(key, 0)  # Default to 0 if key is not in margin_scores
        agg_scores[key] = score1 + score2

    return agg_scores

def load_or_train_topic_model(base_dir, docs, lbs, openai_API_KEY, args, seed_topic_list=None):
    if args.train_fresh_topicmodel:
        # Train the topic model from scratch
        logging.info("Training a fresh topic model...")
        
        # Prepare the topic model
        embedding_model, topic_model = get_topic_model(
            embedding_model=args.embedding_model_name,
            hdbscan_min_cluster_size=args.min_cluster_size,
            min_topic_size=args.min_topic_size,
            openai_API_KEY=openai_API_KEY,
            language='multilingual',
            calculate_probabilities=False,
            low_memory=False,
            seed_topic_list=seed_topic_list if args.guided_topic_modelling else None
        )

        # Pre-calculate/Load embeddings
        try:
            doc_embeddings, label_embeddings = load_embeddings(base_dir)
            # import pdb; pdb.set_trace()
            embeddings = np.concatenate((doc_embeddings, label_embeddings), axis=0)
            logging.info(f"Loaded Embeddings found at {base_dir}.")
        except FileNotFoundError:
            logging.info(f"Embeddings not found at {base_dir}. Creating fresh embeddings using {args.embedding_model_name}.")
            embeddings = embedding_model.encode(docs, batch_size=args.embed_bs, show_progress_bar=True)
            save_embeddings(embeddings, lbs, base_dir)

        topics, probs = topic_model.fit_transform(docs, embeddings)
        # topic_model.get_document_info(docs).to_csv('rep_docs.csv', index=False)

        # Custom labels
        if openai_API_KEY:
            # ChatGPT's labels
            chatgpt_topic_labels = {
                topic: " | ".join(list(zip(*values))[0])
                for topic, values in topic_model.topic_aspects_["OpenAI"].items()
            }
            topic_model.set_topic_labels(chatgpt_topic_labels)
        else:
            # KeyBERT-inspired labels
            keybert_topic_labels = {
                topic: " | ".join(list(zip(*values))[0][:3])
                for topic, values in topic_model.topic_aspects_["KeyBERT"].items()
            }
            topic_model.set_topic_labels(keybert_topic_labels)
        
        # Save the topic model
        logging.info(f"The parameters of the topic model are {topic_model.get_params()}")
        logging.info(f"Saving the topic model at {base_dir}...")
        save_topic_model(topic_model, base_dir, args.embedding_model_name)
    else:
        # Load an existing topic model
        logging.info(f"Attempting to load an existing topic model from {base_dir}...")
        try:
            doc_embeddings, label_embeddings = load_embeddings(base_dir)
            topic_model = BERTopic.load(base_dir, embedding_model=args.embedding_model_name)
            logging.info(f"Successfully loaded the topic model from {base_dir}.")
            # topic_model.push_to_hf_hub(
            #     repo_id="topicmodel",
            #     save_ctfidf=True,
            #     save_embedding_model=args.embedding_model_name
            # )
            # logging.info(f"Successfully pushed the topic model to huggingface.")

        except Exception as e:
            logging.error(f"Failed to load the topic model from {base_dir}. Error: {e}")
            raise  # Re-raise the exception after logging
    
    return topic_model, doc_embeddings, label_embeddings

def is_valid_str(value):
    """
    Return True if value is a non-empty string (not NaN, not empty or whitespace-only).
    """
    # Check for NaN
    if pd.isna(value):
        return False
    # Check for string type and non-empty content
    return isinstance(value, str) and value.strip() != ""

def validate_columns(df):
    # Ensure the required columns exist
    required_columns = ["_id", "text", "metadata"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the dataframe.")
    
    # Identify rows with invalid '_id' or 'text'
    invalid_ids = df[~df["_id"].apply(is_valid_str)]
    invalid_texts = df[~df["text"].apply(is_valid_str)]
    
    if invalid_ids.empty and invalid_texts.empty:
        logging.info("All values in '_id' and 'text' columns are valid non-empty strings (and not NaN).")
        return True
    else:
        if not invalid_ids.empty:
            logging.info("Rows with invalid '_id':")
            logging.info(invalid_ids)
        if not invalid_texts.empty:
            logging.info("Rows with invalid 'text':")
            logging.info(invalid_texts)
        return False

def find_summarization_jsonl_files(base_dir):
    # Define separate patterns for train and test files
    train_pattern = os.path.join(base_dir, '**', 'train*summarization*results.jsonl')
    test_pattern = os.path.join(base_dir, '**', 'test*summarization*results.jsonl')
    
    # Find all matching files recursively for each pattern
    train_files = glob.glob(train_pattern, recursive=True)
    test_files = glob.glob(test_pattern, recursive=True)

    return train_files, test_files

def extract_query_summary_from_jsonl(file_path):
    data_list = []
    # Open the .jsonl file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_data = json.loads(line)

            # Extract the user message and ummary
            _id = json_data[2]['_id']
            user_message = json_data[0]['messages'][1]['content']
            summary = json_data[1]['choices'][0]['message']['content']
            # text = f"The description of a Wikipedia article titled ***{_id}*** is ***{summary}***. What could be a related title?"
            text = summary
            data_list.append({"_id": _id, "text": text, "metadata": {}})

    # Create a DataFrame from the list of data
    df = pd.DataFrame(data_list)

    return df

def chatgpt_query_summary(base_dir):
    # Retrieve train and test files separately
    train_files, test_files = find_summarization_jsonl_files(base_dir)
    
    logging.info(f"Found {len(train_files)} train files and {len(test_files)} test files for summarization results.")
    logging.info(f"Train files: {train_files}")
    logging.info(f"Test files: {test_files}")

    def process_files(files):
        data_frames = []
        for file in files:
            df_extracted = extract_query_summary_from_jsonl(file)
            is_valid = validate_columns(df_extracted)
            if is_valid:
                data_frames.append(df_extracted)
            else:
                logging.info(f"The file {file} has invalid entries.")
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no valid data

    # Process train and test files separately
    train_df = process_files(train_files)
    test_df = process_files(test_files)

    # Save train results if any train files exist
    if train_files:
        train_out_path = Path(train_files[0]).parent / 'train_chatgpt_queries.jsonl'
        train_df.to_json(train_out_path, orient='records', lines=True)
        logging.info(f"Saved train queries to {train_out_path}")
    else:
        logging.info("No train files found.")

    # Save test results if any test files exist
    if test_files:
        test_out_path = Path(test_files[0]).parent / 'test_chatgpt_queries.jsonl'
        test_df.to_json(test_out_path, orient='records', lines=True)
        logging.info(f"Saved test queries to {test_out_path}")
    else:
        logging.info("No test files found.")

def main():
    """
    Main function to control the program execution. Parses arguments and calls the appropriate process functions.
    """
    # Setup argparse
    parser = argparse.ArgumentParser(description='Topic Modelling.')
    parser.add_argument("--source_url", type=str, help="URL of the dataset")
    parser.add_argument('--dataset', type=str, help='Name of the dataset')
    parser.add_argument("--base_dir", type=str, help="Path to the base directory that will contain intermediate results and models.")
    parser.add_argument("--labels_file_name", type=str, help="Path to the file containing label descriptions")
    parser.add_argument("--train_file_path", type=str, help="Path to the json file containing training examples")
    parser.add_argument("--test_file_path", type=str, help="Path to the json file containing testing examples")
    parser.add_argument("--train_test_file_path", type=str, help="Path to the csv file containing training and testing examples")
    parser.add_argument("--topicfied_file_name", default="trn_clustered.json",type=str, help="Path to the file containing training examples")
    parser.add_argument("--embedding_model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the embedding model to download from Sentence Transformer")
    parser.add_argument('--guided_topic_modelling', action='store_true', help='Enable guided topic modelling if this flag is set.')
    parser.add_argument('--quiet', action='store_true', help='Suppress logging output')
    parser.add_argument('--pseudolabel', action='store_true', help='Label the clustered data')
    parser.add_argument('--pseudolabel_topk', type=int, default=50, help='Set the pseudo-labelling topk label size. Default is 50.')
    parser.add_argument('--embed_bs', type=int, default=32, help='Set the embedding batch size. Default is 32.')
    parser.add_argument('--split_sz', type=int, default=None, help='Set the embedding batch size. Default is 32.')
    parser.add_argument('--predict', action='store_true', help='Label the clustered data')
    parser.add_argument('--test_set_to_query', action='store_true', help='Convert the test data into queries. This will be used during cross-reranking.')
    parser.add_argument('--predict_topk', type=int, default=10, help='Set the k for prediction topk label size. Default is 10.')
    parser.add_argument('--min_cluster_size', type=int, default=15, help='It controls the minimum size of a cluster and thereby the number of clusters that will be generated. It is set to 10 as a default. Increasing this value results in fewer clusters but of larger size whereas decreasing this value results in more micro clusters being generated. Typically, I would advise increasing this value rather than decreasing it.')
    parser.add_argument('--min_topic_size', type=int, default=10, help='It is used to specify what the minimum size of a topic can be. The lower this value the more topics are created. If you set this value too high, then it is possible that simply no topics will be created! Set this value too low and you will get many microclusters.')
    
    parser.add_argument('--prompt_rerank', action='store_true', help='Prompt based reranking using LLM')
    parser.add_argument('--prompt_rerank_topk_lbs_cutoff', type=int, default=100, help='The topk value used in the LLM Prompt Reranking Algorithm. Default is 100.')
    parser.add_argument(
            '--prompt_rerank_predicted_scores_col',
            default='predicted_scores',
            type=str,
            help="Column name in 'tst_predicted.csv' for predicted scores."
        )
    parser.add_argument(
        '--prompt_rerank_predicted_labels_col',
        default='predicted_labels',
        type=str,
        help="Column name in 'tst_predicted.csv' for predicted labels."
    )

    parser.add_argument('--cross_rerank', action='store_true', help='Cross-Encoder based Reranking.')
    parser.add_argument('--logfile', type=str, help='Filename to write logs to')
    parser.add_argument('--prep_margin_mseloss_data', action='store_true', help='Generate corpus, queries and qrels in beir format')
    parser.add_argument('--chatgpt_query_summary', action='store_true', help='Collates ChatGPT Summarization Results')
    parser.add_argument('--train_margin_mseloss', action='store_true', help='Train retrieval model with mse loss')
    parser.add_argument(
        "--retrievers",
        # nargs="+",
        default=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
        help='Indicate retriever names for mining negatives. They could be one or many BM25 ("bm25") or dense retrievers (in SBERT format).',
    )
    parser.add_argument(
        "--retriever_score_functions",
        # nargs="+",
        default=["cos_sim", "cos_sim"],
        # choices=["dot", "cos_sim", "none"],
        help='Score functions of the corresponding retrievers for negative mining. Please set it to "none" for BM25.',
    )
    parser.add_argument(
        "--negatives_per_query",
        type=int,
        default=50,
        help="Mine how many negatives per query per retriever",
    )
    parser.add_argument('--num_pos', type=int, default=10, help='Number of Positive documents for each query. Default is 10.')
    parser.add_argument(
        "--gpl_steps", type=int, default=140000, help="Training steps for GPL."
    )
    parser.add_argument("--batch_size_gpl", type=int, default=32)
    parser.add_argument(
        "--cross_encoder", default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    parser.add_argument("--max_seq_length", type=int, default=350)
    parser.add_argument(
        "--base_ckpt",
        default="distilbert-base-uncased",
        help="Initialization checkpoint in HF or SBERT format. Meaning-pooling will be used.",
    )
    parser.add_argument('--stop_after_gpl_train_data_creation', action='store_true', help='Stop after creating gpl-training-data.tsv. That is skip training and evaluation.')
    parser.add_argument(
        "--gpl_score_function", choices=["dot", "cos_sim"], default="dot"
    )
    parser.add_argument("--checkpoint_save_steps", type=int, default=10000)
    parser.add_argument(
        "--output_dir", required=True, help="Output path for the Margin-MSELoss model."
    )
    parser.add_argument(
        "--evaluation_data",
        default="./",
        type=str,
        help="Path to the BeIR-format dataset for evaluation.",
    )
    parser.add_argument(
        "--evaluation_output", default="output", help="Path for the evaluation output."
    )
    parser.add_argument(
        "--do_evaluation",
        action="store_true",
        default=False,
        help="Wether to do the evaluation (after training Margin-MSE loss)",
    )
    parser.add_argument("--label_frac_sup", type=int, default=0, help="The percentage of supervision.")
    parser.add_argument(
        '--no_fresh_topicmodel',
        action='store_false',
        dest='train_fresh_topicmodel',
        help='If specified, does not train a fresh topic model. Defaults to training a fresh model (True).'
    )
    parser.add_argument(
        '--no_topic_labels',
        action='store_true',
        help='If specified, disables topic labels (default: False).'
    )
    parser.add_argument(
        '--do_summarize',
        action='store_true',
        help='If specified, enables summarization (default: False).'
    )
    parser.add_argument(
        '--max_content_length',
        type=int,
        default=500,
        help='Maximum allowed content length in queries for beir format. Default is 500.'
    )
    parser.add_argument("--max_records_per_file", type=int, default=10000, help='Maximum number of records for CHATGPT Re-Ranking API requests. Default is 10000.')
    parser.add_argument(
        '--max_doc_length_chatgpt_ranking',
        type=int,
        default=None,
        help='Maximum document length for ChatGPT ranking. (default: None)'
    )

    args = parser.parse_args()

    source = untar_xxx(eval(args.source_url))

    # Setting the base directory
    global base_dir
    base_dir = args.base_dir + '/' + args.source_url.split('.')[1]

    # This is where the model is saved
    output_dir = str(Path(base_dir)/args.output_dir)

    # Setup logger and print arguments
    setup_logging(args, base_dir)
    print_args(args)

    logging.info("****Program started****")

    if args.test_set_to_query:
        train_test_file = source/args.train_test_file_path
        try:
            logging.info(f"Opening the file {train_test_file} to convert the test examples into queries.")
            df_test = pd.read_csv(train_test_file)
            df_test = df_test[df_test['split'] == 'test'].reset_index(drop=True)
            convert_test_data_to_queries(df_test)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:  
            logging.error(f"An unexpected error occurred while loading topic model: {e}")
            logging.error("Make sure the topic model was trained and saved in the appropriate directory")
        finally:
            sys.exit(1)

    if args.predict:
            try:
                topic_model = BERTopic.load(base_dir, embedding_model=args.embedding_model_name)
                logging.info("Topic Model loaded successfully for making predictions!")
                ## ******
                topic_model.push_to_hf_hub(
                repo_id="topicmodel",
                save_ctfidf=True,
                save_embedding_model=args.embedding_model_name
                )
                logging.info(f"Successfully pushed the topic model to huggingface.")
                ## ******
                topics_file = Path(base_dir)/'topics.csv'
                train_test_file = source/args.train_test_file_path
                label_cluster_file = Path(base_dir)/'trn_labels_for_each_cluster.csv'
                label_file = Path(source/args.labels_file_name)
                doc_embeddings, label_embeddings = load_embeddings(base_dir)
                make_predcitions(topic_model, train_test_file, label_cluster_file, label_file, topics_file, label_embeddings, top_k=args.predict_topk, prompt_rerank=args.prompt_rerank, 
                                prompt_rerank_topk_lbs_cutoff=args.prompt_rerank_topk_lbs_cutoff,
                                prompt_rerank_predicted_labels_col=args.prompt_rerank_predicted_labels_col,
                                prompt_rerank_predicted_scores_col=args.prompt_rerank_predicted_scores_col,
                                cross_rerank=args.cross_rerank,
                                max_records_per_file=args.max_records_per_file,
                                topic_labels = not args.no_topic_labels,
                                max_doc_length_chatgpt_ranking=args.max_doc_length_chatgpt_ranking)
            except FileNotFoundError as e:
                print(e)
            except Exception as e:  
                logging.error(f"An unexpected error occurred while loading topic model: {e}")
                logging.error("Make sure the topic model was trained and saved in the appropriate directory")
            finally:
                sys.exit(1)

    if args.pseudolabel:
        try:
            topic_model = BERTopic.load(base_dir, embedding_model=args.embedding_model_name)
            logging.info("Topic Model loaded successfully to pseudo label training data!")
            clustered_file = Path(base_dir)/'trn_clustered.csv'
            label_file = Path(source/args.labels_file_name)
            doc_embeddings, label_embeddings = load_embeddings(base_dir)
            print("Embeddings successfully loaded.")
            label_data(topic_model, clustered_file, label_file, doc_embeddings, label_embeddings, top_k=args.pseudolabel_topk)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:  
            logging.error(f"An unexpected error occurred while loading topic model: {e}")
            logging.error("Make sure the topic model was trained and saved in the appropriate directory")
        finally:
            sys.exit(1)

    if args.prep_margin_mseloss_data:
        try:
            pseudo_labelled_trnfile = Path(base_dir)/'trn_clustered_pseudo_labelled.csv'
            train_test_file = source/args.train_test_file_path

            label_file = Path(source/args.labels_file_name)
            prepare_dataset_for_margin_mse_loss(pseudo_labelled_trnfile, train_test_file, label_file, num_pos=args.num_pos, 
                                                label_frac_sup=args.label_frac_sup, summarize=args.do_summarize, max_content_length=args.max_content_length)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:  
            logging.error(f"An unexpected error occurred while preparing the beir format dataset: {e}")
        finally:
            sys.exit(1)
    
    if args.chatgpt_query_summary:
        try:
            chatgpt_query_summary(base_dir)
        except Exception as e:
            logging.error(f"An unexpected error occurred while collating the ChatGPT summarization: {e}")
        finally:
            sys.exit(1)

    if args.train_margin_mseloss:
        try:
            if "qgen-qrels" in os.listdir(base_dir) and "qgen-queries.jsonl" in os.listdir(base_dir):
                logging.info("Loading from existing generated data")
                corpus, gen_queries, gen_qrels = GenericDataLoader(base_dir, prefix='qgen').load(split="train")
            else:
                raise FileNotFoundError("Please use the --prep_margin_mseloss_data first to prepare the data")
            
            # Hard Negative Mining
            if "hard-negatives.jsonl" in os.listdir(base_dir):
                logging.info("Using exisiting hard-negative data")
            else:
                logging.info("No hard-negative data found. Now mining it")
                args.retrievers =  ast.literal_eval(args.retrievers)
                args.retriever_score_functions = ast.literal_eval(args.retriever_score_functions)
                miner = NegativeMiner(
                    base_dir,
                    'qgen',
                    retrievers=args.retrievers,
                    retriever_score_functions=args.retriever_score_functions,
                    nneg=args.negatives_per_query,
                    use_train_qrels=False,
                )
                miner.run()
            
            # Pseudo Labelling
            gpl_training_data_fname = "gpl-training-data.tsv"
            if gpl_training_data_fname in os.listdir(base_dir):
                logging.info("Using existing GPL-training data")
            else:
                logging.info(f"No GPL-training data found. Now generating it via pseudo labeling with {args.cross_encoder}")
                pseudo_labeler = PseudoLabeler( 
                    base_dir, 
                    gen_queries, 
                    corpus, 
                    args.gpl_steps, 
                    args.batch_size_gpl, 
                    args.cross_encoder, 
                    args.max_seq_length
                )
                pseudo_labeler.run()

            if args.stop_after_gpl_train_data_creation: 
                sys.exit(1)

            # Train with MarginMSE loss
            #### This will be skipped if the checkpoint at the indicated training steps can be found ####
            ckpt_dir = os.path.join(output_dir, str(args.gpl_steps))
            if not os.path.exists(ckpt_dir) or (
                os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)
            ):
                logging.info(f"Now doing training on the generated data with the MarginMSE loss with the model {args.base_ckpt}...")
                model: SentenceTransformer = load_sbert(args.base_ckpt, pooling=None, max_seq_length=args.max_seq_length)
                fpath_gpl_data = os.path.join(base_dir, gpl_training_data_fname)
                logging.info(f"Load GPL training data from {fpath_gpl_data}")
                train_dataset = GenerativePseudoLabelingDataset(
                    fpath_gpl_data, gen_queries, corpus
                )
                train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size_gpl, drop_last=True)  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
                train_loss = MarginDistillationLoss(model=model, similarity_fct=args.gpl_score_function)

                # assert gpl_steps > 1000
                model.fit(
                    [
                        (train_dataloader, train_loss),
                    ],
                    epochs=1,
                    steps_per_epoch=args.gpl_steps,
                    warmup_steps=1000,
                    checkpoint_save_steps=args.checkpoint_save_steps,
                    checkpoint_save_total_limit=10000,
                    output_path=output_dir,
                    checkpoint_path=output_dir,
                    use_amp=False,
                )
            else:
                logging.info("Trained GPL model found. Now skip training")
            
            ### Evaluation the Margin-MSE loss model
            if args.do_evaluation:
                logging.info("Doing evaluation for GPL")
                queries, results = evaluate(
                    base_dir,
                    base_dir+'/'+args.evaluation_output,
                    ckpt_dir,
                    args.max_seq_length,
                    score_function=args.gpl_score_function,
                    pooling=None,
                    split='test',
                )
                tst_predicted_path = os.path.join(base_dir, 'tst_predicted.csv')
                if os.path.exists(tst_predicted_path):
                    logging.info(f"Loading the file {tst_predicted_path} to write the margin-mseloss results")
                    df_predicted = pd.read_csv(tst_predicted_path)
                    label_file = Path(source/args.labels_file_name)
                    label_uid_to_title = get_label_uid_to_title_dict(label_file)
                    # Apply the function to each row in df_predicted and create the new columns
                    df_predicted[['margin_mseloss_predicted_labels', 'margin_mseloss_predicted_scores']] = df_predicted['uid'].apply(
                        lambda uid: pd.Series(get_top_labels_and_scores(uid, results, label_uid_to_title))
                    )
                    # Add scores to create the 'agg_predicted_scores' column
                    df_predicted['agg_predicted_scores'] = df_predicted.apply(aggregate_predicted_scores, axis=1)
                    df_predicted['agg_predicted_scores_sorted'] = df_predicted['agg_predicted_scores'].apply(
                        lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True))
                    )
                    label_file = Path(source/args.labels_file_name)
                    label_uid_to_title = get_label_uid_to_title_dict(label_file)
                    df_predicted['agg_predicted_labels_sorted'] = df_predicted['agg_predicted_scores_sorted'].apply(
                        lambda x: {key: label_uid_to_title.get(key, 'Unknown') for key in x.keys()}
                    )
                    df_predicted.to_csv(tst_predicted_path, index=False)
                    logging.info("The margin-mseloss results written to {tst_predicted_path}.")
                    logging.info("Computing metrics with topic predictions and margin-mseloss aggregated scores")
                    qrels, results = get_qrels_results_agg(df_predicted)
                    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
                    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values=[1, 3, 5, 10, 20, 30, 40], ignore_identical_ids=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred while training margin mse-loss retrieval: {e}")
        finally:
            sys.exit(1)
    
    try:
        # Process data based on arguments
        logging.info(f"Loading the docs from the train file with split size {slice(0, args.split_sz)}")
        docs = get_dataset(args.dataset, split='train', split_sz=args.split_sz)
        logging.info(f"Loaded {len(docs)} documents.")

        # Get the labels description
        lbs = get_labels(source/args.labels_file_name)
        # lbs = lbs[:10000] # remove later
        logging.info(f"Loaded {len(lbs)} labels.")
        seed_topic_list = [word_tokenize(label_sentence) for label_sentence in lbs]

        # make sure that the documents and the labels are embedded into the same space
        # lbs_splitted = [sent_tokenize(lbl) for lbl in lbs]
        # lbs_splitted = [sentence for lbl in lbs for sentence in lbl]
        docs.extend(lbs)

        topic_model, doc_embeddings, label_embeddings = load_or_train_topic_model(base_dir, docs, lbs, openai_API_KEY, args, seed_topic_list=seed_topic_list)

        # get the topic model
        # embedding_model, topic_model = get_topic_model(embedding_model=args.embedding_model_name, 
        #                                             hdbscan_min_cluster_size=args.min_cluster_size,
        #                                             min_topic_size=args.min_topic_size,
        #                                             openai_API_KEY=openai_API_KEY, 
        #                                             language='multilingual', 
        #                                             calculate_probabilities=False,
        #                                             low_memory=False,
        #                                             seed_topic_list=seed_topic_list if args.guided_topic_modelling else None)

        # # Pre-calculate/Load embeddings
        # try:
        #     doc_embeddings, label_embeddings = load_embeddings(base_dir)
        #     embeddings = np.concatenate((doc_embeddings, label_embeddings), axis=0)
        #     logging.info(f"Loaded Embeddings found at {base_dir}.")
        # except FileNotFoundError:
        #     logging.info(f"Embeddings not found at {base_dir}. Creating fresh embeddings using {args.embedding_model_name}.")
        #     embeddings = embedding_model.encode(docs, batch_size=args.embed_bs, show_progress_bar=True)
        #     save_embeddings(embeddings, lbs, base_dir)
        
        # # Train model
        # #
        # topics, probs = topic_model.fit_transform(docs, embeddings) # topics and the corresponding probabilities

        # # topic_model.get_document_info(docs).to_csv('rep_docs.csv', index=False)

        # # Custom labels
        # if openai_API_KEY:
        #     # ChatGPT's labels
        #     chatgpt_topic_labels = {topic: " | ".join(list(zip(*values))[0]) for topic, values in topic_model.topic_aspects_["OpenAI"].items()}
        #     # chatgpt_topic_labels[-1] = "Outlier Topic"
        #     topic_model.set_topic_labels(chatgpt_topic_labels)
        # else:
        #     # or use one of the other topic representations, like KeyBERTInspired
        #     keybert_topic_labels = {topic: " | ".join(list(zip(*values))[0][:3]) for topic, values in topic_model.topic_aspects_["KeyBERT"].items()}
        #     topic_model.set_topic_labels(keybert_topic_labels)

        # # Save the topic model and the embedding model
        # logging.info(f"The parameters of the topic model are {topic_model.get_params()}")
        # logging.info(f"Saving the topic model at {base_dir}...")
        # save_topic_model(topic_model, base_dir, args.embedding_model_name)

        # Topicfy data points based off of topics
        topicfy_data(source/args.train_file_path, topic_model, args.topicfied_file_name, doc_embeddings, base_dir=base_dir, split_sz=args.split_sz)

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        sys.exit(1)

    logging.info("Program ended successfully")

# Ensure the script can be used as a script as well as an importable module
if __name__ == "__main__":
    main()
