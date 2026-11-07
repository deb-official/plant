#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: my_script.py
Description: This script demonstrates how to use argparse to take inputs from the command line. It performs basic operations based on the provided arguments.
Author: Your Name
Created Date: YYYY-MM-DD
Last Modified Date: YYYY-MM-DD
Version: 1.0
Usage: python topics.py --sum 10 20
"""

# Import standard libraries
import os
import sys
import argparse
import logging
import json
import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import string
import copy
from pprint import pprint

# Import xcube specific libraries
from xcube.data.all import *
from xcube.imports import *

# Import from topics
from topics import get_label_uid_to_title_dict

# Import special libraries
import editdistance
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm

# Constants
CONSTANT_EXAMPLE = 'Hello World'

def find_jsonl_files(base_dir):
    # Construct the full path for the file pattern
    pattern = os.path.join(base_dir, 'api_requests*results.jsonl')
    
    # Use glob to find all files matching the pattern
    # The '**' allows for recursive search (all directories and subdirectories)
    files = glob.glob(pattern, recursive=True)
    
    return files

def find_closest_key(keys_predicted, target_key):
    """
    Finds the closest key to the target_key in the list of keys_predicted based on edit distance.

    Parameters:
    - keys_predicted (list of str): A list of key strings.
    - target_key (str): The target key string to compare against.

    Returns:
    - tuple: A tuple containing the closest key and the minimum edit distance.
    """
    # Initialize variables to track the minimum distance and the closest key
    min_distance = float('inf')
    closest_key = None

    # Calculate edit distance for each key in the list
    for key in keys_predicted:
        distance = editdistance.eval(key, target_key)
        if distance < min_distance:
            min_distance = distance
            closest_key = key

    return closest_key, min_distance

def get_key_value_old(dict_str, target_key):
    # Regex pattern to extract the value for the target key
    # Pattern explanation: Look for the target key, then match a colon, optional whitespace,
    # and capture what's within single or double quotes following it.
    pattern = r"'{}' *: *['\"]([^'\"]+)['\"]".format(target_key)

    # Using re.search to find the value associated with the target key
    match = re.search(pattern, dict_str)

    if match:
        value = match.group(1)
        return value
    else:
        return None

def escape_quotes(text):
    # Regex to find quotes not preceded by a space and not followed by a colon or comma
    # Uses a negative lookbehind to check that the previous character is not a space
    # Uses a negative lookahead to check that the following character is not a colon or comma
    pattern = r'(?<!\s)(["\'])(?![\s:,])'
    # Replace matched quotes with their escaped versions
    escaped_text = re.sub(pattern, r'\\\\\1', text).replace("\\\\'", "")
    return escaped_text

def extract_value_from_dict_str(dict_str, key):
    pattern = rf'"{key}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, dict_str)
    return match.group(1) if match else None

def get_key_value(dict_str, target_key):
    try:
        # Prepare the regex pattern to extract the value for the target key
        # This format safely escapes the target_key to avoid regex errors from special characters.
        # pattern = r"'{}' *: *['\"]([^'\"]+)['\"]".format(re.escape(target_key))
        pattern = r"'{}' *: *['\"]([^'\"]+)['\"](?=,|\n)".format(re.escape(target_key))


        # Using re.search to find the value associated with the target key
        match = re.search(pattern, dict_str)

        if match:
            value = match.group(1)
            return value
        else:
            return None
    except re.error as err:
        # Handle regex errors by returning None and optionally logging the error
        # logging.info(f"Regex error occurred: {err}")
        return None

# Tokenizer function that processes text
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def get_with_longest_prefix(dictionary, target_key, bm25, corpus, force_lex_search=False):
    """
    Retrieves the value for the target_key from the dictionary.
    If the key doesn't exist, it returns the value of the key with the longest matching prefix.

    Parameters:
    - dictionary (dict): The dictionary from which to retrieve the value.
    - target_key (str): The key to search for in the dictionary.

    Returns:
    - The value corresponding to the exact key or the longest prefix match.
    """
    # Try to get the exact match first
    if target_key in dictionary:
        return dictionary[target_key]

    # Initialize variables to find the longest prefix match
    longest_prefix = None
    max_length = 0

    # Iterate over all keys to find the longest prefix match
    for key in dictionary.keys():
        # Check if the current key is a prefix of the target key
        if target_key.startswith(key) and len(key) > max_length:
            longest_prefix = key
            max_length = len(key)

    # Return the value of the longest prefix if found
    if longest_prefix and not force_lex_search:
        return dictionary[longest_prefix]
    
    # BM25 search if no exact or prefix match is found
    # Prepare corpus
    # corpus = list(dictionary.keys())
    # tokenized_corpus = [bm25_tokenizer(doc) for doc in corpus]
    # bm25 = BM25Okapi(tokenized_corpus)

    # Get scores for the target key
    bm25_scores = bm25.get_scores(bm25_tokenizer(target_key))
    top_index = np.argmax(bm25_scores)  # Get the index of the highest score
    if bm25_scores[top_index] > 0:  # Check if there's a significant score
        return dictionary[corpus[top_index]]

    # Return None if no match is found
    return None

def extract_uids(dict_str):
    # Regex pattern to match quoted keys consisting of digits or uppercase alphabets
    pattern = r"['\"]([A-Z0-9]+)['\"]\s*:"
    # Find all keys that match the pattern
    uids = re.findall(pattern, dict_str)
    return uids

def extract_uids_wikidataset(dict_str):
    # Regex pattern to match quoted keys consisting of digits or uppercase alphabets
    # Find all keys that match the pattern
    # uids = re.findall(pattern, dict_str)
    # uids = [match[1] for match in re.findall(r"(['\"])([^'\"]+)\1\s*:", dict_str)]
    uids = [match[1] for match in re.findall(r"(['\"])((?:(?!\1\s*:).)+)\1\s*:", dict_str)]

    return uids

def extract_data_from_jsonl(file_path, label_file):
    # Initialize a list to store the data
    data_list = []

    label_uid_to_title = get_label_uid_to_title_dict(label_file)
    label_title_to_uid = {v: k for k, v in label_uid_to_title.items()}

    # BM25 search if no exact or prefix match is found
    # Prepare corpus
    label_title_corpus = list(label_title_to_uid.keys())
    tokenized_corpus = [bm25_tokenizer(doc) for doc in label_title_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Open the .jsonl file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            json_data = json.loads(line)

            # Assuming the user's message is always the second message in the list
            user_message = json_data[0]['messages'][1]['content']

            # <OLD> Extract the ITEM NUMBER using regex
            # item_number_match = re.search(r'ITEM (\d+):', user_message)
            # <NEW> This regex matches 'Item Index:', then optional whitespace and a newline,
            # followed by optional whitespace, a double quote, the digits, and a closing quote.
            item_number_match = re.search(r'### \*\*Item Index:\*\*\s*\n\s*"(\d+)"', user_message)
            item_number = item_number_match.group(1) if item_number_match else 'No ITEM NUMBER found'

            # Extract the top 5 similar products from the response
            top_products = json_data[1]['choices'][0]['message']['content']
            top_products = re.sub(r'[ ]+', ' ', top_products)

            predicted_products_text = json_data[0]['messages'][1]['content']
            
            # Regular expression pattern to extract the dictionary
            # pattern = r'```python\n({.*?})\n```'
            # pattern = r"```python\n\s*({.*?})\s*\n```"
            pattern = r"```python\n\s*({.*?})\s*\n\s*```"
            top_products_match = re.search(pattern, top_products, re.DOTALL)
            if top_products_match:
                top_products_dict_str = top_products_match.group(1)
            else:
                data_list.append({'item_number': item_number, 'top_products': {}})
                logging.info(f"Could not extract top products in file {file_path} for item number {item_number}. But continuing along...")
                # import pdb; pdb.set_trace()
                continue

            # Regex pattern to extract keys from a dictionary string
            # This pattern looks for anything between single quotes followed immediately by a colon
            pattern = r"'([^']+)':"

            try:
                # Using re.findall to get all keys
                # uids = re.findall(pattern, top_products_dict_str)
                # uids = extract_uids(top_products_dict_str) # Use this for Amazon Dataset
                # uids = extract_uids_wikidataset(top_products_dict_str) # Use this for Wiki Dataset
                # uids = list(ast.literal_eval(top_products_dict_str).keys())# Use this for Wiki Dataset
                try:
                    uids = list(ast.literal_eval(top_products_dict_str).keys())
                except Exception:
                    uids = extract_uids_wikidataset(top_products_dict_str)
                top_products_dict = {}
                # Check if keys were found
                if uids:
                    # top_products_dict = {}
                    for uid in uids:
                        title = label_uid_to_title.get(uid, False)
                        if title: # the label uid was found
                            top_products_dict[uid] = title
                        else: # the label uid was not found 
                            # try to extract the title and get the uid corresponding to the title
                            extracted_title = get_key_value(top_products_dict_str, uid)
                            found_uid = get_with_longest_prefix(label_title_to_uid, extracted_title, bm25, label_title_corpus) if extracted_title else None
                            if found_uid: # could find a correct uid corresponding to the label title 
                                top_products_dict[found_uid] = label_uid_to_title[found_uid]
                            else:
                                # logging.error("did not find the correct uid even with lexical search")
                                if extracted_title in label_uid_to_title: # the extracted title is the uid
                                    # import pdb; pdb.set_trace()
                                    top_products_dict[extracted_title] = label_uid_to_title.get(extracted_title)
                                elif uid in label_title_to_uid: # the uid is the title
                                    top_products_dict[label_title_to_uid[uid]] = uid
                                else:
                                    predicted_uids = re.findall(pattern, predicted_products_text)
                                    closest_uid, min_distance = find_closest_key(predicted_uids, uid)
                                    # logging.info(f"the min dist is {min_distance}")
                                    if min_distance > 1 :
                                        try:
                                            # top_products_dict_str = escape_quotes(top_products_dict_str) # Use this line for Amazon dataset
                                            extracted_title_again = get_key_value(top_products_dict_str, uid)
                                            found_uid_again = get_with_longest_prefix(label_title_to_uid, extracted_title_again, bm25, label_title_corpus, force_lex_search=True) if extracted_title_again else None
                                            if found_uid_again: 
                                                top_products_dict[found_uid_again] = label_uid_to_title[found_uid_again]
                                                continue
                                            corrected_extracted_title = ast.literal_eval(top_products_dict_str)[uid]
                                            corrected_uid =get_with_longest_prefix(label_title_to_uid, corrected_extracted_title, bm25, label_title_corpus)
                                            if corrected_uid: top_products_dict[corrected_uid] = label_uid_to_title[corrected_uid]
                                        except SyntaxError as e:
                                            logging.warning(f"Could not convert top_products into dictionary: file->{Path(file_path).name} Item-> {item_number}")
                                            # import pdb; pdb.set_trace()
                                    else:
                                        top_products_dict[closest_uid] = label_uid_to_title[closest_uid]
                else:
                    logging.error(f"No keys were extracted. Check the format. File: {Path(file_path).name}, Item-> {item_number}")
                    # import pdb; pdb.set_trace()
            except re.error as re_err:
                logging.error(f"Regex error occurred: {re_err}")
                import pdb; pdb.set_trace()
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                import pdb; pdb.set_trace()

            # Append the extracted data to the list
            data_list.append({'item_number': item_number, 'top_products': top_products_dict})

    # Create a DataFrame from the list of data
    df = pd.DataFrame(data_list)
    
    return df

def extract_data_from_jsonl_new(file_path, label_file, label_uid_mapping):
    # Initialize a list to store the data
    data_list = []

    label_uid_to_title = get_label_uid_to_title_dict(label_file)
    label_title_to_uid = {title: uid for uid, title in label_uid_to_title.items()}

    # BM25 search if no exact or prefix match is found
    # Prepare corpus
    label_title_corpus = list(label_title_to_uid.keys())
    tokenized_corpus = [bm25_tokenizer(doc) for doc in label_title_corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Open the .jsonl file and read line by line
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    with open(file_path, "r", encoding="utf-8") as file:
        for line in tqdm(file, total=total_lines, desc=f"Processing ChatGPT ranked top products in {file_path.split('/')[-1]}", unit="lines"):
            # Parse each line as a JSON object
            json_data = json.loads(line)

            # Assuming the user's message is always the second message in the list
            user_message = json_data[0]['messages'][1]['content']

            # <NEW> This regex matches 'Item Index:', then optional whitespace and a newline,
            # followed by optional whitespace, a double quote, the digits, and a closing quote.
            item_number_match = re.search(r'### \*\*Item Index:\*\*\s*\n\s*"(\d+)"', user_message)
            item_number = item_number_match.group(1) if item_number_match else 'No ITEM NUMBER found'

            # Extract the top 5 similar products from the response
            top_products = json_data[1]['choices'][0]['message']['content']
            top_products = re.sub(r'[ ]+', ' ', top_products)
            top_products_dict = {}

            # Regular expression pattern to extract the dictionary
            pattern = r"```python\n\s*({.*?})\s*\n\s*```"
            top_products_match = re.search(pattern, top_products, re.DOTALL)
            if top_products_match:
                top_products_dict_str = top_products_match.group(1)
            else:
                logging.warning(f"Could not extract top products in file {file_path} for item number {item_number}. But continuing along to read the next line...")
                data_list.append({'item_number': item_number, 'top_products': top_products_dict})
                # import pdb; pdb.set_trace()
                continue

            try:
                uids = extract_uids_wikidataset(top_products_dict_str)
                if uids:
                    for uid in uids:
                        original_uid = label_uid_mapping.get(uid)
                        if original_uid:
                            title = label_uid_to_title.get(original_uid, False)
                            top_products_dict[original_uid] = title
                        else:
                            extracted_title = extract_value_from_dict_str(top_products_dict_str, uid)
                            if not extracted_title:
                                extracted_title = ast.literal_eval(top_products_dict_str)[uid]
                            corrected_uid = label_title_to_uid.get(extracted_title, None)
                            if corrected_uid:
                                top_products_dict[corrected_uid] = extracted_title
                            # else:
                                # import pdb; pdb.set_trace()
                                # Regex pattern to extract label title values from user_message to chatgpt
                                # pattern = r"\d+\>\s*'label_uid_\d+':\s*'([^']+)'"
                                # # Find all matches
                                # predicted_label_titles = re.findall(pattern, user_message)
                                # continue
                                # closest_title, min_dist = find_closest_key(label_title_corpus, extracted_title)
                                # _uid =get_with_longest_prefix(label_title_to_uid, extracted_title, bm25, label_title_corpus)
                                # import pdb; pdb.set_trace()
                                # corrected_uid = label_title_to_uid[closest_title]
                                # if corrected_uid:
                                #     top_products_dict[corrected_uid] = closest_title
                                # else:
                                #     logging.info(f"In file {file_path} for item number {item_number} could not get the original uid for {uid}")
                                #     import pdb; pdb.set_trace()
                                #     print('kk')
                else:
                    logging.error(f"No uids were extracted. Check the format. File: {Path(file_path).name}, Item-> {item_number}")

            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                import pdb; pdb.set_trace()

            # Append the extracted data to the list
            data_list.append({'item_number': item_number, 'top_products': top_products_dict})

    # Create a DataFrame from the list of data
    df = pd.DataFrame(data_list)
    
    return df

def load_reverse_label_uid_mapping(mapping_csv_path):
    """
    Load the mapping CSV file and return a reverse mapping dictionary.
    
    The CSV file is expected to have two columns:
      - 'old_label_uid'
      - 'new_label_uid'
    
    The resulting dictionary maps new_label_uid to old_label_uid.
    
    Parameters:
        mapping_csv_path (str): The file path to the mapping CSV.
        
    Returns:
        dict: A dictionary with keys as new_label_uid and values as old_label_uid.
    """
    mapping_df = pd.read_csv(mapping_csv_path)
    reverse_mapping = dict(zip(mapping_df['new_label_uid'], mapping_df['old_label_uid']))
    return reverse_mapping

def extract_data(base_dir, label_file, label_mapping_file):
    jsonl_files = find_jsonl_files(base_dir)
    logging.info(f"There are {len(jsonl_files)} files that are used to collate the reranked results from")
    logging.info(f"Those files are:")
    logging.info("\n".join(jsonl_files))

    reverse_label_uid_mapping = load_reverse_label_uid_mapping(label_mapping_file)
    data_frames = []
    # import pdb; pdb.set_trace()
    for file in jsonl_files:
        # df_extracted = extract_data_from_jsonl(file, label_file)
        df_extracted = extract_data_from_jsonl_new(file, label_file, reverse_label_uid_mapping)
        data_frames.append(df_extracted)
    
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(data_frames, ignore_index=True)

    filename = Path(jsonl_files[0]).parent/'reranked_results.csv'
    combined_df.to_csv(filename, index=False)

    return combined_df

def detailed_safe_eval(s):
    try:
        return ast.literal_eval(s)
    except SyntaxError as e:
        print(f"SyntaxError: {e.msg} at index {e.lineno}")
        return None  

def _is_column_all_dicts(dataframe, col):
    """A function to check if all elements in a DataFrame column are dictionaries."""
    if col in dataframe.columns:
        return dataframe[col].apply(lambda x: isinstance(x, dict)).all()
    else:
        raise ValueError(f"Column {col} does not exist in the DataFrame.")

def get_qrels_results(df, score_col=None):
    if score_col is None: score_col='predicted_scores'
    if not _is_column_all_dicts(df, 'labels'): df['labels'] = df['labels'].apply(ast.literal_eval)
    if not _is_column_all_dicts(df, score_col): df[score_col] = df[score_col].apply(ast.literal_eval)
    qrels = {}
    results= {}
    results_reranked={}
    for index, row in df.iterrows():
        uid = row['uid']

        label_ids = row['labels'].keys()
        qrels[uid] = {label_id: 1 for label_id in label_ids}

        label_id_to_score_dict = row[score_col]
        results[uid] = {label_id: score for label_id, score in label_id_to_score_dict.items()}

    return qrels, results, results_reranked

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

def safe_literal_eval(s, index):
    # Define a function to apply ast.literal_eval safely, logging the index of errors
    try:
        return ast.literal_eval(s)
    except Exception as e:
        # Logging the error along with the row index
        print(f"Error at row {index}: {e} - Value: {s}")
        return None  # Return None or you could choose to return the original string

def reorder_scores(original_scores):
    """
    Reorders the scores of a given dictionary in descending order
    while maintaining the original order of the keys.

    Parameters:
    original_scores (dict): A dictionary with item IDs as keys and scores as values.

    Returns:
    dict: A new dictionary with the same keys in original order but with scores in descending order.
    """
    # Get the original order of keys
    original_order = list(original_scores.keys())
    
    # Sort the original scores in descending order
    sorted_scores = sorted(original_scores.values(), reverse=True)

    # Create a new dictionary with the new scores
    scores_in_original_order = {item_id: sorted_scores[idx] for idx, item_id in enumerate(original_order)}
    
    return scores_in_original_order

def fill_and_sort(chat_scores_reordered, scores, k):
    chat_dict = copy.deepcopy(chat_scores_reordered)
    # Get the keys that are already present.
    existing_keys = set(chat_dict.keys())
    
    # Sort the scores dictionary keys by their score in descending order.
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Fill in chat_scores_reordered with keys from scores that are not already present.
    for key in sorted_keys:
        if len(chat_dict) >= k:
            break
        if key not in existing_keys:
            chat_dict[key] = scores[key]
            existing_keys.add(key)
    
    # Finally, sort chat_scores_reordered by descending score.
    # chat_scores_reordered = dict(sorted(chat_scores_reordered.items(), key=lambda x: x[1], reverse=True))
    return chat_dict

def score_chatgpt_ranks(row, k=5):
    # scores = row['predicted_scores']
    scores = row['topic_margin_cross_agg_predicted_scores']
    ranked = row['chatgpt_preds_ranked']

    if not isinstance(scores, dict): scores = ast.literal_eval(scores)
    if not isinstance(ranked, dict): ranked = ast.literal_eval(ranked)

    k = min(k, len(ranked))

    # if len(ranked) != k:
    # if len(ranked) < k:
        # logging.info(f"ChatGPT extracted {len(ranked)} predictions, but k is {k}. So changing k to {len(ranked)}.")
        # k = len(ranked)

    
    """
    try:
        # Create a dictionary with scores for the ranked items
        chat_scores = {item_id: scores[item_id] for item_id in ranked}
        
        # Reorder the scores based on the original order
        chat_scores_reordered = reorder_scores(chat_scores)

        # return chat_scores_reordered

    except KeyError as e:
        # Handle the case where an item ID in ranked is not found in scores
        # logging.info(f"KeyError: {e} - One of the item IDs in ranked was not found in scores.")
        # Extract the top k scores
        top_k_scores = [value for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
        # Assigning those top k scores to the chatgpt top ranked k labels
        chat_scores_reordered = {item_id: top_k_scores[i] for i, item_id in enumerate(ranked) if i < k}

    except IndexError as e:
        print(f"An index error occurred: {e}")
        import pdb; pdb.set_trace()
    except Exception as e:
        # Handle any other unexpected exceptions
        logging.info(f"An error occurred: {e}")
        import pdb; pdb.set_trace()
        return {}

    """
    # Extract the top k scores
    top_k_scores = [value for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]]
    # Assigning those top k scores to the chatgpt top ranked k labels
    chat_scores_reordered = {item_id: top_k_scores[i] for i, item_id in enumerate(ranked) if i < k}
    return chat_scores_reordered
    # return fill_and_sort(chat_scores_reordered, scores, 5)
    # return reorder_scores(fill_and_sort(chat_scores_reordered, scores, 4))

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

# Function to check if any key in the dictionary is NaN
def contains_nan_key(d):
    if isinstance(d, dict):
        return any(pd.isna(key) for key in d.keys())
    return False  # If it's not a dict, return False

# Function to check and remove NaN keys
def clean_nan_keys(d):
    if isinstance(d, dict):
        nan_keys = [key for key in d.keys() if pd.isna(key)]  # Find NaN keys
        for key in nan_keys:
            del d[key]  # Remove NaN keys
        return d, len(nan_keys)  # Return cleaned dict and count of NaN keys
    return d, 0  # If not a dict, return as is

def main():
    """
    Main function to control the program execution. Parses arguments and calls the appropriate process functions.
    """

    # Setup argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quiet', action='store_true', help='Suppress logging output')
    parser.add_argument("--base_dir", type=str, help="Path to the base directory that will contain intermediate results and models.")
    parser.add_argument("--predicted_file", type=str, help="Path to the csv file containing the predictions on the test set")
    parser.add_argument("--label_mapping_file", type=str, help="Path to the csv file containing the mappings from old label_uid to new label_uid")
    parser.add_argument("--labels_file_name", type=str, help="Path to the file containing label descriptions")
    parser.add_argument("--source_url", type=str, help="URL of the dataset")
    parser.add_argument('--fresh', action='store_true', help='Removes the _chatgpt_reranked.csv file if it exists and starts fresh')
    parser.add_argument('--logfile', type=str, help='Filename to write logs to', default='chatgpt_rerank_logfile')

    args = parser.parse_args()

    source = untar_xxx(eval(args.source_url))
    # Setting the base directory
    global base_dir
    base_dir = Path(args.base_dir + '/' + args.source_url.split('.')[1])
    label_file = Path(source/args.labels_file_name)
    label_mapping_file = base_dir/args.label_mapping_file

    # Setup logger and print arguments
    setup_logging(args, base_dir)
    # print_args(args)

    logging.info("Program started")

    chatgpt_reranked_file =base_dir/(args.predicted_file.split('.')[0]+'_chatgpt_reranked.csv') 
    if args.fresh and chatgpt_reranked_file.exists(): chatgpt_reranked_file.unlink()
    if chatgpt_reranked_file.exists():
        logging.info(f"The CHATGPT reranked file already exists as {chatgpt_reranked_file}. Loading it now!!!")
        df_predicted = pd.read_csv(chatgpt_reranked_file)
    
    else: # doing it for the first time
        df_extracted = extract_data(base_dir, label_file, label_mapping_file)
    
        try:
            # Load the predictions file if it is available
            df_predicted = pd.read_csv(base_dir/args.predicted_file)

            df_extracted['item_number'] = pd.to_numeric(df_extracted['item_number'], errors='coerce')
            contains_nans = df_extracted['item_number'].isna()
            if contains_nans.any():
                logging.info(f"Number of NaNs in the extracted = {contains_nans.sum()}")

            # Sort df_extracted by 'item_number'
            df_extracted = df_extracted.sort_values(by='item_number').reset_index(drop=True)
            
            # Check if each element in column 'top_products' is a dictionary
            is_dict = df_extracted['top_products'].apply(lambda x: isinstance(x, dict))
            all_are_dicts = is_dict.all()
            logging.info(f"All the top products are dictionary: {all_are_dicts}")

            is_empty_dict = df_extracted['top_products'].apply(lambda x: isinstance(x, dict) and x=={})
            logging.info(f"Number of empty dictionaries in top products: {is_empty_dict.sum()}")

            # Remove any nans in the top products
            df_extracted['top_products'], nan_counts = zip(*df_extracted['top_products'].apply(clean_nan_keys))
            logging.info(f"Total NaN keys removed from `df_extracted`: {sum(nan_counts)}")

            # df_predicted['chatgpt_preds'] = df_extracted['top_products']

            # Ensure 'item_number' is the same type as the index or matching column in df_predicted
            df_extracted['item_number'] = df_extracted['item_number'].astype(df_predicted.index.dtype)

            # Create a dictionary from df_extracted for mapping `item_number` to `top_products` 
            product_map = df_extracted.set_index('item_number')['top_products'].to_dict()

            # Map the 'top_products' to a new column in df_predicted based on the index
            df_predicted['chatgpt_preds_ranked'] = df_predicted.index.map(product_map)

            # if the 'chatgpt_preds_ranked' col is empty {} (that is chatgpt did not give any preds) then replace it with another column 
            df_predicted['chatgpt_preds_ranked'] = df_predicted.apply(lambda row: row['topic_margin_cross_agg_predicted_scores'] if row['chatgpt_preds_ranked'] == {} else row['chatgpt_preds_ranked'], axis=1)

            # Score the ChatGPT ranks in order to perform metric computation later
            # df_predicted['chatgpt_preds_scored'] = df_predicted.apply(score_chatgpt_ranks, axis=1)
            df_predicted['chatgpt_preds_scored'] = df_predicted.apply(
                lambda row: score_chatgpt_ranks(row) if not pd.isna(row['chatgpt_preds_ranked']) else pd.NA, axis=1
            )

            # Replace NaN values in 'chatgpt_preds_ranked' with values from 'topic_margin_cross_agg_predicted_scores'
            if not _is_column_all_dicts(df_predicted, 'topic_margin_cross_agg_predicted_scores'):
                df_predicted['topic_margin_cross_agg_predicted_scores'] = df_predicted['topic_margin_cross_agg_predicted_scores'].apply(ast.literal_eval)
            df_predicted['chatgpt_preds_scored'] = df_predicted['chatgpt_preds_scored'].fillna(df_predicted['topic_margin_cross_agg_predicted_scores'])

            # Save
            df_predicted.to_csv(chatgpt_reranked_file, index=False)
            logging.info(f"The CHATGPT reranked is created and file is saved at {chatgpt_reranked_file}.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Make sure you have the predictions first in the file {base_dir/args.predicted_file}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    # import pdb; pdb.set_trace()
    # col_agg1, col_agg2='topic_margin_cross_agg_predicted_scores', 'chatgpt_preds_scored'
    # if not _is_column_all_dicts(df_predicted, col_agg1): df_predicted[col_agg1] = df_predicted[col_agg1].apply(ast.literal_eval)
    # if not _is_column_all_dicts(df_predicted, col_agg2): df_predicted[col_agg2] = df_predicted[col_agg2].apply(ast.literal_eval)
    # df_predicted['trial_scores'] = df_predicted.apply(aggregate_two_scores, axis=1, col_agg1=col_agg1, col_agg2=col_agg2)
    # df_predicted['trial_scores'] = df_predicted['trial_scores'].apply(lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True)))


    # Computing metrics
    for score_col in ('predicted_scores', 'predicted_scores_reranked', 'topic_margin_cross_agg_predicted_scores', 'chatgpt_preds_scored'):
        # Check if the column exists and if it contains any NaN values
        if score_col not in df_predicted or df_predicted[score_col].isna().any():
            logging.warning(f"Skipping column {score_col} because it is either missing or contains NaN values.")
            continue  # Skip to the next column
        
        logging.info(f"Using column {score_col}, the metrics are as follows:")
        
        # Get qrels and results for the score column
        qrels, results, _ = get_qrels_results(df_predicted, score_col=score_col)
        
        # Call the evaluation function
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, k_values=[1, 3, 5, 10], ignore_identical_ids=True
        )

    logging.info("Program ended successfully")

# Ensure the script can be used as a script as well as an importable module
if __name__ == "__main__":
    main()
