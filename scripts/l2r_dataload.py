#!/usr/bin/env python
# =============================================================================
# L2R Data Load Script
# =============================================================================

import os
import logging
import argparse
import random
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from datetime import datetime
from tabulate import tabulate
import json


def setup_logging(logfile="l2r_load", **kwargs):
    """Set up logging configuration"""
    log_dir = os.path.join(kwargs.get('output_dir', '.'), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Format the current date and time for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = (
        f"{logfile}_{current_time}.log"
        if logfile
        else f"default_logfile_{current_time}.log"
    )
    logfile_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()
        ]
    )
    return logfile_path


def join_path_file(fname, path, ext=''):
    """Join path and filename with extension"""
    return Path(path) / f"{fname}{ext}"


def print_args(**kwargs):
    """Print function arguments"""
    logging.info("Arguments:")
    for key, value in kwargs.items():
        logging.info(f"  {key}: {value}")


# Global variable to store generated files
generated_files = []


# =============================================================================
# Loading Function
# =============================================================================
def load_l2r_data(
    output_dir: str,
    fname: str
) -> Dict[str, Union[pd.DataFrame, List]]:
    """
    Load Learning-to-Rank data from saved files
    
    Args:
        output_dir: Directory containing the saved files
        fname: Base filename used when saving the data
    
    Returns:
        Dictionary containing loaded dataframes and related data
    """
    global generated_files
    generated_files = []
    
    logging.info(f"Loading Learning-to-Rank data from {output_dir}")
    
    # Create paths for the files
    token_label_rankings_file = join_path_file(fname + '_tok_rank_per_lbl', Path(output_dir), ext='.ft')
    tokens_file = join_path_file(fname + '_tok', Path(output_dir), ext='.ft')
    labels_file = join_path_file(fname + '_lbl', Path(output_dir), ext='.ft')
    
    # Check that all required files exist
    files_to_check = [
        (token_label_rankings_file, "Token-label rankings"),
        (tokens_file, "Token mappings"),
        (labels_file, "Label mappings")
    ]
    
    for file_path, desc in files_to_check:
        if not file_path.exists():
            logging.error(f"Required file not found: {file_path} ({desc})")
            raise FileNotFoundError(f"Required file not found: {file_path}")
        else:
            logging.info(f"Found {desc} file: {file_path}")
    
    # Load the dataframes
    logging.info("Loading token-label rankings...")
    df_l2r = pd.read_feather(token_label_rankings_file)
    logging.info(f"Loaded token-label rankings with shape: {df_l2r.shape}")
    
    logging.info("Loading token mappings...")
    df_toks = pd.read_feather(tokens_file)
    logging.info(f"Loaded {len(df_toks)} token mappings")
    
    logging.info("Loading label mappings...")
    df_lbs = pd.read_feather(labels_file)
    logging.info(f"Loaded {len(df_lbs)} label mappings")
    
    # Extract token and label lists
    toks = df_toks['tok_val'].tolist()
    lbs = df_lbs['lbl_val'].tolist()
    
    # Create summary of loaded data
    loaded_data = {
        'df_l2r': df_l2r,
        'df_toks': df_toks,
        'df_lbs': df_lbs,
        'toks': toks,
        'lbs': lbs
    }
    
    # Add to generated files list for reporting
    generated_files.extend([
        (token_label_rankings_file, "DataFrame containing token-label pairs with mutual information scores and rankings"),
        (tokens_file, "DataFrame mapping token IDs to their string values"),
        (labels_file, "DataFrame mapping label IDs to their string values")
    ])
    
    # Print summary of loaded files
    print("\n" + "="*80)
    print("LOADED FILES SUMMARY")
    print("="*80)
    for file_path, description in generated_files:
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1]
        file_size = os.path.getsize(file_path)
        
        # Format file size for human readability
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024*1024:
            size_str = f"{file_size/1024:.2f} KB"
        else:
            size_str = f"{file_size/(1024*1024):.2f} MB"
            
        print(f"File: {file_path}")
        print(f"  - Type: {file_ext[1:].upper()} file")
        print(f"  - Size: {size_str}")
        print(f"  - Description: {description}")
        print("-"*80)
    
    return loaded_data


# =============================================================================
# Analysis Functions
# =============================================================================
def analyze_token_label_rankings_old(df_l2r: pd.DataFrame, df_toks: pd.DataFrame, df_lbs: pd.DataFrame, k:int=75) -> Dict:
    """
    Analyze the token-label rankings dataframe
    
    Args:
        df_l2r: Token-label rankings dataframe
        df_toks: Token mappings dataframe
        df_lbs: Label mappings dataframe
        k: the cutoff for most informative tokens and labels
        
    Returns:
        Dictionary with analysis results
    """
    logging.info("Analyzing token-label rankings data...")
    
    # Create a merged dataframe with token and label values
    df_merged = df_l2r.merge(df_toks, on='token').merge(df_lbs, left_on='label', right_on='lbl')
    
    results = {}
    
    # --- Overall statistics ---
    
    # Find labels with highest overall mutual information scores
    top_labels_overall = df_merged.groupby('lbl_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(k)
    results['top_labels_overall'] = top_labels_overall.to_dict()
    
    # Find tokens with highest overall mutual information across all labels
    top_tokens_overall = df_merged.groupby('tok_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(k)
    results['top_tokens_overall'] = top_tokens_overall.to_dict()
    
    # Get summary statistics for mutual information scores
    info_stats = {
        'mean': float(df_l2r['bcx_mutual_info'].mean()),
        'median': float(df_l2r['bcx_mutual_info'].median()),
        'min': float(df_l2r['bcx_mutual_info'].min()),
        'max': float(df_l2r['bcx_mutual_info'].max()),
        'std': float(df_l2r['bcx_mutual_info'].std())
    }
    results['info_stats'] = info_stats
    
    # --- Token-specific analysis ---
    
    # For each token, find the top k most informative labels
    logging.info(f"Computing top {k} most informative labels for each token...")
    
    # Create a dictionary to store results
    token_to_top_labels = {}
    
    # Get unique tokens
    unique_tokens = df_toks['tok_val'].unique()
    
    # For reporting progress
    total_tokens = len(unique_tokens)
    
    for i, token_val in enumerate(unique_tokens):
        if i % 1000 == 0 and i > 0:
            logging.info(f"Processed {i}/{total_tokens} tokens...")
            
        # Get token ID
        token_id = df_toks[df_toks['tok_val'] == token_val]['token'].values[0]
        
        # Filter data for this token
        token_data = df_merged[df_merged['token'] == token_id]
        
        # Get top 20 labels by mutual information
        top_labels = token_data.sort_values('bcx_mutual_info', ascending=False).head(k)[['lbl_val', 'bcx_mutual_info']]
        
        # Store results as a dictionary
        token_to_top_labels[token_val] = dict(zip(top_labels['lbl_val'], top_labels['bcx_mutual_info']))
    
    results['token_to_top_labels'] = token_to_top_labels
    
    # --- Label-specific analysis ---
    
    # For each label, find the top k most informative tokens
    logging.info(f"Computing top {k} most informative tokens for each label...")
    
    # Create a dictionary to store results
    label_to_top_tokens = {}
    
    # Get unique labels
    unique_labels = df_lbs['lbl_val'].unique()
    
    # For reporting progress
    total_labels = len(unique_labels)
    
    for i, label_val in enumerate(unique_labels):
        if i % 100 == 0 and i > 0:
            logging.info(f"Processed {i}/{total_labels} labels...")
            
        # Get label ID
        label_id = df_lbs[df_lbs['lbl_val'] == label_val]['lbl'].values[0]
        
        # Filter data for this label
        label_data = df_merged[df_merged['label'] == label_id]
        
        # Get top 50 tokens by mutual information
        top_tokens = label_data.sort_values('bcx_mutual_info', ascending=False).head(k)[['tok_val', 'bcx_mutual_info']]
        
        # Store results as a dictionary
        label_to_top_tokens[label_val] = dict(zip(top_tokens['tok_val'], top_tokens['bcx_mutual_info']))
    
    results['label_to_top_tokens'] = label_to_top_tokens
    
    logging.info("Analysis complete")
    
    return results


def analyze_token_label_rankings(df_l2r: pd.DataFrame, df_toks: pd.DataFrame, df_lbs: pd.DataFrame, k:int=75) -> Dict:
    """
    Analyze the token-label rankings dataframe
    
    Args:
        df_l2r: Token-label rankings dataframe
        df_toks: Token mappings dataframe
        df_lbs: Label mappings dataframe
        k: the cutoff for most informative tokens and labels
        
    Returns:
        Dictionary with analysis results
    """
    logging.info("Analyzing token-label rankings data...")
    
    # Create a merged dataframe with token and label values
    df_merged = df_l2r.merge(df_toks, on='token').merge(df_lbs, left_on='label', right_on='lbl')
    
    results = {}
    
    # --- Overall statistics ---
    
    # Find labels with highest overall mutual information scores
    top_labels_overall = df_merged.groupby('lbl_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(k)
    results['top_labels_overall'] = top_labels_overall.to_dict()
    
    # Find tokens with highest overall mutual information across all labels
    top_tokens_overall = df_merged.groupby('tok_val')['bcx_mutual_info'].mean().sort_values(ascending=False).head(k)
    results['top_tokens_overall'] = top_tokens_overall.to_dict()
    
    # Get summary statistics for mutual information scores
    info_stats = {
        'mean': float(df_l2r['bcx_mutual_info'].mean()),
        'median': float(df_l2r['bcx_mutual_info'].median()),
        'min': float(df_l2r['bcx_mutual_info'].min()),
        'max': float(df_l2r['bcx_mutual_info'].max()),
        'std': float(df_l2r['bcx_mutual_info'].std())
    }
    results['info_stats'] = info_stats
    
    # --- Token-specific analysis ---
    
    # For each token, find the top k most informative labels
    logging.info(f"Computing top {k} most informative labels for each token...")
    
    token_groups = df_merged.groupby('tok_val')
    token_to_top_labels = {}
    
    for token_val, group in tqdm(token_groups, desc="Processing tokens"):
        top_labels = group.nlargest(k, 'bcx_mutual_info')[['lbl_val', 'bcx_mutual_info']]
        token_to_top_labels[token_val] = dict(zip(top_labels['lbl_val'], top_labels['bcx_mutual_info']))
    
    results['token_to_top_labels'] = token_to_top_labels
    
    # --- Label-specific analysis ---
    
    # For each label, find the top k most informative tokens
    logging.info(f"Computing top {k} most informative tokens for each label...")
    
    label_groups = df_merged.groupby('lbl_val')
    label_to_top_tokens = {}
    
    for label_val, group in tqdm(label_groups, desc="Processing labels"):
        top_tokens = group.nlargest(k, 'bcx_mutual_info')[['tok_val', 'bcx_mutual_info']]
        label_to_top_tokens[label_val] = dict(zip(top_tokens['tok_val'], top_tokens['bcx_mutual_info']))
    
    results['label_to_top_tokens'] = label_to_top_tokens
    
    logging.info("Analysis complete")
    
    return results


# =============================================================================
# Report Generating Functions
# =============================================================================
def generate_icd10_token_report(analysis_results: dict, output_dir: str,
                               max_tokens_per_code: int = 20,
                               max_labels_per_token: int = 15,
                               include_summary: bool = True,
                               sort_codes: bool = True,
                               sort_tokens: bool = True) -> dict:
    """
    Generates and saves Markdown and JSON reports for ICD-10 code token analysis with mutual information scores.
    Creates separate reports for label-to-tokens and token-to-labels mappings.
    
    Args:
        analysis_results: Dictionary containing 'label_to_top_tokens' and 'token_to_top_labels' keys.
        output_dir: Directory to save the report files.
        max_tokens_per_code: Maximum number of top tokens to display per ICD-10 code (default: 20).
        max_labels_per_token: Maximum number of top labels to display per token (default: 15).
        include_summary: Whether to include a summary section in the reports (default: True).
        sort_codes: Whether to sort ICD-10 codes alphabetically (default: True).
        sort_tokens: Whether to sort tokens alphabetically (default: True).
    
    Returns:
        dict: Paths to generated files and summary statistics.
    
    Raises:
        ValueError: If analysis_results is invalid or missing required keys.
        OSError: If unable to create output directory or write files.
    """
    # Input validation
    if not analysis_results:
        raise ValueError("analysis_results cannot be empty")
    
    required_keys = ['label_to_top_tokens', 'token_to_top_labels']
    missing_keys = [key for key in required_keys if key not in analysis_results]
    if missing_keys:
        raise ValueError(f"analysis_results must contain keys: {missing_keys}")
    
    label_to_tokens = analysis_results['label_to_top_tokens']
    token_to_labels = analysis_results['token_to_top_labels']
    
    if not label_to_tokens:
        raise ValueError("label_to_top_tokens cannot be empty")
    
    if not token_to_labels:
        raise ValueError("token_to_top_labels cannot be empty")
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory {output_dir}: {e}")
    
    logging.info(f"🏥 Generating ICD-10 Token Analysis Reports")
    
    # Generate both reports
    files_created = {}
    all_stats = {}
    
    # 1. Generate Label-to-Tokens Report
    label_report_result = _generate_label_to_tokens_report(
        label_to_tokens, output_dir, timestamp, max_tokens_per_code, 
        include_summary, sort_codes
    )
    files_created.update(label_report_result['files'])
    all_stats['label_to_tokens'] = label_report_result['stats']
    
    # 2. Generate Token-to-Labels Report
    token_report_result = _generate_token_to_labels_report(
        token_to_labels, output_dir, timestamp, max_labels_per_token,
        include_summary, sort_tokens
    )
    files_created.update(token_report_result['files'])
    all_stats['token_to_labels'] = token_report_result['stats']
    
    logging.info(f"✅ All ICD-10 analysis reports generated successfully!")
    
    # Return comprehensive summary
    return {
        'files_created': files_created,
        'summary_stats': all_stats,
        'timestamp': timestamp,
        'total_codes_processed': len(label_to_tokens),
        'total_tokens_processed': len(token_to_labels),
        'total_label_token_pairs': sum(len(tokens) for tokens in label_to_tokens.values()),
        'total_token_label_pairs': sum(len(labels) for labels in token_to_labels.values())
    }


def _generate_label_to_tokens_report(label_to_tokens: dict, output_dir: str, timestamp: str,
                                   max_tokens_per_code: int, include_summary: bool, 
                                   sort_codes: bool) -> dict:
    """Generate the ICD-10 codes to tokens report."""
    
    # Get ICD-10 codes
    icd_codes = list(label_to_tokens.keys())
    if sort_codes:
        icd_codes.sort()
    
    # Generate summary statistics
    summary_stats = {}
    if include_summary:
        total_codes = len(icd_codes)
        total_tokens = sum(len(tokens) for tokens in label_to_tokens.values())
        
        token_counts = {code: len(tokens) for code, tokens in label_to_tokens.items()}
        max_tokens_code = max(token_counts, key=token_counts.get) if token_counts else None
        min_tokens_code = min(token_counts, key=token_counts.get) if token_counts else None
        
        all_scores = []
        for tokens in label_to_tokens.values():
            all_scores.extend(tokens.values())
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0
        
        summary_stats = {
            'total_icd_codes': total_codes,
            'total_tokens_analyzed': total_tokens,
            'average_tokens_per_code': total_tokens / total_codes if total_codes > 0 else 0,
            'code_with_most_tokens': max_tokens_code,
            'code_with_least_tokens': min_tokens_code,
            'max_tokens_for_code': token_counts.get(max_tokens_code, 0) if max_tokens_code else 0,
            'min_tokens_for_code': token_counts.get(min_tokens_code, 0) if min_tokens_code else 0,
            'mutual_information_stats': {
                'average_score': avg_score,
                'min_score': min_score,
                'max_score': max_score,
                'score_range': max_score - min_score
            }
        }
    
    # Generate Markdown report
    report_md = "# 🏥 ICD-10 Codes → Top Tokens Report\n\n"
    report_md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    report_md += "This report shows the top tokens associated with each ICD-10 code based on mutual information scores.\n\n"
    
    # Add summary section
    if include_summary and summary_stats:
        report_md += "## 📊 Summary Statistics\n\n"
        report_md += f"- **Total ICD-10 Codes Analyzed:** {summary_stats['total_icd_codes']}\n"
        report_md += f"- **Total Tokens Analyzed:** {summary_stats['total_tokens_analyzed']:,}\n"
        report_md += f"- **Average Tokens per Code:** {summary_stats['average_tokens_per_code']:.1f}\n"
        report_md += f"- **Code with Most Tokens:** {summary_stats['code_with_most_tokens']} ({summary_stats['max_tokens_for_code']} tokens)\n"
        report_md += f"- **Code with Least Tokens:** {summary_stats['code_with_least_tokens']} ({summary_stats['min_tokens_for_code']} tokens)\n\n"
        
        mi_stats = summary_stats['mutual_information_stats']
        report_md += "### 🧮 Mutual Information Score Statistics\n\n"
        report_md += f"- **Average Score:** {mi_stats['average_score']:.4f}\n"
        report_md += f"- **Score Range:** {mi_stats['min_score']:.4f} to {mi_stats['max_score']:.4f}\n"
        report_md += f"- **Total Range:** {mi_stats['score_range']:.4f}\n\n"
        report_md += "---\n\n"
    
    # Process each ICD-10 code
    for idx, icd_code in tqdm(enumerate(icd_codes, 1), 
                              total=len(icd_codes), 
                              desc="🔍 Processing ICD-10 Codes → Tokens", 
                              unit="code"):
        
        tokens_data = label_to_tokens.get(icd_code, {})
        
        report_md += f"## 🏷️ ICD-10 Code: `{icd_code}`\n\n"
        
        if not tokens_data:
            report_md += "⚠️ *No token data available for this ICD-10 code.*\n\n"
            continue
        
        # Sort tokens by mutual information score (descending order)
        sorted_tokens = sorted(tokens_data.items(), key=lambda x: x[1], reverse=True)
        tokens_to_show = sorted_tokens[:max_tokens_per_code]
        
        # Add metadata
        total_tokens = len(sorted_tokens)
        best_score = sorted_tokens[0][1] if sorted_tokens else 0
        worst_score = sorted_tokens[-1][1] if sorted_tokens else 0
        
        report_md += f"📈 **Total Tokens:** {total_tokens} | "
        report_md += f"**Best MI Score:** {best_score:.4f} | "
        report_md += f"**Worst MI Score:** {worst_score:.4f}\n\n"
        
        # Create table
        table_data = []
        for rank, (token_name, mi_score) in enumerate(tokens_to_show, 1):
            rank_emoji = _get_rank_emoji(rank)
            clean_token = token_name.replace('▁', ' ').strip()
            if not clean_token:
                clean_token = token_name
            
            table_data.append([
                f"{rank_emoji} {rank}",
                f"`{clean_token}`",
                f"`{token_name}`",
                f"{mi_score:.6f}"
            ])
        
        table = tabulate(
            table_data,
            headers=['🏅 Rank', '🔤 Token (Clean)', '🔍 Token (Raw)', '📊 MI Score'],
            tablefmt="pipe"
        )
        
        report_md += table + "\n"
        
        if len(sorted_tokens) > max_tokens_per_code:
            remaining = len(sorted_tokens) - max_tokens_per_code
            report_md += f"\n💡 *Showing top {max_tokens_per_code} tokens. {remaining} more tokens available.*\n"
        
        report_md += "\n---\n\n"
    
    # Save files
    files_created = {}
    
    # Save Markdown report
    report_path = os.path.join(output_dir, f"icd10_codes_to_tokens_{timestamp}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    files_created['codes_to_tokens_markdown'] = report_path
    logging.info(f"📄 ICD-10 codes→tokens report saved to {report_path} ✅")
    
    # Prepare JSON data
    json_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'ICD-10 Codes to Top Tokens',
            'total_codes_analyzed': len(icd_codes),
            'max_tokens_per_code_displayed': max_tokens_per_code,
            'codes_sorted': sort_codes
        },
        'summary_statistics': summary_stats if include_summary else {},
        'icd_codes_analyzed': icd_codes,
        'codes_to_tokens_analysis': {}
    }
    
    for icd_code in icd_codes:
        tokens_data = label_to_tokens.get(icd_code, {})
        if tokens_data:
            sorted_tokens = sorted(tokens_data.items(), key=lambda x: x[1], reverse=True)
            
            json_data['codes_to_tokens_analysis'][icd_code] = {
                'total_tokens': len(sorted_tokens),
                'best_mi_score': sorted_tokens[0][1] if sorted_tokens else None,
                'worst_mi_score': sorted_tokens[-1][1] if sorted_tokens else None,
                'top_tokens': [
                    {
                        'rank': idx + 1,
                        'token': token,
                        'clean_token': token.replace('▁', ' ').strip(),
                        'mutual_information_score': score
                    }
                    for idx, (token, score) in enumerate(sorted_tokens[:max_tokens_per_code])
                ],
                'all_tokens': dict(sorted_tokens)
            }
    
    # Save JSON report
    json_report_path = os.path.join(output_dir, f"icd10_codes_to_tokens_{timestamp}.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    files_created['codes_to_tokens_json'] = json_report_path
    logging.info(f"🗂️ ICD-10 codes→tokens JSON saved to {json_report_path} ✅")
    
    return {
        'files': files_created,
        'stats': summary_stats
    }


def _generate_token_to_labels_report(token_to_labels: dict, output_dir: str, timestamp: str,
                                   max_labels_per_token: int, include_summary: bool,
                                   sort_tokens: bool) -> dict:
    """Generate the tokens to ICD-10 codes report."""
    
    # Get tokens
    tokens = list(token_to_labels.keys())
    if sort_tokens:
        tokens.sort()
    
    # Generate summary statistics
    summary_stats = {}
    if include_summary:
        total_tokens = len(tokens)
        total_labels = sum(len(labels) for labels in token_to_labels.values())
        
        label_counts = {token: len(labels) for token, labels in token_to_labels.items()}
        max_labels_token = max(label_counts, key=label_counts.get) if label_counts else None
        min_labels_token = min(label_counts, key=label_counts.get) if label_counts else None
        
        all_scores = []
        for labels in token_to_labels.values():
            all_scores.extend(labels.values())
        
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 0
        
        # Get unique ICD codes across all tokens
        unique_codes = set()
        for labels in token_to_labels.values():
            unique_codes.update(labels.keys())
        
        summary_stats = {
            'total_tokens': total_tokens,
            'total_labels_analyzed': total_labels,
            'unique_icd_codes': len(unique_codes),
            'average_labels_per_token': total_labels / total_tokens if total_tokens > 0 else 0,
            'token_with_most_labels': max_labels_token,
            'token_with_least_labels': min_labels_token,
            'max_labels_for_token': label_counts.get(max_labels_token, 0) if max_labels_token else 0,
            'min_labels_for_token': label_counts.get(min_labels_token, 0) if min_labels_token else 0,
            'mutual_information_stats': {
                'average_score': avg_score,
                'min_score': min_score,
                'max_score': max_score,
                'score_range': max_score - min_score
            }
        }
    
    # Generate Markdown report
    report_md = "# 🔤 Tokens → Top ICD-10 Codes Report\n\n"
    report_md += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    report_md += "This report shows the top ICD-10 codes associated with each token based on mutual information scores.\n\n"
    
    # Add summary section
    if include_summary and summary_stats:
        report_md += "## 📊 Summary Statistics\n\n"
        report_md += f"- **Total Tokens Analyzed:** {summary_stats['total_tokens']:,}\n"
        report_md += f"- **Total Token-Label Associations:** {summary_stats['total_labels_analyzed']:,}\n"
        report_md += f"- **Unique ICD-10 Codes:** {summary_stats['unique_icd_codes']}\n"
        report_md += f"- **Average Labels per Token:** {summary_stats['average_labels_per_token']:.1f}\n"
        report_md += f"- **Token with Most Labels:** `{summary_stats['token_with_most_labels']}` ({summary_stats['max_labels_for_token']} labels)\n"
        report_md += f"- **Token with Least Labels:** `{summary_stats['token_with_least_labels']}` ({summary_stats['min_labels_for_token']} labels)\n\n"
        
        mi_stats = summary_stats['mutual_information_stats']
        report_md += "### 🧮 Mutual Information Score Statistics\n\n"
        report_md += f"- **Average Score:** {mi_stats['average_score']:.4f}\n"
        report_md += f"- **Score Range:** {mi_stats['min_score']:.4f} to {mi_stats['max_score']:.4f}\n"
        report_md += f"- **Total Range:** {mi_stats['score_range']:.4f}\n\n"
        report_md += "---\n\n"
    
    # Process each token
    for idx, token in tqdm(enumerate(tokens, 1), 
                          total=len(tokens), 
                          desc="🔤 Processing Tokens → ICD-10 Codes", 
                          unit="token"):
        
        labels_data = token_to_labels.get(token, {})
        
        # Clean token for display
        clean_token = token.replace('▁', ' ').strip()
        if not clean_token:
            clean_token = token
        
        report_md += f"## 🎯 Token: `{clean_token}` (Raw: `{token}`)\n\n"
        
        if not labels_data:
            report_md += "⚠️ *No ICD-10 code data available for this token.*\n\n"
            continue
        
        # Sort labels by mutual information score (descending order)
        sorted_labels = sorted(labels_data.items(), key=lambda x: x[1], reverse=True)
        labels_to_show = sorted_labels[:max_labels_per_token]
        
        # Add metadata
        total_labels = len(sorted_labels)
        best_score = sorted_labels[0][1] if sorted_labels else 0
        worst_score = sorted_labels[-1][1] if sorted_labels else 0
        
        report_md += f"📈 **Total Associated ICD-10 Codes:** {total_labels} | "
        report_md += f"**Best MI Score:** {best_score:.4f} | "
        report_md += f"**Worst MI Score:** {worst_score:.4f}\n\n"
        
        # Create table
        table_data = []
        for rank, (icd_code, mi_score) in enumerate(labels_to_show, 1):
            rank_emoji = _get_rank_emoji(rank)
            
            table_data.append([
                f"{rank_emoji} {rank}",
                f"`{icd_code}`",
                f"{mi_score:.6f}"
            ])
        
        table = tabulate(
            table_data,
            headers=['🏅 Rank', '🏷️ ICD-10 Code', '📊 MI Score'],
            tablefmt="pipe"
        )
        
        report_md += table + "\n"
        
        if len(sorted_labels) > max_labels_per_token:
            remaining = len(sorted_labels) - max_labels_per_token
            report_md += f"\n💡 *Showing top {max_labels_per_token} ICD-10 codes. {remaining} more codes available.*\n"
        
        report_md += "\n---\n\n"
    
    # Save files
    files_created = {}
    
    # Save Markdown report
    report_path = os.path.join(output_dir, f"tokens_to_icd10_codes_{timestamp}.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    files_created['tokens_to_codes_markdown'] = report_path
    logging.info(f"📄 Tokens→ICD-10 codes report saved to {report_path} ✅")
    
    # Prepare JSON data
    json_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'Tokens to Top ICD-10 Codes',
            'total_tokens_analyzed': len(tokens),
            'max_labels_per_token_displayed': max_labels_per_token,
            'tokens_sorted': sort_tokens
        },
        'summary_statistics': summary_stats if include_summary else {},
        'tokens_analyzed': tokens,
        'tokens_to_codes_analysis': {}
    }
    
    for token in tokens:
        labels_data = token_to_labels.get(token, {})
        if labels_data:
            sorted_labels = sorted(labels_data.items(), key=lambda x: x[1], reverse=True)
            
            json_data['tokens_to_codes_analysis'][token] = {
                'clean_token': token.replace('▁', ' ').strip(),
                'total_labels': len(sorted_labels),
                'best_mi_score': sorted_labels[0][1] if sorted_labels else None,
                'worst_mi_score': sorted_labels[-1][1] if sorted_labels else None,
                'top_labels': [
                    {
                        'rank': idx + 1,
                        'icd_code': icd_code,
                        'mutual_information_score': score
                    }
                    for idx, (icd_code, score) in enumerate(sorted_labels[:max_labels_per_token])
                ],
                'all_labels': dict(sorted_labels)
            }
    
    # Save JSON report
    json_report_path = os.path.join(output_dir, f"tokens_to_icd10_codes_{timestamp}.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    files_created['tokens_to_codes_json'] = json_report_path
    logging.info(f"🗂️ Tokens→ICD-10 codes JSON saved to {json_report_path} ✅")
    
    return {
        'files': files_created,
        'stats': summary_stats
    }


def _get_rank_emoji(rank: int) -> str:
    """Get appropriate emoji for ranking position."""
    if rank == 1:
        return "🥇"
    elif rank == 2:
        return "🥈"
    elif rank == 3:
        return "🥉"
    elif rank <= 5:
        return "⭐"
    elif rank <= 10:
        return "🔸"
    else:
        return "▫️"


# =============================================================================
# Main Function
# =============================================================================
def main(
    output_dir: str = "task2_output",
    data: str = "mimic4_icd10_full",
    base_dir: str = None,
    source_url: str = "XURLs.MIMIC4_L2R",
    logfile: str = "l2r_load",
    run_analysis: bool = True
):
    """
    Main function to load L2R data from previously generated files
    
    Args:
        output_dir: Directory containing the saved files
        data: Dataset name
        base_dir: Base directory for the project
        source_url: Source URL for the data
        logfile: Name of the log file
        run_analysis: Whether to run analysis on the loaded data
    """
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------
    global generated_files
    generated_files = []
    
    # Setup path
    fname = '_'.join(data.split('_')[:-1])
    if base_dir is not None:
        base_dir = base_dir + '/' + source_url.split('.')[1]
        output_dir = str(Path(base_dir) / output_dir)
    
    # Setup logging
    logfile_path = setup_logging(logfile=logfile, output_dir=output_dir)
    print(f"You can tail the logfile: {logfile_path} to check logs")
    print_args(**locals())
    
    logging.info("****L2R Data Loading Started****")
    
    # -----------------------------------------------------------------------------
    # Load Data
    # -----------------------------------------------------------------------------
    try:
        loaded_data = load_l2r_data(output_dir, fname)
        logging.info("Successfully loaded all L2R data")
        
        # Basic validation
        df_l2r = loaded_data['df_l2r']
        df_toks = loaded_data['df_toks']
        df_lbs = loaded_data['df_lbs']
        
        logging.info(f"Token-label rankings shape: {df_l2r.shape}")
        logging.info(f"Number of unique tokens: {df_toks.shape[0]}")
        logging.info(f"Number of unique labels: {df_lbs.shape[0]}")
        
        # Validate relationships
        expected_pairs = len(loaded_data['toks']) * len(loaded_data['lbs'])
        actual_pairs = len(df_l2r)
        
        if expected_pairs == actual_pairs:
            logging.info(f"Data validation successful: Found expected {actual_pairs} token-label pairs")
        else:
            logging.warning(f"Data validation issue: Expected {expected_pairs} token-label pairs, found {actual_pairs}")
        
        # -----------------------------------------------------------------------------
        # Analysis (Optional)
        # -----------------------------------------------------------------------------
        if run_analysis:
            analysis_results = analyze_token_label_rankings(df_l2r, df_toks, df_lbs)
            
            # Create section dividers
            section_header = "="*80
            subsection_header = "-"*80
            
            # Log main header
            logging.info(f"\n{section_header}")
            logging.info("ANALYSIS RESULTS")
            logging.info(f"{section_header}")
            
            # Overall statistics - formatted in blocks
            stats_output = "Top 10 labels by overall mutual information:\n"
            for label, score in list(analysis_results['top_labels_overall'].items())[:10]:
                stats_output += f"  {label:<10} {score:.6f}\n"
            logging.info(stats_output)
            
            token_output = "Top 10 tokens by overall mutual information:\n"
            for token, score in list(analysis_results['top_tokens_overall'].items())[:10]:
                token_output += f"  {token:<15} {score:.6f}\n"
            logging.info(token_output)
            
            mi_stats_output = "Mutual information statistics:\n"
            for stat, value in analysis_results['info_stats'].items():
                mi_stats_output += f"  {stat:<20} {value:.6f}\n"
            logging.info(mi_stats_output)
            
            # Sample token-to-label results
            logging.info(f"\n{subsection_header}")
            logging.info("SAMPLE TOKEN TO TOP LABELS MAPPING")
            logging.info(f"{subsection_header}")
            
            # Get random tokens from all tokens
            all_tokens = list(analysis_results['top_tokens_overall'].keys())
            sample_tokens = random.sample(all_tokens, min(15, len(all_tokens)))
            
            for token in sample_tokens:
                token_labels_output = f"\nTop 50 labels for token '{token}':\n"
                
                # Get the top 50 labels for this token
                top_labels = analysis_results['token_to_top_labels'].get(token, {})
                
                if top_labels:
                    for i, (label, score) in enumerate(sorted(top_labels.items(), key=lambda x: x[1], reverse=True), 1):
                        token_labels_output += f"  {i:2d}. {label:<10} {score:.6f}\n"
                else:
                    token_labels_output += "  No data available for this token\n"
                logging.info(token_labels_output)
            
            # Sample label-to-token results
            logging.info(f"\n{subsection_header}")
            logging.info("SAMPLE LABEL TO TOP TOKENS MAPPING")
            logging.info(f"{subsection_header}")
            
            # Get random labels from all labels
            all_labels = list(analysis_results['top_labels_overall'].keys())
            sample_labels = random.sample(all_labels, min(15, len(all_labels)))

            for label in sample_labels:
                label_tokens_output = f"\nTop 50 tokens for label '{label}':\n"
                
                # Get the top 50 tokens for this label
                top_tokens = analysis_results['label_to_top_tokens'].get(label, {})
                
                if top_tokens:
                    for i, (token, score) in enumerate(sorted(top_tokens.items(), key=lambda x: x[1], reverse=True), 1):
                        label_tokens_output += f"  {i:2d}. {token:<15} {score:.6f}\n"
                else:
                    label_tokens_output += "  No data available for this label\n"
                logging.info(label_tokens_output)
            
            logging.info(f"{subsection_header}")
            
            # Display counts
            summary_output = f"\nComputed top labels for {len(analysis_results['token_to_top_labels'])} tokens\n"
            summary_output += f"Computed top tokens for {len(analysis_results['label_to_top_tokens'])} labels\n"
            logging.info(summary_output)

            # Preparing reports for most informative tokens for each label and vice-versa
            logging.info(f"\n{subsection_header}")
            logging.info("PREPARING REPORTS FOR LABEL TO TOP TOKENS MAPPING AND TOKENS TO TOP LABELS MAPPING")
            logging.info(f"{subsection_header}")
            # Your analysis_results with both keys
            results = generate_icd10_token_report(
                analysis_results=analysis_results,  # Contains both 'label_to_top_tokens' and 'token_to_top_labels'
                output_dir=output_dir,
                max_tokens_per_code=50,     # Top 20 tokens per ICD-10 code
                max_labels_per_token=50,    # Top 15 ICD-10 codes per token
                include_summary=True,
                sort_codes=True,
                sort_tokens=True
            )

            logging.info("Files created:")
            for report_type, path in results['files_created'].items():
                print(f"  {report_type}: {path}")

            logging.info(f"\nProcessed:")
            logging.info(f"  - {results['total_codes_processed']} ICD-10 codes")
            logging.info(f"  - {results['total_tokens_processed']} tokens")
            logging.info(f"{subsection_header}")

            # Offer to save the detailed results
            save_path = join_path_file(fname + '_analysis', Path(output_dir), ext='.pkl')
            save_output = f"\nDetailed analysis results can be saved to: {save_path}\n"
            save_output += "To save, uncomment the relevant code in the script.\n"
            logging.info(save_output)
            
            # Uncomment to save the analysis results
            # logging.info(f"Saving detailed analysis results to {save_path}...")
            # torch.save(analysis_results, save_path)
            # logging.info(f"Successfully saved detailed analysis results to {save_path}")

        logging.info("****L2R Data Loading Completed Successfully****")
        return loaded_data
        
    except Exception as e:
        logging.error(f"Error loading L2R data: {e}")
        raise


# =============================================================================
# Command-line Interface
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Load and check Learning to Rank data')
    parser.add_argument('--source_url', type=str, default="XURLs.MIMIC4_L2R",
                        help='Source URL for data')
    parser.add_argument('--data', type=str, default="mimic4_icd10_full",
                        help='Dataset name')
    parser.add_argument('--base_dir', type=str, default="../tmp",
                        help='Base directory')
    parser.add_argument('--output_dir', type=str, default="task2_output",
                        help='Output directory')
    parser.add_argument('--logfile', type=str, default="l2r_datacheck",
                        help='Log file name')
    parser.add_argument('--run_analysis', type=bool, default=True,
                        help='Whether to run analysis')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Call the main function
    main(
        output_dir=args.output_dir,
        data=args.data,
        base_dir=args.base_dir,
        source_url=args.source_url,
        logfile=args.logfile,
        run_analysis=args.run_analysis
    )