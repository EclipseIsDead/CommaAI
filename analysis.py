#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset, DatasetDict
from collections import Counter, defaultdict
import seaborn as sns
from tqdm import tqdm
import multiprocessing
from scipy import stats
import pandas as pd

def analyze_token_file(example):
    """Load and analyze a single token file"""
    path = Path(example['path'])
    tokens = np.load(path)

    results = {
        'filename': path.name,
        'shape': tokens.shape,
        'unique_count': len(np.unique(tokens)),
        'min_token': int(np.min(tokens)),
        'max_token': int(np.max(tokens)),
        'mean': float(np.mean(tokens)),
        'std': float(np.std(tokens)),
    }

    token_counts = np.bincount(tokens.flatten().astype(np.int32) - results['min_token'])
    results['entropy'] = float(stats.entropy(token_counts))

    # Temporal analysis (differences between adjacent frames)
    if tokens.shape[0] > 1:
        diffs = tokens[1:] - tokens[:-1]
        results['mean_diff'] = float(np.mean(np.abs(diffs)))
        results['zero_diff_percent'] = float(np.mean(diffs == 0) * 100)

        # Analyze runs of identical tokens
        flat_tokens = tokens.reshape(-1)
        runs = np.diff(np.concatenate(([0], np.where(np.diff(flat_tokens) != 0)[0] + 1, [len(flat_tokens)])))
        results['max_run_length'] = int(np.max(runs))
        results['mean_run_length'] = float(np.mean(runs))

    # Get token frequency distribution (for first 1000 tokens to keep it manageable)
    sample = tokens.flatten()[:1000]
    results['token_counts'] = Counter(sample.tolist())

    # Analyze patterns in positions
    position_stats = {}
    for i in range(min(8, tokens.shape[1])):
        for j in range(min(16, tokens.shape[2])):
            pos_tokens = tokens[:, i, j]
            position_stats[f"{i},{j}"] = {
                'mean': float(np.mean(pos_tokens)),
                'std': float(np.std(pos_tokens)),
                'unique': int(len(np.unique(pos_tokens))),
            }
    results['position_stats'] = position_stats

    return results

def create_distribution_plots(all_results, output_dir):
    """Create visualizations of token distributions"""
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate all token counts
    all_tokens = Counter()
    for result in all_results:
        all_tokens.update(result['token_counts'])

    # Plot overall token distribution (top 100)
    plt.figure(figsize=(15, 8))
    tokens, counts = zip(*all_tokens.most_common(100))
    plt.bar(range(len(tokens)), counts)
    plt.title('Top 100 Token Frequency Distribution')
    plt.xlabel('Token Index (Sorted by Frequency)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/token_distribution.png")
    plt.close()

    # Plot run length distribution
    run_lengths = [res['mean_run_length'] for res in all_results if 'mean_run_length' in res]
    if run_lengths:
        plt.figure(figsize=(12, 6))
        plt.hist(run_lengths, bins=30)
        plt.title('Distribution of Mean Run Lengths')
        plt.xlabel('Mean Run Length')
        plt.ylabel('Count')
        plt.savefig(f"{output_dir}/run_length_distribution.png")
        plt.close()

    # Plot positional token entropy heatmap
    pos_entropy = defaultdict(list)
    for result in all_results:
        for pos, stats in result.get('position_stats', {}).items():
            pos_entropy[pos].append(stats['unique'])

    if pos_entropy:
        entropy_matrix = np.zeros((8, 16))
        for pos, values in pos_entropy.items():
            i, j = map(int, pos.split(','))
            entropy_matrix[i, j] = np.mean(values)

        plt.figure(figsize=(16, 8))
        sns.heatmap(entropy_matrix, annot=False, cmap='viridis')
        plt.title('Average Unique Token Count by Position')
        plt.savefig(f"{output_dir}/position_entropy.png")
        plt.close()

    # Plot temporal difference stats
    if all('mean_diff' in res for res in all_results):
        diff_means = [res['mean_diff'] for res in all_results]
        zero_diff_pcts = [res['zero_diff_percent'] for res in all_results]

        plt.figure(figsize=(12, 6))
        plt.hist(diff_means, bins=30)
        plt.title('Distribution of Mean Token Differences Between Frames')
        plt.xlabel('Mean Absolute Difference')
        plt.ylabel('Count')
        plt.savefig(f"{output_dir}/temporal_diff_distribution.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.hist(zero_diff_pcts, bins=30)
        plt.title('Distribution of Zero-Difference Percentages')
        plt.xlabel('Percentage of Unchanged Tokens Between Frames')
        plt.ylabel('Count')
        plt.savefig(f"{output_dir}/zero_diff_distribution.png")
        plt.close()

def analyze_token_correlations(all_results, output_dir):
    """Analyze correlations between tokens in different positions"""
    # Find files with small enough sizes for memory
    valid_results = [r for r in all_results if r.get('shape', (0,))[0] < 1000]
    if not valid_results:
        return

    sample_files = valid_results[:min(5, len(valid_results))]

    for idx, result in enumerate(sample_files):
        path = Path(result['filename'])
        tokens = np.load(path)

        # Sample correlation between positions
        positions = [(0,0), (0,8), (4,0), (4,8), (7,15)]
        pos_data = {}

        for i, j in positions:
            if i < tokens.shape[1] and j < tokens.shape[2]:
                pos_data[f"pos_{i}_{j}"] = tokens[:100, i, j]

        if len(pos_data) > 1:
            df = pd.DataFrame(pos_data)
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.title(f'Token Correlation Between Positions (File {idx+1})')
            plt.savefig(f"{output_dir}/position_correlation_{idx}.png")
            plt.close()

def analyze_delta_compression_potential(all_results, output_dir):
    """Analyze the potential for delta compression"""
    zero_diff_pcts = [res.get('zero_diff_percent', 0) for res in all_results if 'zero_diff_percent' in res]

    if not zero_diff_pcts:
        return

    # Create a report on delta compression potential
    with open(f"{output_dir}/delta_compression_report.txt", 'w') as f:
        f.write("Delta Compression Potential Analysis\n")
        f.write("===================================\n\n")
        f.write(f"Files analyzed: {len(zero_diff_pcts)}\n")
        f.write(f"Mean unchanged tokens between frames: {np.mean(zero_diff_pcts):.2f}%\n")
        f.write(f"Median unchanged tokens between frames: {np.median(zero_diff_pcts):.2f}%\n")
        f.write(f"Min unchanged tokens between frames: {np.min(zero_diff_pcts):.2f}%\n")
        f.write(f"Max unchanged tokens between frames: {np.max(zero_diff_pcts):.2f}%\n\n")

        if np.mean(zero_diff_pcts) > 50:
            f.write("RECOMMENDATION: High potential for delta compression.\n")
            f.write("Consider encoding only differences between frames.\n")
        elif np.mean(zero_diff_pcts) > 30:
            f.write("RECOMMENDATION: Moderate potential for delta compression.\n")
            f.write("Consider a hybrid approach with selective delta encoding.\n")
        else:
            f.write("RECOMMENDATION: Low potential for delta compression.\n")
            f.write("Consider alternative approaches like context modeling.\n")

def generate_summary_report(all_results, output_dir):
    """Generate a comprehensive summary report"""
    with open(f"{output_dir}/token_analysis_summary.txt", 'w') as f:
        f.write("Token Analysis Summary Report\n")
        f.write("===========================\n\n")

        # Basic stats
        shapes = [tuple(res['shape']) for res in all_results]
        common_shape = max(set(shapes), key=shapes.count)

        f.write(f"Files analyzed: {len(all_results)}\n")
        f.write(f"Most common shape: {common_shape}\n")
        f.write(f"Min token value: {min(res['min_token'] for res in all_results)}\n")
        f.write(f"Max token value: {max(res['max_token'] for res in all_results)}\n\n")

        # Token stats
        avg_unique = np.mean([res['unique_count'] for res in all_results])
        avg_entropy = np.mean([res['entropy'] for res in all_results])
        f.write(f"Average unique tokens per file: {avg_unique:.2f}\n")
        f.write(f"Average entropy: {avg_entropy:.2f} bits\n")

        if 'mean_run_length' in all_results[0]:
            avg_run = np.mean([res['mean_run_length'] for res in all_results])
            f.write(f"Average run length: {avg_run:.2f} tokens\n\n")

        # Compression suggestions
        f.write("Compression Strategy Suggestions:\n")
        f.write("-------------------------------\n")

        if avg_unique < 1000:
            f.write("1. Consider dictionary-based compression as token vocabulary is limited.\n")
        else:
            f.write("1. Large token vocabulary detected, consider context modeling approaches.\n")

        if 'zero_diff_percent' in all_results[0] and np.mean([res['zero_diff_percent'] for res in all_results]) > 40:
            f.write("2. High temporal redundancy detected, delta encoding recommended.\n")
        else:
            f.write("2. Moderate temporal redundancy, consider hybrid encoding schemes.\n")

        if 'mean_run_length' in all_results[0] and np.mean([res['mean_run_length'] for res in all_results]) > 5:
            f.write("3. Long run lengths detected, run-length encoding should be effective.\n")

        # Positional patterns
        f.write("\nPositional Analysis:\n")
        f.write("------------------\n")
        f.write("See position_entropy.png for positional uniqueness heatmap\n")

if __name__ == '__main__':
    output_dir = Path('./token_analysis_results')
    os.makedirs(output_dir, exist_ok=True)
    num_proc = max(1, multiprocessing.cpu_count() - 1)

    print("Loading commaVQ dataset (splits 0 and 1)...")
    # Load split 0 and 1 (limiting to first 100 examples for faster analysis)
    splits = ['0', '1']
    data_files = {'0': 'data_0_to_2500.zip', '1': 'data_2500_to_5000.zip'}
    ds = load_dataset('commaai/commavq', num_proc=num_proc, split=splits, data_files=data_files)
    ds = DatasetDict(zip(splits, ds))

    # Limit analysis to a sample for faster processing
    SAMPLE_SIZE = 200  # Adjust this number based on your computational resources
    for split in splits:
        if len(ds[split]) > SAMPLE_SIZE:
            ds[split] = ds[split].select(range(SAMPLE_SIZE))

    print(f"Analyzing token distributions for {sum(len(ds[split]) for split in splits)} files...")

    # Process in batches to avoid memory issues
    all_results = []

    for split in splits:
        for i in tqdm(range(0, len(ds[split]), num_proc), desc=f"Processing split {split}"):
            batch = ds[split].select(range(i, min(i + num_proc, len(ds[split]))))
            results = [analyze_token_file(example) for example in batch]
            all_results.extend(results)

    print("Creating visualizations and reports...")
    create_distribution_plots(all_results, output_dir)
    analyze_token_correlations(all_results, output_dir)
    analyze_delta_compression_potential(all_results, output_dir)
    generate_summary_report(all_results, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")
    print("Examine the reports and visualizations to inform your compression strategy.")
