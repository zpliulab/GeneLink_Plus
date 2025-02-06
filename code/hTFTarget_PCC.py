import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import time
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

def load_and_filter_network(TFT_path, gene_list):
    # Load and filter regulatory network
    network_df = pd.read_csv(TFT_path, sep='\t', header=None, names=['TF', 'Target', 'Source'])
    # Ensure source contains only "XXX" or at least includes "XXX"
    tissue_network = network_df[network_df['Source'].str.contains('XXX')]
    valid_network = tissue_network[(tissue_network['TF'].isin(gene_list)) & (tissue_network['Target'].isin(gene_list))]
    # valid_network = network_df[(network_df['TF'].isin(gene_list)) & (network_df['Target'].isin(gene_list))]

    # Calculate network density
    possible_links = len(set(valid_network['TF'])) * len(gene_list)
    print(len(set(valid_network['TF'])))
    print(len(gene_list))
    print(possible_links)
    network_density = len(valid_network) / possible_links
    print(f"Network density: {network_density:.6f}")

    return valid_network[['TF', 'Target']]

def add_high_correlation_links(expression_df, background_network, TFT_path ,high_threshold=0.85, low_threshold=0.50):
    '''
    htftarget_df = pd.read_csv(TFT_path, sep='\t', header=None, names=['TF', 'Target', 'Source'])
    tfs_in_htftarget = set(htftarget_df['TF'])
    tfs = tfs_in_htftarget.intersection(set(expression_df.index))
    '''
    # Calculate Pearson correlation among genes once
    correlations = expression_df.T.corr()
    np.fill_diagonal(correlations.values, 0)  # Set diagonal to 0 to remove self-loops

    # Extract TFs from the background network
    tfs = set(background_network['TF'])
    # print(tfs)
    # print(len(tfs))

    # Initialize sets to store high and low correlation links
    high_corr_links_set = set()
    low_corr_links_set = set()

    # Process each TF and its potential targets
    for tf in tfs:
        tf_correlations = correlations.loc[tf].dropna()  # Drop NaN to avoid self-comparison issues
        for target, correlation in tf_correlations.items():
            # print(abs(correlation))
            # Check if correlation is above high threshold
            if abs(correlation) >= high_threshold:
                high_corr_links_set.add((tf, target))
            # Check if correlation is below low threshold
            elif abs(correlation) <= low_threshold:
                low_corr_links_set.add((tf, target))


    # Remove low correlation links from the background network
    background_network_set = set(map(tuple, background_network[['TF', 'Target']].values))
    background_network_set -= low_corr_links_set

    # Combine high correlation links with background network
    final_network_set = background_network_set.union(high_corr_links_set)

    # Convert set back to DataFrame
    final_network = pd.DataFrame(list(final_network_set), columns=['TF', 'Target'])

    # Report changes
    print(f"Original number of relationships in the background network: {len(background_network)}")
    print(f"Number of low correlation links removed: {len(background_network)-len(background_network_set)}")
    print(f"Number of new high correlation links added: {len(final_network)-len(background_network_set)}")
    print(f"Final number of relationships in the network: {len(final_network)}")

    # Calculate network density
    possible_links = len(set(final_network['TF'])) * len(expression_df.index)
    print(len(set(final_network['TF'])))
    print(len(expression_df.index))
    print(possible_links)
    network_density = len(final_network) / possible_links
    print(f"Network density: {network_density:.6f}")

    return final_network


def process_all_datasets():
    TFT_path ='dataset_o\TF-Target-information.txt'
    data_dir = ' '
    save_dir = ' '

    expression_file = os.path.join(data_dir, f' ')
    save_file = os.path.join(save_dir, f' ')

    print(f'Processing {expression_file}...')

    expression_df = pd.read_csv(expression_file, index_col=0)

    start_time = time.time()
    background_network = load_and_filter_network(TFT_path, expression_df.index.tolist())
    print(f"Background network construction is time-consuming {time.time() - start_time:.2f} 秒.")

    start_time = time.time()
    final_network = add_high_correlation_links(expression_df, background_network, TFT_path)
    print(f"Adding high links takes time {time.time() - start_time:.2f} 秒.")

    final_network.to_csv(save_file, index=False)
    print(f"Files saved successfully for {expression_file}.\n")


if __name__ == '__main__':
    process_all_datasets()
