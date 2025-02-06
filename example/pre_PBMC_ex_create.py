import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import time
import scanpy as sc
from Arguments import parser

def filter_genes_by_variance(expression_df, n_top_genes=4000, tf_file_path=None):
    # Load TFs from the Excel file
    if tf_file_path:
        tf_genes = pd.read_excel(tf_file_path, header=None).iloc[:, 0].tolist()
        print(f"Number of TFs loaded: {len(tf_genes)}")
    else:
        tf_genes = []

    # Convert DataFrame to AnnData
    adata = sc.AnnData(expression_df.T)
    print(f"Initial gene count: {adata.shape[1]}, Initial cell count: {adata.shape[0]}")

    # Calculate highly variable genes using scanpy
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    print(f"Number of highly variable genes identified: {sum(adata.var.highly_variable)}")

    # Filter the expression dataframe based on highly variable genes
    highly_variable_genes = adata.var[adata.var.highly_variable].index.tolist()

    # Ensure TFs are not filtered out
    genes_to_keep = set(highly_variable_genes).union(set(tf_genes))
    filtered_expression_df = expression_df.loc[expression_df.index.intersection(genes_to_keep)]

    print(f"Original gene count: {expression_df.shape[0]}, Filtered gene count: {filtered_expression_df.shape[0]}")

    return filtered_expression_df


def perform_pca(expression_df, n_components_ratio=0.25):
    # Reduce dimensions with PCA along genes (rows)
    pca = PCA(n_components=int(expression_df.shape[1] * n_components_ratio))  # Reduce to ~1/5 of original dimensions
    pca_result = pca.fit_transform(expression_df)
    return pd.DataFrame(pca_result, index=expression_df.index)


def load_and_filter_network(TFT_path, gene_list, args):
    # Load and filter regulatory network
    network_df = pd.read_csv(TFT_path, sep='\t', header=None, names=['TF', 'Target', 'Source'])
    # Ensure source contains only "blood" or at least includes "blood"
    blood_network = network_df[network_df['Source'].str.contains(args.tissue)]
    if args.li_tissue:
        valid_network = blood_network[(blood_network['TF'].isin(gene_list)) & (blood_network['Target'].isin(gene_list))]
    else:
        valid_network = network_df[(network_df['TF'].isin(gene_list)) & (network_df['Target'].isin(gene_list))]

    # Calculate network density
    possible_links = len(set(valid_network['TF'])) * len(gene_list)
    network_density = len(valid_network) / possible_links
    print(f"Network density: {network_density:.6f}")
    # print(len(set(valid_network['TF'])))
    # print(len(gene_list))
    # print(possible_links)
    # print(len(set(valid_network['TF'])))

    return valid_network[['TF', 'Target']]

def add_high_correlation_links(expression_df, background_network, high_threshold=0.98, low_threshold=0.80):
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

    # Print high and low correlation links
    # print("High correlation links set:")
    # print(high_corr_links_set)
    # print("Low correlation links set:")
    # print(low_corr_links_set)

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

def hTFTarget_PCC(args):
    expression_path = 'filtered_gene_bc_matrices\GRCh38\B_out.csv'
    TFT_path = 'Net_hTFTarget\TF-Target-information.txt'
    save_dir = 'dataset_PBMC'

    start_time = time.time()
    expression_df = pd.read_csv(expression_path, index_col=0)
    filtered_expression = filter_genes_by_variance(expression_df)

    if args.PCA:
        filtered_expression = perform_pca(filtered_expression,args.n_components_ratio)
        print(f"Time consuming of PCA processing: {time.time() - start_time:.2f} s.")

    start_time = time.time()
    background_network = load_and_filter_network(TFT_path, filtered_expression.index.tolist(), args)
    print(background_network)
    print(f"Time consuming of background network construction: {time.time() - start_time:.2f} s.")
    start_time = time.time()
    final_network = add_high_correlation_links(filtered_expression, background_network, args.high_threshold, args.low_threshold)
    print(f"Time consuming of add high connections: {time.time() - start_time:.2f} s.")

    pca_expression = filtered_expression.round(3)
    pca_expression.to_csv(os.path.join(save_dir, 'B_ex.csv'))
    final_network.to_csv(os.path.join(save_dir, 'B_GRN.csv'), index=False)
    print("Files saved successfully.")

if __name__ == '__main__':
    args = parser.parse_args()
    hTFTarget_PCC(args)

