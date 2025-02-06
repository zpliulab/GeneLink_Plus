import pandas as pd
from scipy.io import mmread
import os
import scanpy as sc
import magic
import numpy as np
import matplotlib.pyplot as plt
import random
import tarfile

def pre_PBMCs():
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # Adjust global font size as needed

    # Set random seed to ensure reproducibility
    np.random.seed(12)
    random.seed(12)

    # Check if the tar.gz file is already extracted; if not, extract it
    tar_file_path = 'pbmc8k_filtered_gene_bc_matrices.tar.gz'
    extracted_dir = 'pbmc8k_filtered_gene_bc_matrices'

    if not os.path.exists(extracted_dir):
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall()
        print(f"Extracted {tar_file_path}.")

    # Load gene and cell identifiers
    basic_dir = 'filtered_gene_bc_matrices\\GRCh38'
    genes_dir = os.path.join(basic_dir, 'genes.tsv')
    barcodes_dir = os.path.join(basic_dir, 'barcodes.tsv')

    # Load gene names, ensuring string type for the index
    genes = pd.read_csv(genes_dir, header=None, sep='\t', usecols=[1]).squeeze("columns")
    if not genes.dtype == 'object':
        genes = genes.astype('str')

    # Load cell barcodes, using .squeeze("columns") to replace deprecated squeeze=True
    barcodes = pd.read_csv(barcodes_dir, header=None, sep='\t').squeeze("columns")

    # Load the expression matrix
    matrix_dir = os.path.join(basic_dir, 'matrix.mtx')
    matrix = mmread(matrix_dir)

    # Convert to dense format and transpose so that each row corresponds to a gene and each column to a cell
    matrix_dense = matrix.todense()

    # Create a DataFrame using gene names as row labels
    gene_expression = pd.DataFrame(matrix_dense, index=genes, columns=barcodes)
    gene_expression.index.name = None  # Ensure the index has no name

    # Save the DataFrame to a CSV file for inspection
    output_path = os.path.join(basic_dir, 'out.csv')
    gene_expression.to_csv(output_path)

    print(f"Output saved to {output_path}")

    # Show a small part of the DataFrame to avoid overwhelming output
    print(gene_expression)

    # Create an AnnData object
    adata = sc.AnnData(gene_expression.T)

    # Ensure the index names are appropriate strings or None
    adata.obs.index.name = None
    adata.var.index.name = None

    # Ensure variable names (gene names) are unique
    adata.var_names_make_unique()

    # Gene filtering condition: ensure each gene is expressed in at least 5% of cells
    sc.pp.filter_genes(adata, min_cells=int(adata.n_obs * 0.05))

    # Cell filtering: ensure each cell expresses at least a certain number of genes
    sc.pp.filter_cells(adata, min_genes=200)  # For example, at least 200 genes are expressed

    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)

    # MAGIC imputation
    magic_operator = magic.MAGIC()
    adata = magic_operator.fit_transform(adata)  # Assuming adata is your AnnData object

    # Logarithmic transformation
    sc.pp.log1p(adata)

    # Create a DataFrame to store the imputed gene expression data
    gene_expression_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    gene_expression_df = gene_expression_df.round(3).T
    # Show a small part of the DataFrame to avoid overwhelming output
    print(gene_expression_df)

    # Set output file path
    output_path = os.path.join(basic_dir, 'out2.csv')

    # Save the DataFrame to a CSV file
    gene_expression_df.to_csv(output_path)

    print(f"Data after MAGIC imputation saved to {output_path}")

    # Principal component analysis
    sc.tl.pca(adata, svd_solver='arpack')

    # Compute the neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

    # Run tSNE for visualization
    sc.tl.tsne(adata)

    # Run Leiden algorithm for clustering
    sc.tl.leiden(adata, resolution=0.1)

    # Identify marker genes
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', n_genes=30)
    sc.pl.rank_genes_groups(adata, n_genes=30, sharey=False)  # View the top 30 marker genes for each cluster

    # Visualize tSNE, coloring by marker genes
    genes_to_plot = ['CD79A', 'MS4A1', 'RUNX3', 'IL32', 'CCR7', 'MAL', 'CD14', 'CD36']

    for gene in genes_to_plot:
        sc.pl.tsne(adata, color=gene, show=True)

    # Visualize clustering results
    sc.pl.tsne(adata, color=['leiden'], show=True)

    # Map Leiden clusters to known cell types
    cell_type_mapping = {'0': 'CD8', '1': 'CD4', '2': 'CD14 Monocytes', '3': 'B'}
    adata.obs['cell_type'] = adata.obs['leiden'].map(cell_type_mapping)

    for cell_type in adata.obs['cell_type'].unique():
        sub_data = adata[adata.obs['cell_type'] == cell_type, :]
        sub_data_df = pd.DataFrame(sub_data.X, index=sub_data.obs_names, columns=sub_data.var_names)
        sub_data_df = sub_data_df.round(3).T
        sub_data_df.to_csv(os.path.join(basic_dir, f'{cell_type}_out.csv'))
        print(f"Data for {cell_type} saved to {cell_type}_out.csv")

    sc.tl.rank_genes_groups(adata, 'cell_type', method='t-test_overestim_var', n_genes=30)
    sc.pl.rank_genes_groups(adata, n_genes=30, xlabel='Adjusted t score', sharey=False)
    # Visualize tSNE, coloring by cell type
    sc.pl.tsne(adata, color='cell_type', title='tSNE by Cell Type')
