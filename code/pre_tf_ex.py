import pandas as pd
import numpy as np
import os

base_dir = ' '
save_dir = ' '
TF_Target_information_dir = os.path.join(base_dir, ' ')
expression_matrix_path = os.path.join(base_dir, ' ')

# Load TF-Target information and expression data
df_filtered = pd.read_csv(TF_Target_information_dir)
expression_df = pd.read_csv(expression_matrix_path, index_col=0)

# Ensure that both TF and Target genes are in the expression matrix
genes_in_expression_df = set(expression_df.index)
df_filtered = df_filtered[df_filtered['TF'].isin(genes_in_expression_df) & df_filtered['Target'].isin(genes_in_expression_df)].copy()

# Create a DataFrame for all genes in the expression matrix with an index
all_genes_df = pd.DataFrame(list(genes_in_expression_df), columns=['Gene'])
all_genes_df['index'] = range(len(all_genes_df))

# Apply mappings for TF and Target indices
df_filtered['TF_index'] = df_filtered['TF'].apply(lambda x: all_genes_df.loc[all_genes_df['Gene'] == x, 'index'].values[0])
df_filtered['Target_index'] = df_filtered['Target'].apply(lambda x: all_genes_df.loc[all_genes_df['Gene'] == x, 'index'].values[0])

# Create and save label_df
label_df = df_filtered[['TF_index', 'Target_index']].drop_duplicates().reset_index(drop=True)
label_df.rename(columns={'TF_index': 'TF', 'Target_index': 'Target'}, inplace=True)

# Save files
all_genes_df.to_csv(os.path.join(save_dir, 'Target.csv'), index=True)
tf_df = all_genes_df.loc[all_genes_df['Gene'].isin(df_filtered['TF'].unique())].reset_index(drop=True)
tf_df.to_csv(os.path.join(save_dir, 'TF.csv'), index=True)
label_df.to_csv(os.path.join(save_dir, 'Label.csv'), index=True)
expression_df.to_csv(os.path.join(save_dir, 'BL--ExpressionData.csv'), index=True)

print("Files saved: Target.csv, TF.csv, Label.csv, BL--ExpressionData.csv")

network_df = df_filtered[['TF', 'Target']].reset_index(drop=True)
network_df.to_csv(os.path.join(save_dir, 'BL--Network.csv'), index=True)

print("Additional file saved: BL--Network.csv")

# Calculate network density
possible_links = len(set(label_df['TF'])) * len(all_genes_df)
network_density = len(label_df) / possible_links
print(len(set(label_df['TF'])))
print(len(all_genes_df))
print(f"Network density: {network_density:.6f}")
