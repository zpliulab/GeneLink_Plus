***

# GeneLink+

Advancing Cell-Type-Specific Gene Regulatory Network Inference from Transcriptomics Data at Cellular Resolution

## 1.Introduction

**Genelink+**, an enhanced version of GENElink, is designed to infer **ctGRNs** from transcriptomic data at cellular or sub-cellular resolution. It utilizes **GATv2**, which replaces GAT's fixed linear transformation with a flexible, learnable parameterized transformation, enabling the capture of more complex gene relationships. To address gene node homogenization from over-smoothing, Genelink+ incorporates a residual module with skip connections that retain gene-specific information. Additionally, we improved negative sample selection during training by combining hard negative sampling with network analysis techniques and specific constraints.The figure below shows the overall framework of GeneLink+.

![Fig 1.Overview of GeneLink+ framework.](./figures/Figure 1-01.png)

The environments in which this program can run stably are for reference only:

|    Package   |    Version   |
| :----------: | :----------: |
| magic-impute |     3.0.0    |
|  matplotlib  |     3.8.4    |
|   networkx   |      3.0     |
|     numpy    |    1.23.5    |
|    pandas    |     1.5.3    |
|    scanpy    |    1.10.1    |
| scikit-learn |     1.2.1    |
|     scipy    |    1.10.1    |
|     torch    |  1.9.1+cu111 |
|  torchvision | 0.10.1+cu111 |

## 2.How to use the Demo

### 2.1 Description

The demo shows the mESC and mHSC-E scRNA-seq data with cell-type-specific network.

*   Target.csv --  The indexes of genes in expression file

*   TF.csv -- The indexes of TF genes in expression file

*   Lable.csv -- The indexes of TF and corresponding target genes in network file

*   BL--ExpressionData.csv -- The expression file

*   BL--network.csv -- The ground truth network

### 2.2 Train\_Validation\_Test

*   mESC 500 -- The mESC scRNA-seq data with TFs+500

*   mESC 1000 -- The mESC scRNA-seq data with TFs+1000

*   mHSC-E 500 -- The hESC scRNA-seq data with TFs+500

*   mHSC-E 1000 -- The hESC scRNA-seq data with TFs+1000

### 2.3 Run

In `Demo\G_res4_v2_L2.py`, the following adjustments can be made to the network model:

```python
import argparse
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 150, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='b_dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

args = parser.parse_args()
```

In `Demo\G_res4_v2_L2.py`, you can select the dataset you wish to replicate:

```python
data_type = 'mHSC-E'  # mESC or mHSC-E
num = 1000			  # 500 or 1000
net_type = 'Specific'
```

Once you have chosen the above network model parameters and dataset, run `Demo\G_res4_v2_L2.py`:

` python Demo\G_res4_v2_L2.py`

Other datasets for Benchmark can be found in the `Benchmark Dataset`, with the structure as follows:

     Benchmark Dataset
     │
     ├── Lofgof Dataset
     │   ├── mESC
     │   │   ├── TFs+500
     │   │   │   ├── BL--ExpressionData.csv
     │   │   │   ├── BL--network.csv
     │   │   │   ├── Label.csv
     │   │   │   ├── Target.csv
     │   │   │   └── TF.csv
     │   │   ├── TFs+1000
     │   │   │   ├── BL--ExpressionData.csv
     │   │   │   ├── BL--network.csv
     │   │   │   ├── Label.csv
     │   │   │   ├── Target.csv
     │   │   │   └── TF.csv
     │
     ├── Specific Dataset
     │   ├── hESC
     │   │   ├── TFs+500
     │   │   │   ├── BL--ExpressionData.csv
     │   │   │   ├── BL--network.csv
     │   │   │   ├── Label.csv
     │   │   │   ├── Target.csv
     │   │   │   └── TF.csv
     │   │   ├── TFs+1000
     │   │   │   ├── BL--ExpressionData.csv
     │   │   │   ├── BL--network.csv
     │   │   │   ├── Label.csv
     │   │   │   ├── Target.csv
     │   │   │   └── TF.csv
     │   ├── hHEP
     │   └── (Same structure as hESC)
     │   ├── mDC
     │   └── (Same structure as hESC)
     │   ├── mESC
     │   └── (Same structure as hESC)
     │   ├── mHSC-E
     │   └── (Same structure as hESC)
     │   ├── mHSC-GM
     │   └── (Same structure as hESC)
     │   ├── mHSC-L
     │   └── (Same structure as hESC)
     │
     ├── Non-Specific Dataset
     │   └── (Same structure as Specific Dataset)
     │
     ├── Specific Dataset
     │   └── (Same structure as Specific Dataset)
     │
     └──STRING Dataset
         └── (Same structure as Specific Dataset)
    70 directories, 220 files

If you wish to replicate additional datasets, we provide a detailed introduction and demonstration of the necessary datasets and the complete processing workflow in Section 3: Full Workflow for Inferring Gene Regulatory Networks with GENELink2.

## 3. Full Workflow for Inferring Gene Regulatory Networks with GENELink2

### 3.1 Dataset and Preprocessing

We applied GENELink2 for gene regulatory network inference on **benchmark datasets, scRNA-seq datasets, snRNA-seq datasets,** and **spatially resolved transcriptomics (SRT) data.** The datasets used are as follows:

Benchmark Dataset from [BEELINE](https://doi.org/10.5281/zenodo.3378975).

[The scRNA-seq dataset for PBMC8k](https://www.10xgenomics.com/resources/datasets/8-k-pbm-cs-from-a-healthy-donor-2-standard-2-1-0).

[Human AD snRNA-seq](https://www.synapse.org/#!Synapse\:syn22079621/).

Human breast cancer.

### 3.2 Choosing an Appropriate Method to Construct a Ground Truth Network

We provide two methods for constructing the **ground truth network,** stored in `code / hTFTarget_PCC.py` and `code / PCA_CMI.py`.

For cases involving a large number of genes or datasets with clear constraints on the tissue under study, we recommend using the method in `code / hTFTarget_PCC.py` to construct the ground truth network. This method utilizes the hTFTarget database, which extracts background networks with strict regulatory relationships, and preliminarily combines widely accepted reference networks with data-driven specific regulatory relationships through PCC computation.

To build a background network using hTFTarget, you need to create a new folder named `Net_hTFTarget` in the `example` directory and download [`TF-Target-information.txt`](https://guolab.wchscu.cn/hTFtarget/#!/download) to that location.

For constructing smaller-scale ground truth network, the method provided in `code / PCA_CMI.py` uses the updated version of RegNetwork to build the initial background network, extracting edges with high PCC and MI values from gene expression profiles to form cell-specific gene reference networks. Additionally, redundant regulatory interactions are removed to improve network specificity, producing a more accurate ground truth network.

The updated version of RegNetwork is included in `demo / RegNetwork`.

### 3.3 Performing Gene Regulatory Network Inference

Once the gene expression profiles have been preprocessed, and a ground truth network has been constructed using an appropriate method, the next step is to process the expression matrix and ground truth network into the standard format used by GENELink2 using `code / pre_tf_ex.py` to perform gene regulatory network inference.

It’s important to note that the dataset splitting method in `code / Train_Test_Split.py` should be chosen based on the dataset characteristics. Furthermore, the preprocessing script `code / pre_tf_ex.py` will output the current network density of the prior network, which should be saved in the correct format in `code / utils.py`. This ensures proper segmentation based on network density and facilitates research on how changes in network density affect data splitting and model training.

After model training, the results will be saved in the respective `demo / result` folder as `Channel1.csv` and `Channel2.csv`. The embeddings for the regulatory factors in the prior network are extracted from Channel1, and similarly, the embeddings for all genes (excluding self-loops if desired) are extracted from Channel2. By applying dot product and the Sigmoid function, the inferred gene regulatory network edges can be obtained using a defined threshold.&#x20;

### 3.4 Example

In the `example` section, we present a complete workflow for constructing a cell-specific gene regulatory network for B cells in the PBMCs8k dataset. In `example / Arguments.py`, we list the key parameters in the entire workflow, which can be adjusted according to your needs:

```python
import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='B', help='data type')
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--net_type', type=str, default='hTFTarget', help='Network type')

parser.add_argument('--PCA', type=bool, default=True, help='Whether the data were processed with PCA or not')
parser.add_argument('--n_components_ratio', type=float, default=0.25, help='The proportion of PCA that reduces the data to the original data')
parser.add_argument('--li_tissue', type=bool, default=True, help='Whether to restrict the organizational sources of regulatory relationships in the background network')
parser.add_argument('--tissue', type=str, default='blood', help='Tissue')
parser.add_argument('--high_threshold', type=float, default=0.98, help='The PCC threshold for adding connections.')
parser.add_argument('--low_threshold', type=float, default=0.80, help='The PCC threshold for deleting connections.')

parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
parser.add_argument('--use_distance_method', type=str, default='no', choices=['yes', 'no'], help='use train_val_test_set_with_distance if "yes"')

parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 150, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='b_dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

parser.add_argument('--density', type=float, default=75, help='According to the dot product results, those higher than this percentage were identified as edges present in the final regulatory network.')
```

By downloading the hTFTarget data as suggested in Section 3.2: Choosing an Appropriate Method to Construct a Ground Truth Network, and configuring the above parameters, you can reproduce the entire workflow by running `example / reproduce.py`:

```python
from Arguments import parser

if __name__ == '__main__':
    args = parser.parse_args()
    from pre_PBMC import pre_PBMCs
    from pre_PBMC_ex_create import hTFTarget_PCC
    from pre_tf_exPBMC import pre_data
    from Train_Test_Split import spilt
    from G_res4_v2_L2 import train
    from plote4 import con_net
    from out_st import st

    pre_PBMCs()
    hTFTarget_PCC(args)
    pre_data()
    spilt(args)
    train(args)
    con_net(args)
    st()
```

`pre_PBMCs()` will complete data preprocessing, including simple quality control, normalization, MAGIC imputation, log transformation, clustering, and displaying the marker genes for each cluster, the expression of some known marker genes, the clustering results, and marking and displaying the top four clusters by sample size based on prior knowledge. The figure below highlights part of the outputs generated during this process:

![Fig 2. Outputs from the preprocessing of PBMCs data.](./figures/PBMCs_git.png)  

`hTFTarget_PCC()` will further process the data, including optional PCA dimensionality reduction and selecting the top 4,000 most variable genes. Then, using the hTFTarget data and limiting the regulatory relationships by the tissue of origin, a background network is generated. Cell specificity is introduced by adding edges from regulatory factors in the background network with PCC values above a certain threshold, and further cell specificity is achieved by removing edges from the background network with PCC values below a certain threshold.

`pre_data()` will format the gene expression data and ground truth network data into a standard format.

`spilt(args)` will split the data into training, validation, and test sets, with the flexibility to adjust the proportion of the training set. When the network density is low, you can try adding distance constraints to the selection of negative samples.

`train(args)` will train the model according to the set parameters, resulting in gene embeddings.

`con_net(args)` will construct the final gene regulatory network based on the regulatory factors in the ground truth network and the results from the model training. You can further view and analyze this network in Cytoscape.

`st()` will generate statistics for the gene regulatory network, including measures such as In-Degree Centrality, Out-Degree Centrality, Closeness Centrality, and Clustering Coefficient.

## Citation

Please see citation widget on the sidebar.
