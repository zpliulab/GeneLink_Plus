
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import numpy as np
import networkx as nx
import itertools
import math
import copy
import os
from collections import defaultdict


def pca_cmi(data, new_net_bit, theta, max_order, show=False):
    predicted_graph = nx.DiGraph()
    predicted_graph.add_nodes_from(data.index.to_list())
    for _, row in new_net_bit.iterrows():
        TF = row['TF']
        Gene = row['Gene']
        predicted_graph.add_edge(TF, Gene)
    num_edges = predicted_graph.number_of_edges()
    L = 0
    nochange = False
    data = data.T
    while L < max_order and nochange == False:
        L = L + 1
        predicted_graph, nochange = remove_edges(predicted_graph, data, L, theta)
    if show:
        print("Final Prediction:")
        print("-----------------")
        print("Order : {}".format(L))
        print("Number of edges in the predicted graph : {}".format(predicted_graph.number_of_edges()))
    predicted_adjMatrix = nx.adjacency_matrix(predicted_graph)
    return predicted_adjMatrix, predicted_graph


def remove_edges(predicted_graph, data, L, theta):
    initial_num_edges = predicted_graph.number_of_edges()
    edges = predicted_graph.edges()

    for edge in list(edges):
        neighbors1 = set(predicted_graph.neighbors(edge[0]))
        neighbors2 = set(predicted_graph.neighbors(edge[1]))
        neighbors = neighbors1.intersection(neighbors2)
        nhbrs = copy.deepcopy(sorted(neighbors))
        T = len(nhbrs)
        if (T < L and L != 0) or edge[0] == edge[1]:
            continue
        else:
            x = data[edge[0]].to_numpy()
            if x.ndim == 1:
                x = np.reshape(x, (-1, 1))
            y = data[edge[1]].to_numpy()
            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            K = list(itertools.combinations(nhbrs, L))
            if L == 0:
                cmiVal = conditional_mutual_info(x.T, y.T)

                if cmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
            else:
                maxCmiVal = 0
                for zgroup in K:
                    XYZunique = len(np.unique(list([edge[0], edge[1], zgroup[0]])))
                    if XYZunique < 3:
                        continue
                    else:
                        z = data[list(zgroup)].to_numpy()
                        if z.ndim == 1:
                            z = np.reshape(z, (-1, 1))
                        cmiVal = conditional_mutual_info(x.T, y.T, z.T)
                    if cmiVal > maxCmiVal:
                        maxCmiVal = cmiVal
                if maxCmiVal < theta:
                    predicted_graph.remove_edge(edge[0], edge[1])
    final_num_edges = predicted_graph.number_of_edges()
    if final_num_edges < initial_num_edges:
        return predicted_graph, False
    return predicted_graph, True



def conditional_mutual_info(X, Y, Z=np.array(1)):
    if X.ndim == 1:
        X = np.reshape(X, (-1, 1))
    if Y.ndim == 1:
        Y = np.reshape(Y, (-1, 1))
    if Z.ndim == 0:
        c1 = np.cov(X)
        if c1.ndim != 0:
            d1 = np.linalg.det(c1)
        else:
            d1 = c1.item()
        c2 = np.cov(Y)
        if c2.ndim != 0:
            d2 = np.linalg.det(c2)
        else:
            d2 = c2.item()
        c3 = np.cov(X, Y)
        if c3.ndim != 0:
            d3 = np.linalg.det(c3)
        else:
            d3 = c3.item()
        cmi = (1 / 2) * np.log((d1 * d2) / d3)
    else:
        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))

        c1 = np.cov(np.concatenate((X, Z), axis=0))
        if c1.ndim != 0:
            d1 = np.linalg.det(c1)
        else:
            d1 = c1.item()
        c2 = np.cov(np.concatenate((Y, Z), axis=0))
        if c2.ndim != 0:
            d2 = np.linalg.det(c2)
        else:
            d2 = c2.item()
        c3 = np.cov(Z)
        if c3.ndim != 0:
            d3 = np.linalg.det(c3)
        else:
            d3 = c3.item()
        c4 = np.cov(np.concatenate((X, Y, Z), axis=0))
        if c4.ndim != 0:
            d4 = np.linalg.det(c4)
        else:
            d4 = c4.item()
        cmi = (1 / 2) * np.log((d1 * d2) / (d3 * d4))
    if math.isinf(cmi):
        cmi = 0
    return cmi

#####################   上面是PCA-CMI算法


def cal_del_TF_edge(GeneName):
    TF_list = pd.read_csv('TF.txt', sep='\t')
    GeneName = pd.Series(GeneName)
    TF_list = TF_list[TF_list['Symbol'].isin(GeneName)]['Symbol']
    TF_positions = GeneName[GeneName.isin(TF_list)]
    original_list = list(range(GeneName.shape[0]))
    GENE_ID_list = [x for x in original_list if x not in TF_positions.index]
    TF_ID_list = list(TF_positions.index)
    return GENE_ID_list, TF_ID_list



# calculate each type percent of edges in GRN
def cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig):
    new_bit_crop = new_bit_crop['TF'] + '-' + new_bit_crop['Gene']
    if new_bit_crop.shape[0] == 0:
        return 0, 0, 0
    net_bit_origC = net_bit_orig['TF'] + '-' + net_bit_orig['Gene']
    net_bit_origC = pd.Series(list(set(new_bit_crop) & set(net_bit_origC)))
    NUM_ORIG = net_bit_origC.shape[0] / new_bit_crop.shape[0] * 100

    if len(corr_TF_Gene) > 0:
        corr_TF_GeneC = corr_TF_Gene['TF'] + '-' + corr_TF_Gene['Gene']
        corr_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(corr_TF_GeneC)))
        count_PCC = (~corr_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_PCC = count_PCC / new_bit_crop.shape[0] * 100
    else:
        NUM_PCC = 0

    if len(MI_TF_Gene) > 0:
        MI_TF_GeneC = MI_TF_Gene['TF'] + '-' + MI_TF_Gene['Gene']
        MI_TF_GeneC = pd.Series(list(set(new_bit_crop) & set(MI_TF_GeneC)))
        count_MI = (~MI_TF_GeneC.isin(net_bit_origC)).sum()
        NUM_MI = count_MI / new_bit_crop.shape[0] * 100
    else:
        NUM_MI = 0

    if (NUM_ORIG + NUM_PCC + NUM_MI) != 100:
        SUM1 = (NUM_PCC + NUM_MI)
        NUM_PCC = NUM_PCC * (100 - NUM_ORIG) / SUM1
        NUM_MI = NUM_MI * (100 - NUM_ORIG) / SUM1

    if NUM_PCC + NUM_MI > 50:
        overflow = True
    else:
        overflow = False
    return NUM_ORIG, NUM_PCC, NUM_MI, overflow


def compute_mutual_information(df):
    num_rows = df.shape[0]
    mi_matrix = pd.DataFrame(index=df.index, columns=df.index)

    for i in range(num_rows):
        for j in range(i, num_rows):
            feature1 = df.iloc[i, :].values
            feature2 = df.iloc[j, :].values
            mi = mutual_info_score(feature1, feature2)
            mi_matrix.iloc[i, j] = mi
            mi_matrix.iloc[j, i] = mi

    return mi_matrix


def compare_char(charlist, setlist):
    try:
        index = setlist.index(charlist)
    except ValueError:
        index = None
    return index


def calRegnetwork(human_network, GRN_GENE_symbol):
    human_network_TF_symbol = human_network.iloc[:, 0].values
    human_network_Gene_symbol = human_network.iloc[:, 2].values
    d = 1
    network = []

    for i in range(len(GRN_GENE_symbol)):
        number = [j for j, x in enumerate(human_network_TF_symbol) if str(GRN_GENE_symbol[i]) == x]
        if len(number) > 0:
            for z in range(len(number)):
                networkn = []
                number2 = compare_char(str(human_network_Gene_symbol[number[z]]), GRN_GENE_symbol)
                if number2 is not None:
                    networkn.append(GRN_GENE_symbol[i])  # 调控基因
                    networkn.append(GRN_GENE_symbol[number2])  # 靶基因
                    network.append(networkn)
                    d += 1
    return pd.DataFrame(network, columns=['TF', 'Gene'])

# add high MI
def high_MI(exp_pca_discretized, exp_pca, net_bit, parm):
    row_MI = compute_mutual_information(exp_pca_discretized)
    np.fill_diagonal(row_MI.to_numpy(), 0)
    MI_thrd = 1
    rflag = 1
    while rflag == 1:
        indices = np.where(row_MI > MI_thrd)
        if parm['MI_percent'] * len(indices[0]) > net_bit.shape[0]:
            MI_thrd = MI_thrd + 0.1
            rflag = 1
        else:
            MI_TF = exp_pca.index[indices[0]]
            MI_Gene = exp_pca.index[indices[1]]
            MI_TF_Gene = pd.DataFrame([MI_TF, MI_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, MI_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, MI_TF_Gene


# add high Pearson corrlation
def high_pearson(exp_pca, net_bit, parm):
    row_corr = exp_pca.T.corr(method='pearson')
    np.fill_diagonal(row_corr.to_numpy(), 0)
    pearson_thrd = 0.95
    rflag = 1
    while rflag == 1:
        indices = np.where(row_corr > pearson_thrd)
        if parm['pear_percent'] * len(indices[0]) > net_bit.shape[0]:
            pearson_thrd = pearson_thrd + 0.0005
            rflag = 1
        else:
            corr_TF = exp_pca.index[indices[0]]
            corr_Gene = exp_pca.index[indices[1]]
            corr_TF_Gene = pd.DataFrame([corr_TF, corr_Gene], index=['TF', 'Gene']).T
            net_bit = pd.concat([net_bit, corr_TF_Gene], axis=0).reset_index(drop=True)
            net_bit = net_bit.drop_duplicates()
            rflag = 0
    return net_bit, corr_TF_Gene




def from_cancer_create(BRCA_exp_filter_saver, parm, human_network=None,drop=True):
    exp = BRCA_exp_filter_saver
    scaler = StandardScaler()
    normal_exp = scaler.fit_transform(exp)
    exp = pd.DataFrame(normal_exp, columns=exp.columns, index=exp.index)
    net_bit = calRegnetwork(human_network, exp.index.to_list())
    net_bit_orig = net_bit.copy()

    # 监控背景网络中的调控因子数量
    initial_TF_count = net_bit['TF'].nunique()
    print(f"Number of regulators in the background network:: {initial_TF_count}")

    # pro-process data
    pca = PCA()
    exp_pca = pd.DataFrame(pca.fit_transform(exp), index=exp.index)
    exp_pca = exp_pca.drop(exp_pca.columns[-1], axis=1)
    exp_pca_discretized = pd.DataFrame()
    num_bins = 256
    for column in exp_pca.columns:
        bins = np.linspace(exp_pca[column].min(), exp_pca[column].max(), num_bins + 1)
        #      bins = exp_pca[column].quantile(q=np.linspace(0, 1, num_bins + 1))  # 根据分位数生成等频的区间
        labels = range(num_bins)
        exp_pca_discretized[column] = pd.cut(exp_pca[column], bins=bins, labels=labels, include_lowest=True)  # 执行离散化

    # add high link
    net_bit, corr_TF_Gene = high_pearson(exp_pca, net_bit, parm)
    net_bit, MI_TF_Gene = high_MI(exp_pca_discretized, exp_pca, net_bit, parm)

    if net_bit.shape[1] < 1:
        return None, None, None

    # creat adj
    nodes = np.unique(exp.index)
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for _, row in net_bit.iterrows():
        i = np.where(nodes == row['TF'])[0][0]
        j = np.where(nodes == row['Gene'])[0][0]
        adj_matrix[i, j] = 1

    if drop:
        GENE_ID_list, TF_ID_list = cal_del_TF_edge(exp.index)
        for TF_ID in TF_ID_list:
            for GENE_ID in GENE_ID_list:
                adj_matrix[GENE_ID, TF_ID] = 0  # Gene -> TF is error

    predicted_adj_matrix, new_graph = pca_cmi(exp_pca_discretized, net_bit, parm['pmi_percent'], 1)
    predicted_adj_matrix = predicted_adj_matrix.toarray()
    new_bit_crop = pd.DataFrame(new_graph.edges(), columns=['TF', 'Gene'])
    if np.sum(predicted_adj_matrix) == 0:
        new_row = { 'NUM_ORIG': 0, 'NUM_PCC': 0, 'NUM_MI': 0}
        return None, None, new_row
    elif (np.sum(adj_matrix) / np.sum(predicted_adj_matrix)) < 0.5:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(net_bit_orig, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, adj_matrix, new_row
    else:
        NUM_ORIG, NUM_PCC, NUM_MI, overflow = cal_percent(new_bit_crop, corr_TF_Gene, MI_TF_Gene, net_bit_orig)
        new_row = {'NUM_ORIG': NUM_ORIG, 'NUM_PCC': NUM_PCC, 'NUM_MI': NUM_MI}
        return exp, predicted_adj_matrix, new_row



if __name__=='__main__':
    data_path = ''
    filepath = os.path.join(data_path, ' ')  # Enter the path of the expression data
    Inpue_exp = pd.read_csv(filepath)
    Inpue_exp.set_index(Inpue_exp.columns[0], inplace=True)
    Inpue_exp.drop(columns=Inpue_exp.columns[0], inplace=True)
    Regnetwork_path ='2022.human.source'
    dtypes = {1: str, 3: str}
    human_network = pd.read_csv(Regnetwork_path, sep='\t', header=None, dtype=dtypes)
    parm = {'pear_percent': 4, 'MI_percent': 4, 'pmi_percent': 0.001}
    # pear_percent indicates the number of edges built by PCC *4 <= the number of edges in Regnetwork, that is, the high PCC does not exceed 20% of the total number of edges in the total network
    [exp, adj_matrix, new_row] = from_cancer_create(Inpue_exp, parm, human_network=human_network,drop=True)  # drop indicates the edge that deletes the gene-regulated TF
    regulation_counts = defaultdict(int)

    edges = [(exp.index[i], exp.index[j]) for i, j in zip(*np.where(adj_matrix > 0))]
    for edge in edges:
        regulation_counts[edge] += 1  # 计数

    gru_filepath = os.path.join(data_path, f'')
    pd.DataFrame(edges, columns=['TF', 'Target']).to_csv(gru_filepath, index=False)
    print(f"Gene regulatory networks have been saved to: {gru_filepath}")