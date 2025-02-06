import pandas as pd
import networkx as nx
from Arguments import parser

def st():
    file_path = 'Result\B 500\B_out.csv'
    df = pd.read_csv(file_path, usecols=[0, 1], header=None, skiprows=1)

    G = nx.DiGraph()

    G.add_edges_from(zip(df[0], df[1]))

    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    clustering_coefficient = nx.clustering(G.to_undirected())

    result = pd.DataFrame({
        'Gene': list(G.nodes()),
        'In-Degree Centrality': [in_degree_centrality[node] for node in G.nodes()],
        'Out-Degree Centrality': [out_degree_centrality[node] for node in G.nodes()],
        'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Closeness Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Clustering Coefficient': [clustering_coefficient[node] for node in G.nodes()]
    })

    output_path = 'Result\B 500\B_st.csv'
    result.to_csv(output_path, index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    st()