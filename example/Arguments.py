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

