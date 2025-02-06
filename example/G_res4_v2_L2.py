from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN5 import GENELink
from torch.optim.lr_scheduler import StepLR
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
import numpy as np
import random
import os
import time
from Arguments import parser


def embed2file(tf_embed, tg_embed, bias_weight, gene_file, tf_path, target_path, bias_weight_path=None):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed, index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

    if bias_weight is not None:
        bias_weight = bias_weight.cpu().detach().numpy()
        bias_weight_df = pd.DataFrame(bias_weight)
        bias_weight_df.to_csv(bias_weight_path)



def train(args):
    # Set seed
    seed = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_type = args.data_type
    num = args.num
    net_type = args.net_type

    # Data load
    density = Network_Statistic(data_type, num, net_type)

    # Paths
    if os.name == 'nt':  # Windows
        base_dir = ''
    else:  # Assume it's running on Linux/Unix
        base_dir = ''

    dataset_dir = os.path.join(base_dir, 'dataset', net_type, data_type, f'TFs+{num}')
    train_test_val_dir = os.path.join(base_dir, 'Train_validation_test', net_type, data_type, f'TFs+{num}')
    result_dir = os.path.join(base_dir, 'Result', data_type, str(num))
    model_dir = os.path.join(base_dir, 'model')

    for dir_path in [result_dir, model_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    exp_file = os.path.join(dataset_dir, 'BL--ExpressionData.csv')
    tf_file = os.path.join(dataset_dir, 'TF.csv')
    target_file = os.path.join(dataset_dir, 'Target.csv')

    # Load data (feature)
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()

    # TF and Target
    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)

    feature = torch.from_numpy(feature)
    tf = torch.from_numpy(tf)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_feature = feature.to(device)
    tf = tf.to(device)

    train_file = os.path.join(train_test_val_dir, 'Train_set.csv')
    test_file = os.path.join(train_test_val_dir, 'Test_set.csv')
    val_file = os.path.join(train_test_val_dir, 'Validation_set.csv')

    tf_embed_path = r'Result/' + data_type + ' ' + str(num) + '/Channel1.csv'
    target_embed_path = r'Result/' + data_type + ' ' + str(num) + '/Channel2.csv'
    bias_weight_path = r'Result/' + data_type + ' ' + str(num) + '/Bias_Weight.csv'  # Add path for bias_weight
    if not os.path.exists('Result/' + data_type + ' ' + str(num)):
        os.makedirs('Result/' + data_type + ' ' + str(num))

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

    adj = train_load.Adj_Generate(tf, loop=args.loop)
    adj = adj2saprse_tensor(adj)

    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
    val_data = torch.from_numpy(validation_data)

    model = GENELink(input_dim=feature.size()[1],
                     hidden1_dim=args.hidden_dim[0],
                     hidden2_dim=args.hidden_dim[1],
                     hidden3_dim=args.hidden_dim[2],
                     hidden4_dim=args.hidden_dim[3],
                     output_dim=args.output_dim,
                     num_head1=args.num_head[0],
                     num_head2=args.num_head[1],
                     num_head3=args.num_head[2],
                     alpha=args.alpha,
                     device=device,
                     type=args.Type,
                     reduction=args.reduction
                     )

    adj = adj.to(device)
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    validation_data = val_data.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.015)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.999)

    model_path = 'model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Train
    start_time = time.time()  # time start

    for epoch in range(args.epochs):
        running_loss = 0.0

        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
            model.train()
            optimizer.zero_grad()

            if args.flag:
                train_y = train_y.to(device)
            else:
                train_y = train_y.to(device).view(-1, 1)

            pred = model(data_feature, adj, train_x)

            if args.flag:
                pred = torch.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)

            loss_BCE = F.binary_cross_entropy(pred, train_y)
            loss_BCE.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss_BCE.item()

        model.eval()
        score = model(data_feature, adj, validation_data)

        if args.flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)

        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1], flag=args.flag)
        print('Epoch:{}'.format(epoch + 1),
              'train loss:{}'.format(running_loss),
              'AUC:{:.3F}'.format(AUC),
              'AUPR:{:.3F}'.format(AUPR))

    end_time = time.time()  # time end
    total_time = end_time - start_time

    print(f'Total training time: {total_time:.2f} seconds')

    torch.save(model.state_dict(), model_path + data_type + ' ' + str(num) + '.pkl')
    model.load_state_dict(torch.load(model_path + data_type + ' ' + str(num) + '.pkl'))

    # Test
    model.eval()
    tf_embed, target_embed, bias_weight = model.get_embedding()
    embed2file(tf_embed, target_embed, bias_weight, target_file, tf_embed_path, target_embed_path, bias_weight_path)

    score = model(data_feature, adj, test_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=args.flag)

    print('AUC:{}'.format(AUC),
          'AUPRC:{}'.format(AUPR))


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)







