import pandas as pd
import numpy as np
import os
from utils import Network_Statistic
import networkx as nx
from Arguments import parser


def train_val_test_set(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, p_val):
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]

        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]

            val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
            train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]

    max_attempts = 2000  # try times

    train_neg = {}
    for k in train_pos.keys():
        train_neg[k] = []
        attempts = 0
        for i in range(len(train_pos[k])):
            while attempts < max_attempts:
                neg = np.random.choice(gene_set)
                if neg != k and neg not in pos_dict[k] and neg not in train_neg[k]:
                    train_neg[k].append(neg)
                    break
                attempts += 1
            else:
                print(
                    f"Warning: Unable to find negative example for gene {k} after {max_attempts} attempts. Skipping...")
                break

    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]

    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        attempts = 0
        for i in range(len(val_pos[k])):
            while attempts < max_attempts:
                neg = np.random.choice(gene_set)
                if neg != k and neg not in pos_dict[k] and neg not in train_neg[k] and neg not in val_neg[k]:
                    val_neg[k].append(neg)
                    break
                attempts += 1
            else:
                print(
                    f"Warning: Unable to find negative example for gene {k} after {max_attempts} attempts. Skipping...")
                break

    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k, j])

    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label

    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:, 0]
    val_sample['Target'] = val_set_a[:, 1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)

    print("Validation set construction completed.")

    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    # print(test_pos_set)

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density - count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_neg_set = []
    for i in range(test_neg_num):
        attempts = 0
        while attempts < max_attempts:
            t1 = np.random.choice(tf_set)
            t2 = np.random.choice(gene_set)
            if t1 != t2 and [t1, t2] not in train_set and [t1, t2] not in test_pos_set and [t1, t2] not in val_set and [t1, t2] not in test_neg_set:
                test_neg_set.append([t1, t2])
                # print([t1, t2])
                break
            attempts += 1
        else:
            print(
                f"Warning: Unable to find negative example for gene {t1} after {max_attempts} attempts. Skipping...")

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)

    print("Test set construction completed.")

def train_val_test_set_with_distance(label_file, Gene_file, TF_file, train_set_file, val_set_file, test_set_file, density, p_val):

    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)

    G = nx.Graph()
    for i, j in label.values:
        G.add_edge(i, j)

    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    pos_dict = {tf: [] for tf in tf_set}
    for i, j in label.values:
        pos_dict[i].append(j)

    train_pos, val_pos, test_pos = {}, {}, {}
    train_neg = {}

    max_attempts = 2000  # 尝试次数机制

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0, 1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 2 // 3]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 2 // 3:]
            val_pos[k] = train_pos[k][:len(train_pos[k]) // 5]
            train_pos[k] = train_pos[k][len(train_pos[k]) // 5:]

    for k in train_pos.keys():
        train_neg[k] = []
        for _ in range(len(train_pos[k])):
            attempts = 0
            while attempts < max_attempts:
                possible_negs = [gene for gene in gene_set if gene in path_lengths[k] and path_lengths[k][gene] > 3]
                if not possible_negs:
                    print(f"Warning: No suitable negative example for {k} after {max_attempts} attempts.")
                    break
                neg = np.random.choice(possible_negs)
                if neg not in train_neg[k]:
                    train_neg[k].append(neg)
                    break
                attempts += 1
            else:
                print(f"Warning: Unable to find negative example for gene {k} after {max_attempts} attempts. Skipping...")

    train_pos_set = []
    train_neg_set = []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    tran_pos_label = [1 for _ in range(len(train_pos_set))]

    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])
    tran_neg_label = [0 for _ in range(len(train_neg_set))]

    train_set = train_pos_set + train_neg_set
    train_label = tran_pos_label + tran_neg_label

    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    val_pos_label = [1 for _ in range(len(val_pos_set))]

    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            attempts = 0
            while attempts < max_attempts:
                neg = np.random.choice(gene_set)
                if neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                    attempts += 1
                    continue
                val_neg[k].append(neg)
                break
            else:
                print(f"Warning: Unable to find negative example for gene {k} after {max_attempts} attempts. Skipping...")

    val_neg_set = []
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set.append([k, j])

    val_neg_label = [0 for _ in range(len(val_neg_set))]
    val_set = val_pos_set + val_neg_set
    val_set_label = val_pos_label + val_neg_label

    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    val_sample['TF'] = val_set_a[:, 0]
    val_sample['Target'] = val_set_a[:, 1]
    val_sample['Label'] = val_set_label
    val_sample.to_csv(val_set_file)

    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density - count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_neg_set = []
    for i in range(test_neg_num):
        attempts = 0
        while attempts < max_attempts:
            t1 = np.random.choice(tf_set)
            t2 = np.random.choice(gene_set)
            if t1 != t2 and [t1, t2] not in train_set and [t1, t2] not in test_pos_set and [t1, t2] not in val_set and [t1, t2] not in test_neg_set:
                test_neg_set.append([t1, t2])
                break
            attempts += 1
        else:
            print(f"Warning: Unable to find negative example for gene {t1} after {max_attempts} attempts. Skipping...")

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)


def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file,val_set_file,test_set_file,
                                          ratio, p_val):
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)

        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        if len(pos_dict[k]) ==1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) ==2:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k])*ratio)]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k])*ratio):int(len(pos_dict[k])*(ratio+0.1))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k])*(ratio+0.1)):]

    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_num = len(neg_dict[k])
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(neg_num*ratio)]
        val_neg[k] = neg_dict[k][int(neg_num*ratio):int(neg_num*(0.1+ratio))]
        test_neg[k] = neg_dict[k][int(neg_num*(0.1+ratio)):]



    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k,val])

    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k,val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]



    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k,val])

    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k,val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_set_file)




    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k,j])

    test_neg_set = []
    for k in test_neg.keys():
        for j in test_neg[k]:
            test_neg_set.append([k,j])




    test_set = test_pos_set +test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:,0]
    test['Target'] = test_sample[:,1]
    test['Label'] = test_label
    test.to_csv(test_set_file)

def spilt(args):
    data_type = args.data_type
    net_type = args.net_type

    density = Network_Statistic(data_type=data_type, net_scale=args.num, net_type=net_type)

    base_dir = ''
    dataset_dir = os.path.join(base_dir, 'dataset', net_type, data_type, f'TFs+{args.num}')
    train_test_val_dir = os.path.join(base_dir, 'Train_validation_test', net_type, data_type, f'TFs+{args.num}')

    TF2file = os.path.join(dataset_dir, 'TF.csv')
    Gene2file = os.path.join(dataset_dir, 'Target.csv')
    label_file = os.path.join(dataset_dir, 'Label.csv')

    train_set_file = os.path.join(train_test_val_dir, 'Train_set.csv')
    test_set_file = os.path.join(train_test_val_dir, 'Test_set.csv')
    val_set_file = os.path.join(train_test_val_dir, 'Validation_set.csv')

    if not os.path.exists(train_test_val_dir):
        os.makedirs(train_test_val_dir)

    if args.net_type == 'Specific':
        Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
                                              test_set_file, ratio=args.ratio, p_val=args.p_val)
    elif args.use_distance_method == 'yes':
        train_val_test_set_with_distance(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file,
                                         density, p_val=args.p_val)
    else:
        train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file,
                           density, p_val=args.p_val)


if __name__ == '__main__':
    args = parser.parse_args()
    spilt(args)


