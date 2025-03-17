import numpy as np
import torch
import random

def get_adj(adj_path):      ##
    with open(adj_path, 'r') as file:
        lines = file.readlines()
    
    li = list()
    for i in lines:
        li.append(i.split())

    matrix = np.array(li, dtype=np.float64)
    torch_matrix = torch.tensor(matrix, dtype = torch.float64)
    return torch_matrix

def get_atc_labels(label_path): # 2841 x 14     ##
    with open(label_path, 'r') as file: 
        lines = file.readlines()

    
    header = lines[0].split()[:] 
    data_lines = lines[1:]
    
    
    matrix = np.zeros((len(data_lines), len(header)), dtype=float)

    
    for i, line in enumerate(data_lines):
        values = list(map(int, line.split()[1:]))  
        matrix[i, :] = values
    torch_matrix = torch.FloatTensor(matrix)
    return torch_matrix

def get_atc_labels2(label_path): # 2841 x 14
    with open(label_path, 'r') as file:
        lines = file.readlines()

    
    header = lines[0].split()[:] 
    data_lines = lines[1:]
    
    
    matrix = np.zeros((len(data_lines), len(header)), dtype=float)

    
    for i, line in enumerate(data_lines):
        values = list(map(int, line.split()[1:]))  
        matrix[i, :] = values
    return matrix

def get_node_atc_idx_dict(node_atc_to_idx_path):        ##
    with open(node_atc_to_idx_path, 'r') as file:
        lines = file.readlines()

    data = dict()
    for line in lines:
        i = line.strip().split('\t')
        data[i[0]] = int(i[1])
    return data

def get_drug_ATC_pairwise(node_atc_idx_dict, pairwise_path): 
    with open(pairwise_path, 'r') as file:
        lines = file.readlines()
    
    atc_pairwise = list()

    for line in lines:
        j = line.strip().split('\t')
        j[0] = node_atc_idx_dict[j[0]]
        j[1] = node_atc_idx_dict[j[1]] - 2841
        atc_pairwise.append((j[0], j[1]))
    
    return atc_pairwise 

def get_negative_target_from_file(target_path):
    negative_pairwise = []
    with open(target_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split('\t')
            negative_pairwise.append((int(values[0]), int(values[1])))
    return negative_pairwise

def get_negative_target(target_path, pairwise, size):
    all_pair = set(pairwise)

    negative_pairwise = list()
    n = len(pairwise)

    for i in range(n):
        node = random.randint(0, 2840)
        atc = random.randint(0, size - 1)
        while True:
            if (node, atc) not in all_pair:
                negative_pairwise.append((node, atc))
                all_pair.add((node, atc))
                break
            else:
                node = random.randint(0, 2840)
                atc = random.randint(0, size - 1)

    with open(target_path, 'w') as file:
        for pair in negative_pairwise:
            file.write(f"{pair[0]}\t{pair[1]}\n")

    return negative_pairwise

def merge_pos_neg_pairwise(atc_pairwise, negative_pairwise, train_idx, test_idx):
    train_pos_labels = torch.ones(len(train_idx), 1)
    train_neg_labels = torch.zeros(len(train_idx), 1)
    train_labels = torch.cat((train_pos_labels, train_neg_labels), 0)

    test_pos_labels = torch.ones(len(test_idx), 1)
    test_neg_labels = torch.zeros(len(test_idx), 1)
    test_labels = torch.cat((test_pos_labels, test_neg_labels), 0)

    train_pair = [atc_pairwise[i] for i in train_idx]
    test_pair = [atc_pairwise[i] for i in test_idx]

    for i in train_idx:
        train_pair.append(negative_pairwise[i])

    for i in test_idx:
        test_pair.append(negative_pairwise[i])

    return train_pair, test_pair, train_labels, test_labels

def generate_feature(train_target, test_target, DM, SM, FS): 
    t1 = len(train_target)
    t2 = len(test_target)
    train_features = torch.zeros(t1, 2, FS.size(0) + SM.size(1))
    test_features = torch.zeros(t2, 2, FS.size(0) + SM.size(1))
    
    # Validation Answer Deletion
    DM_batch = DM.clone()
    for t in test_target:   
        DM_batch[t[0]][t[1]] = 0
    DM_batch_t = DM_batch.t()
    
    for i, t in enumerate(train_target):
        DM_batch1 = DM_batch.clone()
        DM_batch1[t[0]][t[1]] = 0
        DM_batch1_t = DM_batch1.t()
        feature1 = torch.cat((DM_batch1[t[0]], FS[t[0]]), dim=0).unsqueeze(0).unsqueeze(0)  
        feature2 = torch.cat((SM[t[1]], DM_batch1_t[t[1]]), dim = 0).unsqueeze(0).unsqueeze(0)  
        train_features[i, 0] = feature1
        train_features[i, 1] = feature2
        del(DM_batch1)
        del(DM_batch1_t)

    for i, t in enumerate(test_target):
        #DM_batch = DM.clone()
        #DM_batch[t[0]][t[1]] = 0
        #DM_batch_t = DM_batch.t()
        feature1 = torch.cat((DM_batch[t[0]], FS[t[0]]), dim=0).unsqueeze(0).unsqueeze(0)  
        feature2 = torch.cat((SM[t[1]], DM_batch_t[t[1]]), dim = 0).unsqueeze(0).unsqueeze(0)  
        test_features[i, 0] = feature1
        test_features[i, 1] = feature2
        #del(DM_batch)
        #del(DM_batch_t)
    del(DM_batch_t)
    del(DM_batch)
    return train_features, test_features

def generate_similarity_matrix(level, atc_list_path):       ##
    # Initialize an empty similarity matrix
    with open(atc_list_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        all_atc = line.split()
    n = len(all_atc)
    
    if level == 1:
        similarity_matrix = torch.ones(14,14) 
        for i in range(14):
            for j in range(14):
                if i != j:
                    similarity_matrix[i,j] = 1.0 / 3
    else:
        similarity_matrix = torch.zeros(n, n)
        level2_weight_sum = [0, 4, 5]
        level3_weight_sum = [0, 9, 13, 14]
        level4_weight_sum = [0, 16, 25, 29, 30] 
        bias = level * level
        # Calculate similarities between pairs of values
        for i in range(n):
            for j in range(i, n):
                # Create sets based on the format {a, a01} from input values
                if level == 2:
                    set_a = set([all_atc[i][0], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j]])
                elif level == 3:
                    set_a = set([all_atc[i][0], all_atc[i][:3], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j][:3], all_atc[j]])
                elif level == 4:
                    set_a = set([all_atc[i][0], all_atc[i][:3], all_atc[i][:4], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j][:3], all_atc[j][:4], all_atc[j]])
                # Calculate similarity using the formula
                intersection = len(set_a.intersection(set_b))
                
                if level == 2:
                    similarity = (2 * level2_weight_sum[intersection] + bias) / (2 * level2_weight_sum[-1] + bias)
                elif level == 3:
                    similarity = (2 * level3_weight_sum[intersection] + bias) / (2 * level3_weight_sum[-1] + bias)
                elif level == 4:
                    similarity = (2 * level4_weight_sum[intersection] + bias) / (2 * level4_weight_sum[-1] + bias) # 17/76, 49/76, 67/76, 74/76
                # Update similarity values in the matrix (symmetric matrix)
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Since it's symmetric
    print(similarity_matrix.shape)
    return n, similarity_matrix

def generate_similarity_matrix_old(level, atc_list_path):
    # Initialize an empty similarity matrix
    with open(atc_list_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        all_atc = line.split()
    n = len(all_atc)
    
    if level == 1:
        similarity_matrix = torch.ones(14,14) 
        for i in range(14):
            for j in range(14):
                if i != j:
                    similarity_matrix[i,j] = 1.0 / 3
    else:
        similarity_matrix = torch.zeros(n, n)
        # Calculate similarities between pairs of values
        for i in range(n):
            for j in range(i, n):
                # Create sets based on the format {a, a01} from input values
                if level == 2:
                    set_a = set([all_atc[i][0], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j]])
                elif level == 3:
                    set_a = set([all_atc[i][0], all_atc[i][:3], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j][:3], all_atc[j]])
                elif level == 4:
                    set_a = set([all_atc[i][0], all_atc[i][:3], all_atc[i][:4], all_atc[i]])
                    set_b = set([all_atc[j][0], all_atc[j][:3], all_atc[j][:4], all_atc[j]])
                # Calculate similarity using the formula
                intersection = len(set_a.intersection(set_b))
                
                similarity = (2 * intersection + 1) / (2 * level + 1) 
                # Update similarity values in the matrix (symmetric matrix)
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Since it's symmetric
    print(similarity_matrix.shape)
    return n, similarity_matrix
    
def generate_npsim(level, adj_path, label_path):
    # 1. Chemical similarity matrix
    #Sd = np.loadtxt(adj_path)
    
    # Sd_normalized = (Sd - np.min(Sd)) / (np.max(Sd) - np.min(Sd))
    
    # 2. Level 1 Drug-ATC matrix
    Adj = get_atc_labels2(label_path)
    
    # 3. Column_sum
    column_sum = np.sum(Adj, axis=0)
    all_sum = np.sum(column_sum)
    
    m = len(column_sum)
    
    # 4. A matrix Generate
    A_np = np.random.rand(m)
    for i in range(m):
        A_np[i] = all_sum / column_sum[i]
    A = torch.tensor(A_np)
    
    # 5. Level n ATC distance matrix
    sim_path = '../data/level{}/level{}_ATC_distance_matrix.txt'.format(level, level)
    S_ATC_org_numpy = np.loadtxt(sim_path)
    S_ATC_org = torch.tensor(S_ATC_org_numpy)
    # 6. S_ATC Matrix Generate
    S_ATC = torch.zeros(m, m)
    
    for i in range(m):
        for j in range(m):
            S_ATC[i, j] = A[i] * A[j] * torch.exp(-0.25 * S_ATC_org[i, j])

    S_ATC_min = torch.min(S_ATC)
    S_ATC_max = torch.max(S_ATC)
    
    S_ATC_normalized = (S_ATC - S_ATC_min) / (S_ATC_max - S_ATC_min)
    return m, S_ATC_normalized