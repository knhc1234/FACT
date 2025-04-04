import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

from tqdm import tqdm
from EarlyStopping import EarlyStopping
from Models import Final_Model
import Function as fun
import yaml

class CustomDataset(Dataset):
    def __init__(self, data1, data2, data3, targets):
        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.targets = targets

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx].to(device), self.data2[idx].to(device), self.data3[idx].to(device), self.targets[idx].to(device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    with open("../configs/FACT.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    # Parameters
    level = config["parameters"]["level"]
    adj_paths = config["parameters"]["adj_paths"]
    generate_new_negative_target = config["parameters"]["generate_new_negative_target"]
    
    node_atc_idx_path = '../data/level{}/node_to_idx_level{}.txt'.format(level, level)
    target_path = '../data/level{}/level{}_drug_ATC_pairwise.txt'.format(level, level)
    label_path = '../data/level{}/level{}_drug_ATC_mxn.txt'.format(level, level)
    atc_list_path = '../data/level{}/level{}_ATC_list.txt'.format(level, level)
    t_path = '../data/target/level{}_target.txt'.format(level)
    shuffle = True

    # Hyper Parameters
    epochs = config["hyperparameters"]["epochs"]
    early_stopping_count = config["hyperparameters"]["early_stopping_count"]
    batch_size = config["hyperparameters"]["batch_size"]
    num_layers = config["hyperparameters"]["num_layers"]
    tr_hidden = config["hyperparameters"]["tr_hidden"]
    num_head = config["hyperparameters"]["num_head"]
    ff_weight = config["hyperparameters"]["ff_weight"]
    dropout = config["hyperparameters"]["dropout"]
    learning_rate = config["hyperparameters"]["learning_rate"]

    title = "Final_layer{}_level{}_head{}_hidden{}_ff{}_dropout{}_lr{}".format(num_layers, level, num_head, tr_hidden, ff_weight, dropout, learning_rate)

    print(title)
    DM = fun.get_atc_labels(label_path).to(device) 
    n, SM = fun.generate_similarity_matrix(level, atc_list_path)
    
    SM = SM.to(device)
    FS1 = fun.get_adj(adj_paths[0]).to(device) 
    FS2 = fun.get_adj(adj_paths[1]).to(device) 
    FS3 = fun.get_adj(adj_paths[2]).to(device) 

    kf = KFold(n_splits=10)

    node_atc_idx_dict = fun.get_node_atc_idx_dict(node_atc_idx_path)

    atc_pairwise = fun.get_drug_ATC_pairwise(node_atc_idx_dict, target_path) 
    
    if generate_new_negative_target:
        print("check")
        negative_pairwise = fun.get_negative_target(t_path, atc_pairwise, n)
    else:
        negative_pairwise = fun.get_negative_target_from_file(t_path)
    average_roc_auc = []
    average_precision = []

    print(f'Positive Pair:{len(atc_pairwise)}, Negative Pair:{len(negative_pairwise)}')
    print(f'Level: {level}')
    for fold, (train_idx, test_idx) in enumerate(kf.split(atc_pairwise)):
        print(f'Fold {fold + 1}/{10}')
        train_target, test_target, train_label, test_label = fun.merge_pos_neg_pairwise(atc_pairwise, negative_pairwise, train_idx, test_idx)
        
        train_feature1, test_feature1 = fun.generate_feature(train_target, test_target, DM, SM, FS1)
        train_feature2, test_feature2 = fun.generate_feature(train_target, test_target, DM, SM, FS2)
        train_feature3, test_feature3 = fun.generate_feature(train_target, test_target, DM, SM, FS3)

        custom_dataset = CustomDataset(train_feature1, train_feature2, train_feature3, train_label)
        data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
        custom_test_dataset = CustomDataset(test_feature1, test_feature2, test_feature3, test_label)
        test_data_loader = DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=shuffle)

        model = Final_Model(input_dim = 2, layers = num_layers, hidden_dim = tr_hidden, num_head = num_head, ff_weight = ff_weight, drop_out = dropout, num_classes = 1).to(device)
        model_name = '../result/model/level{}/{}_{}.pt'.format(level, title, fold)
        
        model_dir = os.path.dirname(model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        n_epochs = epochs
        es = EarlyStopping(patience = early_stopping_count, verbose = True, delta=0, path=model_name)

        fold_best_label = None
        fold_best_out = None
        best_roc_auc = 0
        best_ap = 0
        for epoch in range(0, n_epochs):
            model.train()
            total_loss = 0.0
            for batch_data1, batch_data2, batch_data3, batch_labels in tqdm(data_loader):
                inputs1 = batch_data1.float()
                inputs1 = inputs1.squeeze(1).to(device)
                inputs2 = batch_data2.float()
                inputs2 = inputs2.squeeze(1).to(device)
                inputs3 = batch_data3.float()
                inputs3 = inputs3.squeeze(1).to(device)
                
                labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                out = model(inputs1, inputs2, inputs3)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            torch.cuda.empty_cache()
            print(f"Epoch : {epoch+1:4d}, Train Loss : {total_loss:.4f}") 
            
            with torch.no_grad():
                model.eval()
                total_loss = 0.0
                total_out = list()
                total_label = list()
                for batch_data1, batch_data2, batch_data3, batch_labels in tqdm(test_data_loader):
                    inputs1 = batch_data1.float()
                    inputs1 = inputs1.squeeze(1).to(device)
                    inputs2 = batch_data2.float()
                    inputs2 = inputs2.squeeze(1).to(device)
                    inputs3 = batch_data3.float()
                    inputs3 = inputs3.squeeze(1).to(device)
                    labels = batch_labels
                    
                    out = model(inputs1, inputs2, inputs3)
                    total_out.append(out)
                    loss = criterion(out, labels)
                    total_label.append(labels)
                    total_loss += loss.item()
                
                torch.cuda.empty_cache()
                print(f"Valid Loss : {total_loss:.4f}") 

                numpy_label_list = [tensor.cpu().numpy().flatten() for tensor in total_label]
                numpy_out_list = [tensor.cpu().numpy().flatten() for tensor in total_out]
                        
                all_labels = np.concatenate(numpy_label_list)
                all_out = np.concatenate(numpy_out_list)
                        
                roc_auc = roc_auc_score(all_labels, all_out)
                ap = average_precision_score(all_labels, all_out)
                
                if best_roc_auc < roc_auc:
                    best_roc_auc = roc_auc
                    best_ap = ap
                    fold_best_label = all_labels
                    fold_best_out = all_out
                print(f"Updated ROC AUC: {roc_auc}, AP: {ap}")
                
                es(-roc_auc, model)
                
                if es.early_stop:
                    break

        average_roc_auc.append(best_roc_auc)
        average_precision.append(best_ap)
        print(f"Best ROC AUC: {best_roc_auc}, Best AP: {best_ap}")
        result_file_path = "../result/level{}/{}_results_{}.txt".format(level, title, fold)
        with open(result_file_path, "w") as result_file:
            result_file.write("Drug\tATC\tout\ttest_label\n")
            for i in range(len(test_target)):
                value1 = test_target[i][0]
                value2 = test_target[i][1]
                out_value = fold_best_out[i]
                label_value = fold_best_label[i]
                result_file.write(f"{value1}\t{value2}\t{out_value}\t{label_value}\n")

    mean1 = np.mean(average_roc_auc)
    std1 = np.std(average_roc_auc)
    mean2 = np.mean(average_precision)
    std2 = np.std(average_precision)
    print(f"Average ROC AUC: {mean1} +- {std1}")
    print(f'All_ROC_AUC:{average_roc_auc}')
    print(f"Average AP: {mean2} +- {std2}")
    print(f'All_AP:{average_precision}')
