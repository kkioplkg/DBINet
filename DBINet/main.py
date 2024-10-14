import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random
import os
import config
from model import MolecularInteractionFusionModule, FingerprintFeatureEnhancementModule, SynergyFusionPredictionModule
from utils_data import split, load_data
from utils_model import train, valid, save_model, test, metric
import warnings

warnings.filterwarnings("ignore")

torch.set_num_threads(32)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(fold, repeat):
    setup_seed(42)

    transform = transforms.ToTensor()
    path_train = f'./data/repeat{repeat}_fold{fold}_train.csv'
    path_val = f'./data/repeat{repeat}_fold{fold}_val.csv'
    edge_index_path = f'./data/repeat{repeat}_fold{fold}_edge_index.npy'
    train_data = load_data(path_train, edge_index_path, transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data = load_data(path_val, edge_index_path, transform)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    result_dir = f'./results/main/repeat{repeat}'
    os.makedirs(result_dir, exist_ok=True)
    log_dir = os.path.join(result_dir, f'fold{fold}.csv')
    with open(log_dir, 'w') as f:
        f.write("Epoch,Train_loss,Valid_loss,ROC_AUC,PR_AUC,BACC,Precision,Recall,Kappa,F1\n")

    model_View1 = MolecularInteractionFusionModule(model_config, cell.shape[1])
    model_View2 = FingerprintFeatureEnhancementModule(model_config, drug_fp, cell)
    model = SynergyFusionPredictionModule(model_config, drug_fp, cell, model_View1, model_View2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=False)
    start_epoch = 0
    best_f1 = 0
    best_model_path = os.path.join(result_dir, f'fold{fold}_best.pth')

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        valid_loss, roc_auc, prc_auc, bacc, precision, recall, kappa, f1 = valid(model, device, valid_loader,
                                                                                 criterion)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, best_model_path)

        res = f"{epoch},{train_loss:.3f},{valid_loss:.3f},{roc_auc:.3f},{prc_auc:.3f},{bacc:.3f},{precision:.3f},{recall:.3f},{kappa:.3f},{f1:.3f}"
        print(f"Epoch {epoch}: {res}")
        with open(log_dir, "a") as f:
            f.write(res + '\n')


def get_res(repeat):
    setup_seed(42)
    for fold in range(1, 6):
        print(f'Start test, repeat {repeat}, fold {fold}')
        transform = transforms.ToTensor()
        path_test = f'./data/repeat{repeat}_fold{fold}_test.csv'
        edge_index_path = f'./data/repeat{repeat}_fold{fold}_edge_index.npy'
        test_data = load_data(path_test, edge_index_path, transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model_View1 = MolecularInteractionFusionModule(model_config, cell.shape[1])
        model_View2 = FingerprintFeatureEnhancementModule(model_config, drug_fp, cell)
        model = SynergyFusionPredictionModule(model_config, drug_fp, cell, model_View1, model_View2)
        model.to(device)

        result_dir = f'./results/main/repeat{repeat}'
        best_model_path = os.path.join(result_dir, f'fold{fold}_best.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print('Load saved model')
        else:
            print('No saved model')
            break

        if fold == 1:
            test_data_pd = pd.read_csv(path_test)
            compare = test(model, device, test_loader)
            compare = compare.reset_index(drop=True)
            test_data_pd['true'] = compare['true'].astype(int)
            test_data_pd['pred'] = compare['pred'].astype(int)
            test_data_pd['prob'] = compare['prob']
        else:
            test_data_pd_temp = pd.read_csv(path_test)
            compare_temp = test(model, device, test_loader)
            compare_temp = compare_temp.reset_index(drop=True)
            test_data_pd_temp['true'] = compare_temp['true'].astype(int)
            test_data_pd_temp['pred'] = compare_temp['pred'].astype(int)
            test_data_pd_temp['prob'] = compare_temp['prob']
            test_data_pd = pd.concat([test_data_pd, test_data_pd_temp])

    pred_dir = os.path.join(result_dir, f'repeat{repeat}_predict.csv')
    test_data_pd.to_csv(pred_dir, index=False)

    roc_auc, prc_auc, bacc, precision, recall, kappa, f1 = metric(test_data_pd)
    res = f'ROC-AUC: {roc_auc:.3f}, PR-AUC: {prc_auc:.3f}, '
    res += f'BACC: {bacc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, '
    res += f'Kappa: {kappa:.3f}, F1: {f1:.3f}'
    print(res)

    res_dir = os.path.join(result_dir, f'repeat{repeat}_metric.txt')
    with open(res_dir, "w") as f:
        f.write(res + '\n')


if __name__ == '__main__':
    split_flag = 1
    train_flag = 1
    test_flag = 1

    if split_flag == 1:
        print('load data')
        # print('划分数据')
        data = pd.read_csv('./Data/data_to_split.csv')
        data = data.drop(columns='Unnamed: 0')
        for repeat in range(1, 3):
            split(data, repeat=repeat)

    print('Load config')
    model_config = config.model_config
    gpu = model_config['gpu']
    batch_size = model_config['batch_size']
    criterion = nn.CrossEntropyLoss()
    lr = model_config['lr']
    epochs = model_config['epochs']
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    cell = pd.read_csv('./Data/Cell_use.csv')
    cell = cell.drop(labels=['id'], axis=1)
    drug_fp = pd.read_csv('./Data/Drug_use.csv')
    drug_fp = drug_fp.drop(labels=['id'], axis=1)

    for repeat in range(1, 3):
        for fold in range(1, 6):
            if train_flag == 1:
                print(f'Start train, repeat {repeat}, fold {fold}')
                run(fold, repeat)
        if test_flag == 1:
            get_res(repeat)
        torch.cuda.empty_cache()
