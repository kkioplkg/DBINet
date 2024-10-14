import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size
        self.all_head_size = hidden_size * num_attention_heads

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores_1(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_2(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, int(self.attention_head_size / self.num_attention_heads))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_1(mixed_query_layer)
        key_layer = self.transpose_for_scores_1(mixed_key_layer)
        value_layer = self.transpose_for_scores_1(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.attention_head_size * self.num_attention_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size * num_attention_heads, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class InteractiveFusionAttentionNetwork(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(InteractiveFusionAttentionNetwork, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, num_attention_heads, hidden_dropout_prob)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels, edge_index = data
        labels = labels.reshape(labels.shape[0], 1).to(device)
        labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs, edge_index)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        count += 1
        running_loss = 0.0
    return total_loss / count


def metric(compare):
    y_true = compare['true']
    y_pred = compare['pred']
    y_true = y_true.astype('int64')
    y_pred = y_pred.astype('int64')
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    y_prob = compare['prob']
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, tpr, thresholds = precision_recall_curve(y_true, y_prob)
    prc_auc = auc(tpr, prec)
    return roc_auc, prc_auc, bacc, precision, recall, kappa, f1


def valid(model, device, valid_loader, criterion):
    model.eval()
    compare = pd.DataFrame(columns=('pred', 'true'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels, edge_index = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            labels = labels.squeeze(1)

            outputs = model(inputs, edge_index)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            probs = F.softmax(outputs, dim=1)
            probs = probs[:, 1]
            _, predicteds = torch.max(outputs.data, 1)

            labels = labels.cpu()
            predicteds = predicteds.cpu()
            probs = probs.cpu()
            labels_list = np.array(labels).tolist()
            predicteds_list = np.array(predicteds).tolist()
            probs_list = np.array(probs).tolist()
            compare_temp = pd.DataFrame(columns=('pred', 'true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = predicteds_list
            compare_temp['prob'] = probs_list
            compare = pd.concat([compare, compare_temp])
    compare_copy = compare.copy()
    roc_auc, prc_auc, bacc, precision, recall, kappa, f1 = metric(compare_copy)
    return total_loss / count, roc_auc, prc_auc, bacc, precision, recall, kappa, f1


def save_model(current_f1, best_f1, epoch, model, optimizer, log_dir_best):
    is_best = current_f1 > best_f1
    best_f1 = max(current_f1, best_f1)
    checkpoint = {
        'best_f1': best_f1,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if is_best:
        torch.save(checkpoint, log_dir_best)
    return best_f1


def test(model, device, test_loader):
    model.eval()
    compare = pd.DataFrame(columns=('pred', 'true'))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, edge_index = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            labels = labels.squeeze(1)
            outputs = model(inputs, edge_index)

            probs = F.softmax(outputs, dim=1)
            probs = probs[:, 1]
            _, predicts = torch.max(outputs.data, 1)

            labels = labels.cpu()
            predicts = predicts.cpu()
            probs = probs.cpu()
            labels_list = np.array(labels).tolist()
            predicteds_list = np.array(predicts).tolist()
            probs_list = np.array(probs).tolist()
            compare_temp = pd.DataFrame(columns=('pred', 'true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = predicteds_list
            compare_temp['prob'] = probs_list
            compare = pd.concat([compare, compare_temp])
    return compare
