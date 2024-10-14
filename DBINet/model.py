import math
import numpy as np
from ban import BANLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from utils_model import InteractiveFusionAttentionNetwork


def clean_edge_index(edge_index, num_nodes):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    valid_mask = (edge_index >= 0) & (edge_index < num_nodes)
    valid_mask = valid_mask.all(dim=0)
    edge_index = edge_index[:, valid_mask]

    return edge_index


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output


class SynergyFusionPredictionModule(nn.Module):
    def __init__(self, model_config, drug_fp, cell, model_View1, model_View2):
        super(SynergyFusionPredictionModule, self).__init__()
        h_dim = model_config['h_dim']
        dropout_prob = model_config['dropout']
        gpu = model_config['gpu']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')

        self.cell = torch.Tensor(np.array(cell))
        self.cell = self.cell.to(self.device)

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob

        self.model_View1 = model_View1
        self.model_View2_A = model_View2
        self.model_View2_B = model_View2

        self.linear_adjust_view1 = nn.Linear(h_dim * 2, h_dim)
        self.linear_adjust_view2 = nn.Linear(1024, h_dim)

        self.classifier1 = nn.Linear(h_dim * 3, 2048)
        self.classifier_bn1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.classifier2 = nn.Linear(2048, 1024)
        self.classifier_bn2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.classifier3 = nn.Linear(1024, 2)

        self.multi_classifier1 = nn.Linear(h_dim * 3, 2048)
        self.multi_classifier_bn1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.multi_classifier2 = nn.Linear(2048, 1024)
        self.multi_classifier_bn2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.multi_classifier3 = nn.Linear(1024, 7)

        self.attention = InteractiveFusionAttentionNetwork(h_dim, 8, dropout_prob, dropout_prob)  # 多模态注意力机制

    def forward(self, triplets, edge_index):
        triplets = triplets.long()
        c = self.cell[triplets[:, 2]]

        x_view1 = self.model_View1(triplets, c, edge_index)
        x_view1 = self.linear_adjust_view1(x_view1)

        xA = self.model_View2_A(triplets[:, [0, 2]])
        xA = self.linear_adjust_view2(xA)

        xB = self.model_View2_B(triplets[:, [1, 2]])
        xB = self.linear_adjust_view2(xB)

        total1 = torch.stack((x_view1, xA, xB), 1)
        attention_output1 = self.attention(total1)
        total_1 = attention_output1.view(attention_output1.shape[0], -1)

        classification = F.relu(self.classifier_bn1(self.classifier1(total_1)), inplace=True)
        classification = F.dropout(classification, training=self.training, p=self.dropout_prob)
        classification = F.relu(self.classifier_bn2(self.classifier2(classification)), inplace=True)
        classification = F.dropout(classification, training=self.training, p=self.dropout_prob)
        classification = self.classifier3(classification)

        multi_class = F.relu(self.multi_classifier_bn1(self.multi_classifier1(total_1)), inplace=True)
        multi_class = F.dropout(multi_class, training=self.training, p=self.dropout_prob)
        multi_class = F.relu(self.multi_classifier_bn2(self.multi_classifier2(multi_class)), inplace=True)
        multi_class = F.dropout(multi_class, training=self.training, p=self.dropout_prob)
        multi_class = self.multi_classifier3(multi_class)

        return classification


class MolecularInteractionFusionModule(nn.Module):
    def __init__(self, model_config, cell_features_dim):
        super(MolecularInteractionFusionModule, self).__init__()
        h_dim = model_config['h_dim']
        in_dim_drug = model_config['in_dim_drug']
        num_heads = model_config['num_heads']
        dropout = model_config['dropout']
        gpu = model_config['gpu']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.drug_linear = nn.Linear(4, in_dim_drug)
        nn.init.kaiming_normal_(self.drug_linear.weight)

        self.drug_gcn1 = GATConv(in_dim_drug, h_dim, heads=num_heads, dropout=dropout)
        self.gcn1_linear = nn.Linear(h_dim * num_heads, h_dim)

        self.drug_gcn2 = GATConv(h_dim, h_dim, heads=5, dropout=dropout)
        self.gcn2_linear = nn.Linear(h_dim * 5, h_dim)

        self.drug_fc = nn.Linear(h_dim, h_dim * 2)
        nn.init.kaiming_normal_(self.drug_fc.weight)

        self.drug_pair_hg_fc = nn.Linear(h_dim * 2, h_dim)
        nn.init.kaiming_normal_(self.drug_pair_hg_fc.weight)

        self.ban_layer = BANLayer(v_dim=h_dim, q_dim=h_dim, h_dim=h_dim, h_out=num_heads)

        self.cell_fc = nn.Linear(cell_features_dim, h_dim)
        nn.init.kaiming_normal_(self.cell_fc.weight)

    def forward(self, triplets, cell_features, edge_index):
        triplets = triplets.long()

        num_nodes = torch.max(edge_index) + 1

        edge_index = clean_edge_index(edge_index, num_nodes)

        edge_index = edge_index.view(2, -1).to(self.device)

        drug_pair_h = triplets[:, :4].to(self.device).float()
        drug_pair_h = self.drug_linear(drug_pair_h)

        drug_pair_h1 = F.elu(self.drug_gcn1(drug_pair_h, edge_index))
        drug_pair_h1 = self.gcn1_linear(drug_pair_h1)

        drug_pair_h2 = F.elu(self.drug_gcn2(drug_pair_h1, edge_index))
        drug_pair_h2 = self.gcn2_linear(drug_pair_h2)

        drug_pair_h = drug_pair_h1 + drug_pair_h2
        drug_pair_h = torch.mean(drug_pair_h, dim=0)

        drug_pair_hg = self.drug_fc(drug_pair_h)
        drug_pair_hg = self.relu(drug_pair_hg)
        drug_pair_hg = self.drug_pair_hg_fc(drug_pair_hg)

        c = self.cell_fc(cell_features)
        c = self.relu(c)

        drug_pair_hg = drug_pair_hg.unsqueeze(0).expand(c.size(0), -1)
        joint_representation, _ = self.ban_layer(drug_pair_hg, c)

        return joint_representation


class FingerprintFeatureEnhancementModule(nn.Module):
    def __init__(self, model_config, drug_fp, cell):
        super(FingerprintFeatureEnhancementModule, self).__init__()
        h_dim = model_config['h_dim']
        dropout = model_config['dropout']
        gpu = model_config['gpu']
        in_dim_cell = model_config['in_dim_cell']
        if gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(gpu)
        else:
            self.device = torch.device('cpu')

        self.drug_fp = torch.Tensor(np.array(drug_fp))
        self.drug_fp = self.drug_fp.to(self.device)
        self.cell = torch.Tensor(np.array(cell))
        self.cell = self.cell.to(self.device)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.cell_fc = nn.Linear(in_dim_cell, h_dim)
        nn.init.kaiming_normal_(self.cell_fc.weight)

        self.fp_cell_fc1 = nn.Linear(h_dim, 2048)
        nn.init.kaiming_normal_(self.fp_cell_fc1.weight)
        self.fp_cell_fc2 = nn.Linear(2048, 1024)
        nn.init.kaiming_normal_(self.fp_cell_fc2.weight)

    def forward(self, drug_cell):
        drug_cell = drug_cell.long()
        c = self.cell[drug_cell[:, 1]]
        c = self.cell_fc(c)
        c = self.relu(c)

        drug_fp = self.drug_fp[drug_cell[:, 0]].to(torch.float32)
        x = torch.cat((drug_fp, c), 1)
        del drug_fp, c
        x = self.relu(self.fp_cell_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fp_cell_fc2(x))
        x = self.dropout(x)

        return x

