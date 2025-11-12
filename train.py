# train.py

import argparse
import random
import os
import yaml
from tqdm import tqdm
import numpy as np
import torch
import pickle
from CCAHCL.loader import DatasetLoader
from CCAHCL.models import HyperEncoder, CCAHCL
from CCAHCL.utils import drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking
from CCAHCL.evaluation import linear_evaluation


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def fuse_node_with_edges(node_emb, edge_emb, hyperedge_index, alpha=0.5):
    """
    node_emb: [N, d]
    edge_emb: [E, d]
    hyperedge_index: [2, num_edges], COO 格式表示超图关系
    alpha: 融合权重
    """
    N = node_emb.shape[0]
    d = node_emb.shape[1]

    device = node_emb.device

    # 初始化 edge 聚合表示
    edge_agg = torch.zeros_like(node_emb)

    # hyperedge_index[0] 是节点索引，hyperedge_index[1] 是超边索引
    node_ids = hyperedge_index[0]
    edge_ids = hyperedge_index[1]

    # 将边表示加到对应节点
    # 检查 edge_ids 是否为空，避免在空张量上进行索引
    if edge_ids.numel() > 0:
        edge_agg.index_add_(0, node_ids, edge_emb[edge_ids])

    # 统计每个节点的超边数量（用于平均）
    deg = torch.bincount(node_ids, minlength=N).clamp(min=1).unsqueeze(1)

    edge_mean = edge_agg / deg

    # 融合
    fused = alpha * node_emb + (1 - alpha) * edge_mean
    return torch.nn.functional.normalize(fused, dim=1)


def train(num_negs):
    # 新增：获取 node_component_index
    features, hyperedge_index, hyperedge_component_index, node_component_index = \
        data.features, data.hyperedge_index, data.hyperedge_component_index, data.node_component_index
    num_nodes, num_edges, num_components = \
        data.num_nodes, data.num_edges, data.num_components

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Hypergraph Augmentation
    hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_en_rate'])
    hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_en_rate'])
    hyperedge_component_index1 = drop_incidence(hyperedge_component_index, params['drop_incidence_ec_rate'])
    hyperedge_component_index2 = drop_incidence(hyperedge_component_index, params['drop_incidence_ec_rate'])
    node_component_index1 = drop_incidence(node_component_index, params['drop_incidence_nc_rate'])
    node_component_index2 = drop_incidence(node_component_index, params['drop_incidence_nc_rate'])
    
    x1 = drop_features(features, params['drop_feature_rate'])
    x2 = drop_features(features, params['drop_feature_rate'])


    # 确保将 node_component_index 传递给 model
    n1, e1, c1 = model(x1, hyperedge_index1, hyperedge_component_index1, node_component_index1, num_nodes, num_edges, num_components)
    n2, e2, c2 = model(x2, hyperedge_index2, hyperedge_component_index2, node_component_index2, num_nodes, num_edges, num_components)

    edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    edge_mask = edge_mask1 & edge_mask2


    n1, n2 = model.node_projection(n1), model.node_projection(n2)
    e1, e2 = model.edge_projection(e1), model.edge_projection(e2)
    c1, c2 = model.comp_projection(c1), model.comp_projection(c2)


    n1_fuse = fuse_node_with_edges(n1, e1, hyperedge_index1, alpha=params['alpha'])
    n2_fuse = fuse_node_with_edges(n2, e2, hyperedge_index2, alpha=params['alpha'])

    # 节点级损失
    loss_n_fuse=model.node_level_loss(n1_fuse, n2_fuse, params['tau_n'], batch_size=params['batch_size_1'], num_negs=num_negs)
    loss_e=model.group_level_loss(e1[edge_mask],e2[edge_mask],params['tau_g'], batch_size=params['batch_size_1'], num_negs=num_negs)
    # # 连通分量级损失
    loss_c = model.component_level_loss(c1, c2, params['tau_c'], batch_size=params['batch_size_1'], num_negs=num_negs)

    # 总损失
    loss = loss_n_fuse +params['w_g']*loss_e + params['w_c'] * loss_c
    loss.backward()
    optimizer.step()
    return loss.item()


def node_classification_eval(num_splits=10):
    model.eval()
    # 确保将 node_component_index 传递给 model
    n, _, _ = model(data.features, data.hyperedge_index, data.hyperedge_component_index, data.node_component_index, data.num_nodes, data.num_edges, data.num_components)

    if data.name == 'pubmed':
        lr = 0.005
        max_epoch = 300
    elif data.name == 'cora' or data.name == 'citeseer':
        lr = 0.005
        max_epoch = 100
    elif data.name == 'Mushroom':
        lr = 0.01
        max_epoch = 200
    else:
        lr = 0.01
        max_epoch = 100

    accs = []
    for i in range(0,num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
    return accs 


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CCAHCL.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                  'ModelNet40'])
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    params = yaml.safe_load(open('AAAI2026/config.yaml'))[args.dataset]
    print(params)

    # 确保 DatasetLoader 加载的数据包含 node_component_index
    # DatasetLoader().load() 会返回 BaseDataset 的子类实例，其中已经包含了 node_component_index
    data = DatasetLoader().load(args.dataset).to(args.device)

    accs = []
    for seed in range(args.num_seeds):
        fix_seed(seed)

        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['hid_dim'], args.dataset, params['dropout'], params['num_layers'])
        model = CCAHCL(encoder, params['proj_dim']).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss = train(num_negs=None)

        acc = node_classification_eval()
        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')
              
    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')
