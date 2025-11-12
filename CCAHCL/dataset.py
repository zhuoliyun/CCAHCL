from typing import Optional
import os.path as osp
import pickle

import torch
from torch.utils.data import random_split
import networkx as nx # Import networkx for connected components calculation


class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = 'cpu'):
        self.type = type
        self.name = name
        self.device = device
        if self.type in ['cocitation', 'coauthorship']:
            self.dataset_dir = osp.join('AAAI2026/dataset', self.type, self.name)
        else:
            self.dataset_dir = osp.join('AAAI2026/dataset', self.name)
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        self.features = None
        self.hypergraph = None
        self.labels = None
        self.processed_hypergraph = None
        self.hyperedge_index = None
        self.num_nodes = None
        self.num_edges = None
        self.edge_to_num = None
        self.num_to_edge = None
        self.num_components = None
        self.hyperedge_component_index = None
        self.node_component_index = None # 新增：节点-连通分量索引
        self.lambda_e_tensor = None # Initialize lambda_e_tensor

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)
        return splits

    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])

        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1 if self.hyperedge_index.numel() > 0 else 0
        self.num_edges = int(self.hyperedge_index[1].max()) + 1 if self.hyperedge_index.numel() > 0 else 0
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        hyperedge_adj_list = {i: set() for i in range(self.num_edges)}
        node_to_hyperedges = {i: [] for i in range(self.num_nodes)}

        for node_idx, edge_idx in incidence_matrix:
            node_to_hyperedges[node_idx].append(edge_idx)

        for node_idx in node_to_hyperedges:
            edges_sharing_node = node_to_hyperedges[node_idx]
            for i in range(len(edges_sharing_node)):
                for j in range(i + 1, len(edges_sharing_node)):
                    e1 = edges_sharing_node[i]
                    e2 = edges_sharing_node[j]
                    hyperedge_adj_list[e1].add(e2)
                    hyperedge_adj_list[e2].add(e1)

        visited_hyperedges = set()
        component_mapping = {} # 映射超边到连通分量ID
        current_component_id = 0

        for i in range(self.num_edges):
            if i not in visited_hyperedges:
                stack = [i]
                visited_hyperedges.add(i)
                while stack:
                    current_edge = stack.pop()
                    component_mapping[current_edge] = current_component_id
                    for neighbor_edge in hyperedge_adj_list[current_edge]:
                        if neighbor_edge not in visited_hyperedges:
                            visited_hyperedges.add(neighbor_edge)
                            stack.append(neighbor_edge)
                current_component_id += 1

        self.num_components = current_component_id

        # Create hyperedge-component incidence matrix
        hyperedge_component_incidence = []
        for edge_num, comp_id in component_mapping.items():
            hyperedge_component_incidence.append([edge_num, comp_id])
        self.hyperedge_component_index = torch.LongTensor(hyperedge_component_incidence).T.contiguous()

        # --- 新增：创建节点-连通分量索引 (node_component_index) ---
        node_component_incidence=[]
        
        # 遍历所有节点
        for node_id in range(self.num_nodes):
            # 获取该节点所属的所有超边
            connected_hyperedges = node_to_hyperedges.get(node_id, [])
            
            # 收集该节点所属的所有连通分量ID（去重）
            node_connected_components = set()
            for he_id in connected_hyperedges:
                if he_id in component_mapping:
                    node_connected_components.add(component_mapping[he_id])
            
            for comp_id in node_connected_components:
                node_component_incidence.append([node_id, comp_id])
        

        self.node_component_index = torch.LongTensor(node_component_incidence).T.contiguous()
        # --- 结束新增 ---

        self.full_lambda_e_tensor=torch.full((self.num_edges + self.num_nodes, 1), 0.01, dtype=torch.float32, device=self.device)
        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.hyperedge_component_index = self.hyperedge_component_index.to(device)
        self.node_component_index = self.node_component_index.to(device) # 新增：移动到设备
        self.labels = self.labels.to(device)
        self.full_lambda_e_tensor = self.full_lambda_e_tensor.to(device) # Move to device
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: Optional[int] = None, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test),
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        return [train_mask, val_mask, test_mask]


class CoraCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'cora', **kwargs)


class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'citeseer', **kwargs)


class PubmedCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'pubmed', **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'cora', **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'dblp', **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'ModelNet40', **kwargs)
