# layers.py

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros


class ProposedConv(MessagePassing):
    _cached_norm_n2e: Optional[Tensor]
    _cached_norm_e2n: Optional[Tensor]
    _cached_norm_e2c: Optional[Tensor]
    _cached_norm_c2e: Optional[Tensor]
    _cached_norm_n2c: Optional[Tensor] # 新增：Cache for node-to-component norm

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, comp_dim: int,
                 dataset, dropout: float = 0.0, act: Callable = nn.PReLU(), bias: bool = True,
                 cached: bool = False, row_norm: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.comp_dim = comp_dim
        self.dropout = dropout
        self.act = act
        self.cached = cached
        self.row_norm = row_norm
        self.dataset = dataset

        # Linear layers for transformations
        self.lin_n2e = Linear(in_dim, hid_dim, bias=False, weight_initializer='glorot') # Node to Edge
        self.lin_e2n = Linear(hid_dim, out_dim, bias=False, weight_initializer='glorot') # Edge to Node
        self.lin_e2c = Linear(hid_dim, comp_dim, bias=False, weight_initializer='glorot') # Edge to Component
        self.lin_c2e = Linear(comp_dim, hid_dim, bias=False, weight_initializer='glorot') # Component to Edge
        self.lin_n2c = Linear(in_dim, comp_dim, bias=False, weight_initializer='glorot') # 新增：Node to Component

        # Biases
        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
            self.bias_e2c = Parameter(torch.Tensor(comp_dim))
            self.bias_c2e = Parameter(torch.Tensor(hid_dim))
            self.bias_n2c = Parameter(torch.Tensor(comp_dim)) # 新增：Node to Component bias
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)
            self.register_parameter('bias_e2c', None)
            self.register_parameter('bias_c2e', None)
            self.register_parameter('bias_n2c', None) # 新增：Node to Component bias

        # 新增：用于融合 E2C 和 N2C 聚合结果的权重
        self.lambda_c = Parameter(torch.Tensor([1])) # 可学习的权重

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        self.lin_e2c.reset_parameters()
        self.lin_c2e.reset_parameters()
        self.lin_n2c.reset_parameters() # Reset new linear layer
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        zeros(self.bias_e2c)
        zeros(self.bias_c2e)
        zeros(self.bias_n2c) # Reset new bias
        # torch.nn.init.constant_(self.lambda_c, 0.5) # 初始化 lambda_c 为 0.5

        self._cached_norm_n2e = None
        self._cached_norm_e2n = None
        self._cached_norm_e2c = None
        self._cached_norm_c2e = None
        self._cached_norm_n2c = None # Reset new cache

    def forward(self, x: Tensor, hyperedge_index: Tensor, hyperedge_component_index: Tensor,
                node_component_index: Tensor, # 新增：传入节点-连通分量索引
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, num_components: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        # Handle case where num_components might be 0 if hyperedge_component_index is empty
        if num_components is None:
            if hyperedge_component_index.numel() > 0:
                num_components = int(hyperedge_component_index[1].max()) + 1
            elif node_component_index.numel() > 0: # 如果 hyperedge_component_index 为空，但 node_component_index 不为空
                num_components = int(node_component_index[1].max()) + 1
            else:
                num_components = 0 # No components if no connections


        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n
        cache_norm_e2c = self._cached_norm_e2c
        cache_norm_c2e = self._cached_norm_c2e
        cache_norm_n2c = self._cached_norm_n2c # Get new cache

        if (cache_norm_n2e is None) or (cache_norm_e2n is None) or \
           (cache_norm_e2c is None) or (cache_norm_c2e is None) or \
           (cache_norm_n2c is None): # Added N2C to condition

            hyperedge_weight = x.new_ones(num_edges)

            # --- Node-Edge Normalization ---
            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)

            # --- Edge-Component Normalization (and Component-Edge) ---
            edge_comp_idx, comp_idx_e2c = hyperedge_component_index # 区分索引变量名
            
            # Degrees for Edge -> Component (E2C)
            Dc_e2c = scatter_add(x.new_ones(hyperedge_component_index.shape[1]),
                                 hyperedge_component_index[1], dim=0, dim_size=num_components)
            # De_e2c = scatter_add(x.new_ones(hyperedge_component_index.shape[1]),
            #                      hyperedge_component_index[0], dim=0, dim_size=num_edges) # Denominator for symmetric for edges

            # Degrees for Component -> Edge (C2E)
            Dc_c2e = scatter_add(x.new_ones(hyperedge_component_index.shape[1]),
                                 hyperedge_component_index[1], dim=0, dim_size=num_components)
            # De_c2e = scatter_add(x.new_ones(hyperedge_component_index.shape[1]),
            #                      hyperedge_component_index[0], dim=0, dim_size=num_edges)

            # --- 新增：Node-Component Normalization ---
            node_comp_idx, comp_idx_n2c = node_component_index # 区分索引变量名

            # Degrees for Node -> Component (N2C)
            Dc_n2c = scatter_add(x.new_ones(node_component_index.shape[1]),
                                 node_component_index[1], dim=0, dim_size=num_components)
            Dn_n2c = scatter_add(x.new_ones(node_component_index.shape[1]),
                                 node_component_index[0], dim=0, dim_size=num_nodes)


            # Handle potential division by zero for degrees
            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                Dc_e2c_inv = 1.0 / Dc_e2c
                Dc_e2c_inv[Dc_e2c_inv == float('inf')] = 0

                Dc_c2e_inv = 1.0 / Dc_c2e
                Dc_c2e_inv[Dc_c2e_inv == float('inf')] = 0

                # 新增：N2C 归一化项
                Dc_n2c_inv = 1.0 / Dc_n2c
                Dc_n2c_inv[Dc_n2c_inv == float('inf')] = 0
                
                # Dn_n2c_inv = 1.0 / Dn_n2c # 如果需要对源节点进行归一化
                # Dn_n2c_inv[Dn_n2c_inv == float('inf')] = 0


                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

                norm_e2c = Dc_e2c_inv[comp_idx_e2c] # E2C: target component degree

                # C2E: target edge degree
                if hyperedge_component_index.numel() > 0:
                    norm_c2e = De_inv[edge_comp_idx] # Use De_inv, as target is edge
                else: # No connections for C2E
                    norm_c2e = x.new_zeros(0) # Empty tensor if no connections
                
                # 新增：N2C 归一化项
                if node_component_index.numel() > 0:
                    norm_n2c = Dc_n2c_inv[comp_idx_n2c] # N2C: target component degree
                else:
                    norm_n2c = x.new_zeros(0)


            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
                self._cached_norm_e2c = norm_e2c
                self._cached_norm_c2e = norm_c2e
                self._cached_norm_n2c = norm_n2c # Cache new norm
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n
            norm_e2c = cache_norm_e2c
            norm_c2e = cache_norm_c2e
            norm_n2c = cache_norm_n2c # Get new norm from cache


        # 1. Node to Edge (N2E): Compute initial hyperedge features
        x_n2e = self.lin_n2e(x)
        e_initial = self.propagate(hyperedge_index, x=x_n2e, norm=norm_n2e,
                               size=(num_nodes, num_edges))

        if self.bias_n2e is not None:
            e_initial = e_initial + self.bias_n2e
        e_initial = self.act(e_initial) # Activation after N2E
        e_initial = F.dropout(e_initial, p=self.dropout, training=self.training)

        # 2. Edge to Component (E2C): Compute component features from edges
        x_e2c = self.lin_e2c(e_initial) # Use the initial hyperedge features
        c_from_e = self.propagate(hyperedge_component_index, x=x_e2c, norm=norm_e2c,
                           size=(num_edges, num_components))

        if self.bias_e2c is not None:
            c_from_e = c_from_e + self.bias_e2c
        # c_from_e = self.act(c_from_e) # 激活函数和Dropout可以放在融合后
        # c_from_e = F.dropout(c_from_e, p=self.dropout, training=self.training)

        # 2.5. 新增：Node to Component (N2C): Compute component features from nodes
        x_n2c = self.lin_n2c(x) # 使用原始节点特征
        c_from_n = self.propagate(node_component_index, x=x_n2c, norm=norm_n2c,
                                  size=(num_nodes, num_components))
        
        if self.bias_n2c is not None:
            c_from_n = c_from_n + self.bias_n2c
        alpha_c = self.lambda_c
        if num_components > 0:
            # 确保维度匹配，如果某个聚合结果为空（例如，没有超边或没有节点连接到分量），则用零张量填充
            if c_from_e.numel() == 0 and c_from_n.numel() == 0:
                c = torch.zeros(num_components, self.comp_dim, device=x.device)
            elif c_from_e.numel() == 0:
                c = c_from_n # 如果没有超边到分量的聚合，只使用节点到分量的聚合
            elif c_from_n.numel() == 0:
                c = c_from_e # 如果没有节点到分量的聚合，只使用超边到分量的聚合
            else:
                c = alpha_c * c_from_e + (1 - alpha_c) * c_from_n
        else:
            c = torch.zeros(0, self.comp_dim, device=x.device) # 如果没有连通分量，则为零张量


        c = self.act(c) # 融合后应用激活函数
        c = F.dropout(c, p=self.dropout, training=self.training)


        # 3. Component to Edge (C2E) & Fusion:
        # Pass information from components back to edges
        x_c2e = self.lin_c2e(c) # Use the component features 'c'

        # Propagate from components to edges.
        # Need to flip hyperedge_component_index as flow is now component -> edge
        if hyperedge_component_index.numel() > 0 and num_components > 0: # Only propagate if connections and components exist
            e_from_c = self.propagate(hyperedge_component_index.flip([0]), x=x_c2e, norm=norm_c2e,
                                    size=(num_components, num_edges))
        else: # No component-edge connections, e_from_c is zeros
            e_from_c = torch.zeros(num_edges, self.hid_dim, device=x.device)


        if self.bias_c2e is not None:
            e_from_c = e_from_c + self.bias_c2e
        e_from_c = self.act(e_from_c) # 可以在这里加激活函数和dropout
        e_from_c = F.dropout(e_from_c, p=self.dropout, training=self.training)

        # Fuse initial hyperedge features with features from components (summation)

        if self.dataset=='cora':
            e_final =  (0.5*e_initial + 1.5* e_from_c)# Summation as discussed  0.1 2*( e_initial + e_from_c)
        elif self.dataset=='citeseer':
            e_final =  (0.25*e_initial + 0.01* e_from_c)
        elif self.dataset=='cora_coauthor':
            e_final =  (0.7*e_initial + 0.3* e_from_c)
        elif self.dataset=='pubmed':
            e_final =  (2*e_initial + 1.5* e_from_c)
        elif self.dataset=='ModelNet40':
            e_final =  (1*e_initial + 0.5* e_from_c)
        elif self.dataset=='dblp_coauthor':
            e_final =  (1*e_initial + 1.5* e_from_c)
        # Apply activation and dropout after fusion
        # e_final = self.act(e_final)
        # e_final = F.dropout(e_final, p=self.dropout, training=self.training)


        # 4. Edge to Node (E2N): Use the final hyperedge features to update nodes
        x_e2n = self.lin_e2n(e_final) # Use the fused hyperedge features (e_final)
        n = self.propagate(hyperedge_index.flip([0]), x=x_e2n, norm=norm_e2n,
                               size=(num_edges, num_nodes))

        if self.bias_e2n is not None:
            n = n + self.bias_e2n
        # Node features (n) are typically the final output for classification/regression,
        # so activation/dropout might be applied in the subsequent layer or output head.
        # If this is the last layer, you might add them here. For now, following typical GNN layer structure.


        return n, e_final, c # Return node, final hyperedge, and component embeddings

    def message(self, x_j: Tensor, norm: Tensor):
        # This message function is generic for all propagate calls
        return norm.view(-1, 1) * x_j
