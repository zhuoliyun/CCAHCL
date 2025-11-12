from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add

from CCAHCL.layers import ProposedConv

class HyperEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, node_dim, comp_dim, dataset, dropout=0.0, num_layers=2, act: Callable = nn.PReLU()): # 保持 comp_dim
        super(HyperEncoder, self).__init__()
        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.comp_dim = comp_dim # 保持
        self.dropout=dropout
        self.num_layers = num_layers
        self.act = act
        self.dataset=dataset

        self.convs = nn.ModuleList()
        if num_layers == 1:
            # 确保 ProposedConv 的初始化参数与 layers.py 中的定义一致
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, self.comp_dim, self.dataset, self.dropout, cached=False, act=act))
        else:
            self.convs.append(ProposedConv(self.in_dim, self.edge_dim, self.node_dim, self.comp_dim, self.dataset, cached=False, act=act))
            for _ in range(self.num_layers - 2):
                self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, self.comp_dim, self.dataset, cached=False, act=act))
            self.convs.append(ProposedConv(self.node_dim, self.edge_dim, self.node_dim, self.comp_dim, self.dataset, cached=False, act=act))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor, hyperedge_component_index: Tensor,
                node_component_index: Tensor, # 新增：传入节点-连通分量索引
                num_nodes: int, num_edges: int, num_components: int):
        
        n_out, e_out, c_out = None, None, None
        for i in range(self.num_layers):
            # 确保将 node_component_index 传递给 ProposedConv
            n_out, e_out, c_out = self.convs[i](x, hyperedge_index, hyperedge_component_index,
                                                node_component_index, # 传递 node_component_index
                                                num_nodes, num_edges, num_components)
            x = self.act(n_out)
        return x, e_out, c_out


class CCAHCL(nn.Module):
    def __init__(self, encoder: HyperEncoder, proj_dim: int):
        super(CCAHCL, self).__init__()
        self.encoder = encoder

        self.node_dim = self.encoder.node_dim
        self.edge_dim = self.encoder.edge_dim
        self.comp_dim = self.encoder.comp_dim

        self.fc1_n = nn.Linear(self.node_dim, proj_dim)
        self.fc2_n = nn.Linear(proj_dim, self.node_dim)
        self.fc1_e = nn.Linear(self.edge_dim, proj_dim)
        self.fc2_e = nn.Linear(proj_dim, self.edge_dim)
        self.fc1_c = nn.Linear(self.comp_dim, proj_dim)
        self.fc2_c = nn.Linear(proj_dim, self.comp_dim)

        self.reset_parameters()
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.fc1_n.reset_parameters()
        self.fc2_n.reset_parameters()
        self.fc1_e.reset_parameters()
        self.fc2_e.reset_parameters()
        self.fc1_c.reset_parameters()
        self.fc2_c.reset_parameters()
        
    def forward(self, x: Tensor, hyperedge_index: Tensor, hyperedge_component_index: Tensor, node_component_index:Tensor, # 新增：传入 node_component_index
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None, num_components: Optional[int] = None):
        if num_nodes is None:
            num_nodes = int(hyperedge_index[0].max()) + 1
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1
        # 确保 num_components 的计算逻辑与 ProposedConv 保持一致
        if num_components is None:
            if hyperedge_component_index.numel() > 0:
                num_components = int(hyperedge_component_index[1].max()) + 1
            elif node_component_index.numel() > 0:
                num_components = int(node_component_index[1].max()) + 1
            else:
                num_components = 0

        node_idx = torch.arange(0, num_nodes, device=x.device)
        edge_idx = torch.arange(num_edges, num_edges + num_nodes, device=x.device)
        self_loop = torch.stack([node_idx, edge_idx])
        self_loop_hyperedge_index = torch.cat([hyperedge_index, self_loop], 1)

        # 确保将 node_component_index 传递给 encoder
        n, e, c = self.encoder(x, self_loop_hyperedge_index, hyperedge_component_index, node_component_index,
                               num_nodes, num_edges + num_nodes, num_components)

        return n, e[:num_edges], c


    def f(self, x, tau):
        return torch.exp(x / tau)
    
    def node_projection(self, z: Tensor):
        return self.fc2_n(F.elu(self.fc1_n(z)))
    
    def edge_projection(self, z: Tensor):
        return self.fc2_e(F.elu(self.fc1_e(z)))
    
    def comp_projection(self, z: Tensor):
        return self.fc2_c(F.elu(self.fc1_c(z)))
    
    def cosine_similarity(self, z1: Tensor, z2: Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def __semi_loss(self, h1: Tensor, h2: Tensor, tau: float, num_negs: Optional[int]):
        # 此方法保持不变，因为它被多种损失函数调用
        if num_negs is None:
            between_sim = self.f(self.cosine_similarity(h1, h2), tau)
            return -torch.log(between_sim.diag() / between_sim.sum(1))
        else:
            pos_sim = self.f(F.cosine_similarity(h1, h2), tau)
            negs = []
            for _ in range(num_negs):
                negs.append(h2[torch.randperm(h2.size(0))])
            negs = torch.stack(negs, dim=-1)
            neg_sim = self.f(F.cosine_similarity(h1.unsqueeze(-1).tile(num_negs), negs), tau)
            return -torch.log(pos_sim / (pos_sim + neg_sim.sum(1)))
        
    def __semi_loss_batch(self, h1: Tensor, h2: Tensor, tau: float, batch_size: int):
        # 此方法保持不变
        device = h1.device
        num_samples = h1.size(0)
        num_batches = (num_samples - 1) // batch_size + 1
        indices = torch.arange(0, num_samples, device=device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size: (i + 1) * batch_size]
            between_sim = self.f(self.cosine_similarity(h1[mask], h2), tau)

            loss = -torch.log(between_sim[:, i * batch_size: (i + 1) * batch_size].diag() / between_sim.sum(1))
            losses.append(loss)
        return torch.cat(losses)

    def __loss(self, z1: Tensor, z2: Tensor, tau: float, batch_size: Optional[int], 
               num_negs: Optional[int], mean: bool):
        # 此方法保持不变
        if batch_size is None or num_negs is not None:
            l1 = self.__semi_loss(z1, z2, tau, num_negs)
            l2 = self.__semi_loss(z2, z1, tau, num_negs)
        else:
            l1 = self.__semi_loss_batch(z1, z2, tau, batch_size)
            l2 = self.__semi_loss_batch(z2, z1, tau, batch_size)

        loss = (l1 + l2) * 0.5
        loss = loss.mean() if mean else loss.sum()
        return loss

    # 原始的节点级对比损失，只和自己做正样本
    def node_level_loss(self, n1: Tensor, n2: Tensor, node_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(n1, n2, node_tau, batch_size, num_negs, mean)
        return loss

    # 2. Hyperedge-level contrastive loss (unchanged)
    def group_level_loss(self, e1: Tensor, e2: Tensor, edge_tau: float, 
                       batch_size: Optional[int] = None, num_negs: Optional[int] = None, 
                       mean: bool = True):
        loss = self.__loss(e1, e2, edge_tau, batch_size, num_negs, mean)
        return loss

    # 4. Connected Component Level Contrastive Loss (保持不变)
    def component_level_loss(self, c1: Tensor, c2: Tensor, comp_tau: float,
                             batch_size: Optional[int] = None, num_negs: Optional[int] = None,
                             mean: bool = True):
        loss = self.__loss(c1, c2, comp_tau, batch_size, num_negs, mean)
        return loss
    
   
