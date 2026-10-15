import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax


class MLP(nn.Module):


    def __init__(self, dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class PairMLP(nn.Module):
    def __init__(self, dim):
        super(PairMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


class WarmUp(nn.Module):
    

    def __init__(self, in_dim, dim):
        super(WarmUp, self).__init__()
        self.weight = nn.Linear(in_dim, dim, bias=False)
        self.mlp = MLP(dim)

    def forward(self, features, adj):
        propagated = torch.matmul(adj, self.weight(features))
        return self.mlp(F.relu(propagated))


class GatedLocalAttention(nn.Module):
   
    def __init__(self, dim, heads):
        super(GatedLocalAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.proj = nn.Linear(dim, heads * dim, bias=False)
        self.eps1 = nn.Linear(dim * 2, 1)
        self.eps2 = nn.Linear(dim * 2, 1)

    def forward(self, hidden, adj, edge_index):
        propagated = torch.matmul(adj, hidden)
        subspaces = self.proj(propagated).view(-1, self.heads, self.dim)
        source, target = edge_index[0], edge_index[1]
        self_state = subspaces[target]
        neighbor_state = subspaces[source]
        pair = torch.cat([self_state, neighbor_state], dim=-1)
        correlation = F.relu(self.eps1(pair)).squeeze(-1)
        attention = torch.stack(
            [scatter_softmax(correlation[:, head], target, dim=0) for head in range(self.heads)],
            dim=1,
        ).unsqueeze(-1)
        gate = torch.sigmoid(self.eps2(pair))
        message = attention * gate * neighbor_state + (1.0 - gate) * self_state
        aggregated = scatter_add(message, target, dim=0, dim_size=hidden.size(0))
        return aggregated.mean(dim=1)


class GlobalResponseCalibration(nn.Module):
   

    def __init__(self, dim):
        super(GlobalResponseCalibration, self).__init__()
        self.lam = nn.Parameter(torch.zeros(()))
        self.eta = nn.Parameter(torch.zeros(()))
        self.mu = nn.Parameter(torch.zeros(dim))

    def forward(self, embedding):
        lam = F.softplus(self.lam) + 1e-6
        eta = F.softplus(self.eta) + 1e-6
        dimension = embedding.size(-1)
        squared = embedding.pow(2)
        numerator = lam * math.sqrt(float(dimension)) * squared
        denominator = lam * squared.sum(dim=-1, keepdim=True) + dimension * eta
        response = numerator / denominator
        return (self.mu * response + 1.0) * embedding


class GlobalSemanticAwareness(nn.Module):
   
    def __init__(self, dim):
        super(GlobalSemanticAwareness, self).__init__()
        self.attn_mlp = MLP(dim)
        self.ff1 = MLP(dim)
        self.ff2 = MLP(dim)

    def forward(self, embedding):
        scale = math.sqrt(float(embedding.size(-1)))
        scores = torch.matmul(embedding, embedding.transpose(0, 1)) / scale
        context = torch.matmul(torch.softmax(scores, dim=-1), embedding)
        semantic = self.attn_mlp(F.relu(context + embedding))
        return self.ff2(self.ff1(semantic) + embedding)


class Handshake(nn.Module):
   
    def __init__(self, dim, heads):
        super(Handshake, self).__init__()
        if dim % heads != 0:
            raise ValueError("embedding dim must be divisible by the number of heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.conv = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.gate = nn.Linear(dim, dim)

    def forward(self, local, global_state, residual):
        shared = torch.cat([local, global_state], dim=-1)
        shared = self.conv(F.relu(shared).unsqueeze(-1)).squeeze(-1)
        shared_heads = shared.view(-1, self.heads, self.head_dim)
        local_heads = local.view(-1, self.heads, self.head_dim)
        global_heads = global_state.view(-1, self.heads, self.head_dim)
        local_score = F.cosine_similarity(shared_heads, local_heads, dim=-1)
        global_score = F.cosine_similarity(shared_heads, global_heads, dim=-1)
        weights = torch.softmax(torch.stack([local_score, global_score], dim=-1), dim=-1)
        mixed = weights[..., 0:1] * local_heads + weights[..., 1:2] * global_heads
        mixed = mixed.reshape(local.size(0), -1)
        theta = torch.sigmoid(self.gate(mixed))
        return theta * mixed + (1.0 - theta) * residual


class GraphProp(nn.Module):
  
    def __init__(self, dim):
        super(GraphProp, self).__init__()
        self.weight = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden, adj):
        return F.relu(torch.matmul(adj, self.weight(hidden)))


class LSAToGSA(nn.Module):
    def __init__(self, dim, heads):
        super(LSAToGSA, self).__init__()
        self.local = GatedLocalAttention(dim, heads)
        self.grc = GlobalResponseCalibration(dim)
        self.global_block = GlobalSemanticAwareness(dim)
        self.handshake = Handshake(dim, heads)
        self.propagate = GraphProp(dim)

    def forward(self, hidden, adj, edge_index):
        local = self.local(hidden, adj, edge_index)
        calibrated = self.grc(local)
        semantic = self.global_block(calibrated)
        fused = self.handshake(local, semantic, residual=local)
        refined = self.propagate(fused, adj)
        return refined, fused, refined


class GSAToLSA(nn.Module):
    def __init__(self, dim, heads):
        super(GSAToLSA, self).__init__()
        self.global_block = GlobalSemanticAwareness(dim)
        self.grc = GlobalResponseCalibration(dim)
        self.pre_local = GraphProp(dim)
        self.local = GatedLocalAttention(dim, heads)
        self.handshake = Handshake(dim, heads)
        self.propagate = GraphProp(dim)

    def forward(self, hidden, adj, edge_index):
        semantic = self.global_block(hidden)
        calibrated = self.grc(semantic)
        prepared = self.pre_local(calibrated, adj)
        local = self.local(prepared, adj, edge_index)
        fused = self.handshake(local, semantic, residual=semantic)
        refined = self.propagate(fused, adj)
        return refined, fused, refined


class CrossAttentionFusion(nn.Module):
   
    def __init__(self, dim, blocks):
        super(CrossAttentionFusion, self).__init__()
        if dim % blocks != 0:
            raise ValueError("embedding dim must be divisible by the number of fusion blocks")
        self.blocks = blocks
        self.block_dim = dim // blocks
        self.block_weight = nn.Parameter(torch.ones(blocks))
        self.block_proj = nn.ModuleList([nn.Linear(self.block_dim, self.block_dim) for _ in range(blocks)])
        self.weight_g = nn.Parameter(torch.ones(dim))
        self.weight_l = nn.Parameter(torch.ones(dim))
        self.mix = nn.Linear(dim * 2, dim)

    def _subspace(self, global_state, local_state):
        global_blocks = global_state.split(self.block_dim, dim=-1)
        local_blocks = local_state.split(self.block_dim, dim=-1)
        global_parts = []
        local_parts = []
        scale = math.sqrt(float(self.block_dim))
        for index, (global_block, local_block) in enumerate(zip(global_blocks, local_blocks)):
            interaction = (global_block * local_block).sum(dim=-1, keepdim=True) / scale
            kappa = torch.sigmoid(self.block_weight[index] * interaction)
            global_parts.append(kappa * self.block_proj[index](global_block))
            local_parts.append(kappa * self.block_proj[index](local_block))
        return torch.cat(global_parts, dim=-1), torch.cat(local_parts, dim=-1)

    def forward(self, global_state, local_state):
        gamma_g, gamma_l = self._subspace(global_state, local_state)
        scale = math.sqrt(float(global_state.size(-1)))
        projected_g = (gamma_g * self.weight_g).sum(dim=-1)
        projected_l = (gamma_l * self.weight_l).sum(dim=-1)
        consistency = torch.sigmoid(projected_g * projected_l / scale).unsqueeze(-1)
        beta = torch.sigmoid(self.mix(torch.cat([global_state, local_state], dim=-1)) * consistency)
        return beta * (gamma_g * global_state) + (1.0 - beta) * (gamma_l * local_state)


class GNNBiHD(nn.Module):
    def __init__(self, in_dim, dim=128, heads=4, blocks=2, tau=0.5):
        super(GNNBiHD, self).__init__()
        self.tau = tau
        self.warm_local = WarmUp(in_dim, dim)
        self.warm_global = WarmUp(in_dim, dim)
        self.local_to_global = nn.ModuleList([LSAToGSA(dim, heads) for _ in range(blocks)])
        self.global_to_local = nn.ModuleList([GSAToLSA(dim, heads) for _ in range(blocks)])
        self.fusion = CrossAttentionFusion(dim, heads)
        self.pair_mlp = PairMLP(dim)

    def encode(self, features, adj, edge_index):
        local_hidden = self.warm_local(features, adj)
        local_pre = local_hidden
        local_out = local_hidden
        for block in self.local_to_global:
            local_out, local_pre, local_out = block(local_hidden, adj, edge_index)
            local_hidden = local_out

        global_hidden = self.warm_global(features, adj)
        global_pre = global_hidden
        global_out = global_hidden
        for block in self.global_to_local:
            global_out, global_pre, global_out = block(global_hidden, adj, edge_index)
            global_hidden = global_out

        embedding = self.fusion(global_out, local_out)
        return embedding, local_pre, local_out, global_pre, global_out

    def score_pairs(self, embedding, source, target):
        return self.pair_mlp(torch.cat([embedding[source], embedding[target]], dim=-1))

    def contrastive(self, anchor, other):
        anchor = F.normalize(anchor, dim=-1)
        other = F.normalize(other, dim=-1)
        sim_anchor = torch.matmul(anchor, anchor.transpose(0, 1)) / self.tau
        sim_other = torch.matmul(anchor, other.transpose(0, 1)) / self.tau
        positive = sim_other.diag()
        denominator = torch.logsumexp(torch.cat([sim_anchor, sim_other], dim=1), dim=1)
        return -(positive - denominator).mean()
