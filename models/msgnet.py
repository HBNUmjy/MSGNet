import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MultiScaleGraph(nn.Module):
    """Mixed global/dynamic graph with parallel 0-, 1-, and 2-hop propagation."""

    def __init__(self, in_channels, out_channels, node_num):
        super().__init__()
        self.node_num = node_num
        self.in_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.global_logits = nn.Parameter(torch.empty(node_num, node_num))
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        self.scale_0 = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.scale_1 = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.scale_2 = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.normal_(self.global_logits, mean=0.0, std=0.01)

    def _off_diagonal_mask(self, device, dtype):
        return 1.0 - torch.eye(self.node_num, device=device, dtype=dtype)

    def _global_adjacency(self):
        adjacency = torch.sigmoid(self.global_logits)
        adjacency = 0.5 * (adjacency + adjacency.transpose(0, 1))
        return adjacency * self._off_diagonal_mask(adjacency.device, adjacency.dtype)

    def _dynamic_adjacency(self, features):
        trajectories = features.mean(dim=1)
        trajectories = trajectories - trajectories.mean(dim=-1, keepdim=True)
        trajectories = F.normalize(trajectories, p=2, dim=-1, eps=1e-6)
        adjacency = F.relu(trajectories @ trajectories.transpose(1, 2))
        return adjacency * self._off_diagonal_mask(adjacency.device, adjacency.dtype)

    @staticmethod
    def _normalize_adjacency(adjacency):
        node_count = adjacency.size(-1)
        identity = torch.eye(
            node_count, device=adjacency.device, dtype=adjacency.dtype
        ).unsqueeze(0)
        adjacency = adjacency + identity
        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt_degree = degree.rsqrt()
        return (
            inv_sqrt_degree.unsqueeze(-1)
            * adjacency
            * inv_sqrt_degree.unsqueeze(-2)
        )

    @staticmethod
    def _propagate(features, adjacency):
        return torch.einsum("bcut,bvu->bcvt", features, adjacency)

    def forward(self, features):
        features = self.in_proj(features)
        global_adjacency = self._global_adjacency().unsqueeze(0)
        global_adjacency = global_adjacency.expand(features.size(0), -1, -1)
        dynamic_adjacency = self._dynamic_adjacency(features)
        mixing_weight = torch.sigmoid(self.mix_logit)
        adjacency = (
            mixing_weight * global_adjacency
            + (1.0 - mixing_weight) * dynamic_adjacency
        )
        adjacency = self._normalize_adjacency(adjacency)

        scale_0 = features * self.scale_0
        scale_1 = self._propagate(scale_0, adjacency) * self.scale_1
        scale_2 = self._propagate(scale_1, adjacency) * self.scale_2
        return self.fusion(torch.cat((scale_0, scale_1, scale_2), dim=1))


class GatedTemporalConv(nn.Module):
    """Depthwise temporal convolution followed by gating and pointwise projection."""

    def __init__(self, channels, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.norm = nn.BatchNorm2d(channels)
        self.depthwise = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, features):
        candidate, gate = self.depthwise(self.norm(features)).chunk(2, dim=1)
        return self.pointwise(torch.tanh(candidate) * torch.sigmoid(gate))


class STBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        node_num,
        temporal_kernel=9,
        dropout=0.2,
    ):
        super().__init__()
        self.graph = MultiScaleGraph(in_channels, out_channels, node_num)
        self.temporal = GatedTemporalConv(out_channels, temporal_kernel)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, features):
        residual = self.residual(features)
        features = self.graph(features)
        features = self.dropout(self.temporal(features))
        return F.relu(features + residual, inplace=True)


class HierarchicalAttentionPool(nn.Module):
    """Additive node attention followed by additive temporal attention."""

    def __init__(self, channels):
        super().__init__()
        attention_dim = max(channels // 4, 16)
        self.node_score = nn.Sequential(
            nn.Linear(channels, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.time_score = nn.Sequential(
            nn.Linear(channels, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, features):
        features = features.permute(0, 2, 3, 1).contiguous()
        node_weights = torch.softmax(self.node_score(features), dim=1)
        temporal_features = (features * node_weights).sum(dim=1)
        time_weights = torch.softmax(self.time_score(temporal_features), dim=1)
        return (temporal_features * time_weights).sum(dim=1)


class UserBiGRU(nn.Module):
    def __init__(self, input_dim=128, hidden_size=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, lengths):
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=features.size(1),
        )
        return self.dropout(self.norm(output))


class MaskedAttentionPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, features, lengths):
        scores = self.score(features).squeeze(-1)
        positions = torch.arange(features.size(1), device=features.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        scores = scores - scores.max(dim=1, keepdim=True).values
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (features * weights).sum(dim=1)


class MSGNet(nn.Module):
    """MSGNet for longitudinal user-level depression-risk classification."""

    def __init__(
        self,
        input_dim=161,
        graph_nodes=16,
        hidden_dim=128,
        user_hidden_size=64,
        dropout=0.2,
    ):
        super().__init__()
        if hidden_dim != 2 * user_hidden_size:
            raise ValueError("hidden_dim must equal twice user_hidden_size")

        self.input_dim = input_dim
        self.graph_nodes = graph_nodes
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, graph_nodes)
        self.video_encoder = nn.Sequential(
            STBlock(1, 32, graph_nodes, dropout=dropout),
            STBlock(32, 64, graph_nodes, dropout=dropout),
            STBlock(64, hidden_dim, graph_nodes, dropout=dropout),
        )
        self.video_pool = HierarchicalAttentionPool(hidden_dim)
        self.video_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.user_encoder = UserBiGRU(
            input_dim=hidden_dim,
            hidden_size=user_hidden_size,
            dropout=dropout,
        )
        self.user_pool = MaskedAttentionPool(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        if isinstance(inputs, (tuple, list)):
            features = inputs[0]
            lengths = inputs[1] if len(inputs) > 1 else None
        else:
            features = inputs
            lengths = None

        features = features.float()
        if features.dim() == 3:
            features = features.unsqueeze(1)
        if features.dim() != 4:
            raise ValueError("expected features with shape [B, N, T, F]")

        batch_size, video_count, time_steps, feature_dim = features.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} frame features, received {feature_dim}"
            )
        if lengths is None:
            lengths = torch.full(
                (batch_size,), video_count, device=features.device, dtype=torch.long
            )
        elif not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=features.device, dtype=torch.long)
        else:
            lengths = lengths.to(features.device, dtype=torch.long)
        lengths = lengths.clamp(min=1, max=video_count)

        features = self.input_projection(features)
        features = features.reshape(
            batch_size * video_count, time_steps, self.graph_nodes
        )
        features = features.permute(0, 2, 1).unsqueeze(1)
        video_features = self.video_encoder(features)
        video_features = self.video_projection(self.video_pool(video_features))
        video_features = video_features.reshape(batch_size, video_count, self.hidden_dim)

        positions = torch.arange(video_count, device=features.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)
        video_features = video_features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        user_features = self.user_encoder(video_features, lengths)
        user_representation = self.user_pool(user_features, lengths)
        return self.classifier(user_representation)
