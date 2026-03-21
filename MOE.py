import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, speaker_emb=0):
        return x + self.pe[:, : x.size(1)] + speaker_emb


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, mlp_ratio=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, mask_modality, mask=None):
        B, seq_len, C = x.shape

        q = self.q(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, seq_len, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q * self.scale).float() @ k.float().transpose(-2, -1)

        if mask is not None:
            mask = mask.bool()
            mask_dict = {
                "a": mask[:, :seq_len],
                "t": mask[:, seq_len : 2 * seq_len],
                "v": mask[:, 2 * seq_len : 3 * seq_len],
            }
            current_mask = mask_dict[mask_modality]
            attn = attn.masked_fill(~current_mask[:, None, None, :], float("-inf"))
            attn = self.attn_drop(attn.softmax(dim=-1).type_as(x))
            attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        else:
            attn = self.attn_drop(attn.softmax(dim=-1).type_as(x))

        x_out = (attn @ v).transpose(1, 2).reshape(B, seq_len, C)
        return x_out + self.mlp(x_out)


class BlockSoftMoE(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, mlp_ratio=1.0):
        super().__init__()
        self.transformer_a = Attention(dim, num_heads, attn_drop, proj_drop, mlp_ratio)
        self.transformer_t = Attention(dim, num_heads, attn_drop, proj_drop, mlp_ratio)
        self.transformer_v = Attention(dim, num_heads, attn_drop, proj_drop, mlp_ratio)

    def forward(self, x, cross_modality, mask_modality, mask=None):
        if cross_modality == "a":
            return self.transformer_a(x, mask_modality, mask)
        if cross_modality == "t":
            return self.transformer_t(x, mask_modality, mask)
        if cross_modality == "v":
            return self.transformer_v(x, mask_modality, mask)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        depth=4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BlockSoftMoE(dim, num_heads, attn_drop, drop, mlp_ratio)
                for _ in range(depth)
            ]
        )

    def forward(self, x, mask=None, modality=None):
        x_cross_a = x.clone()
        x_cross_t = x.clone()
        x_cross_v = x.clone()

        for block in self.blocks:
            x_cross_a = x_cross_a + block(
                x_cross_a, cross_modality="a", mask_modality=modality, mask=mask
            )
            x_cross_t = x_cross_t + block(
                x_cross_t, cross_modality="t", mask_modality=modality, mask=mask
            )
            x_cross_v = x_cross_v + block(
                x_cross_v, cross_modality="v", mask_modality=modality, mask=mask
            )

        return torch.cat([x_cross_a, x_cross_t, x_cross_v], dim=-1)


class FrequencySplitter(nn.Module):
    def __init__(self, cutoff_ratio=0.25):
        super().__init__()
        self.cutoff_ratio = cutoff_ratio

    def forward(self, x):
        B, S, D = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        freq_len = x_freq.shape[1]
        cutoff_idx = max(1, int(freq_len * self.cutoff_ratio))

        mask_low = torch.zeros_like(x_freq)
        mask_low[:, :cutoff_idx, :] = 1.0

        mask_high = torch.zeros_like(x_freq)
        mask_high[:, cutoff_idx:, :] = 1.0

        x_low = torch.fft.irfft(x_freq * mask_low, n=S, dim=1)
        x_high = torch.fft.irfft(x_freq * mask_high, n=S, dim=1)
        return x_low, x_high


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        utters = torch.cat([a.unsqueeze(-2), b.unsqueeze(-2), c.unsqueeze(-2)], dim=-2)
        utters_fc = torch.cat(
            [
                self.fc(a).unsqueeze(-2),
                self.fc(b).unsqueeze(-2),
                self.fc(c).unsqueeze(-2),
            ],
            dim=-2,
        )
        weights = self.softmax(utters_fc)
        return torch.sum(weights * utters, dim=-2)


class MoMKE(nn.Module):
    def __init__(
        self,
        args,
        adim,
        tdim,
        vdim,
        D_e,
        n_classes=4,
        depth=4,
        num_heads=8,
        mlp_ratio=1,
        drop_rate=0,
        attn_drop_rate=0,
        no_cuda=False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.D_e = D_e
        self.adim, self.tdim, self.vdim = adim, tdim, vdim

        dropout_val = getattr(args, "dropout", 0.1) if hasattr(args, "dropout") else 0.1
        self.out_dropout = dropout_val

        self.a_in_proj = nn.Linear(self.adim, D_e)
        self.t_in_proj = nn.Linear(self.tdim, D_e)
        self.v_in_proj = nn.Linear(self.vdim, D_e)

        self.ln_a = nn.LayerNorm(D_e)
        self.ln_t = nn.LayerNorm(D_e)
        self.ln_v = nn.LayerNorm(D_e)

        self.consensus_proj = nn.Linear(D_e, D_e)

        self.dropout_a = nn.Dropout(dropout_val)
        self.dropout_t = nn.Dropout(dropout_val)
        self.dropout_v = nn.Dropout(dropout_val)

        self.pos_emb = PositionalEncoding(D_e)
        self.freq_splitter = FrequencySplitter(cutoff_ratio=0.2)

        self.block = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )

        router_input_dim = 5 * D_e
        self.router_a = Mlp(
            in_features=router_input_dim,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_t = Mlp(
            in_features=router_input_dim,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )
        self.router_v = Mlp(
            in_features=router_input_dim,
            hidden_features=int(D_e * mlp_ratio),
            out_features=3,
            drop=drop_rate,
        )

        self.conflict_gate = nn.Sequential(
            nn.Linear(2 * D_e, D_e // 2),
            nn.ReLU(),
            nn.Linear(D_e // 2, 1),
            nn.Sigmoid(),
        )

        D_joint = 3 * D_e
        self.proj1 = nn.Linear(D_joint, D_joint)
        self.consensus_fusion = Multimodal_GatedFusion(D_e)

    def forward(self, inputfeats, umask, spk_embeddings):
        weight_save = []

        text = inputfeats[:, :, : self.adim]
        audio = inputfeats[:, :, self.adim : self.adim + self.tdim]
        video = inputfeats[:, :, self.adim + self.tdim :]

        B, seq_len, _ = audio.shape

        if umask is None:
            umask = torch.ones(B, seq_len, device=inputfeats.device)

        attn_mask = torch.cat([umask, umask, umask], dim=1)
        mask_float = umask.unsqueeze(-1)

        proj_a = self.pos_emb(self.ln_a(self.a_in_proj(audio)), spk_embeddings) * mask_float
        proj_t = self.pos_emb(self.ln_t(self.t_in_proj(text)), spk_embeddings) * mask_float
        proj_v = self.pos_emb(self.ln_v(self.v_in_proj(video)), spk_embeddings) * mask_float

        low_a, high_a = self.freq_splitter(proj_a)
        low_t, high_t = self.freq_splitter(proj_t)
        low_v, high_v = self.freq_splitter(proj_v)

        consensus_feat = self.consensus_fusion(low_t, low_v, low_a) * mask_float

        diff_a = (high_a + (low_a - consensus_feat)) * mask_float
        diff_t = (high_t + (low_t - consensus_feat)) * mask_float
        diff_v = (high_v + (low_v - consensus_feat)) * mask_float

        stacked_feats = torch.stack([proj_a, proj_t, proj_v], dim=2)
        modality_variance = torch.var(stacked_feats, dim=2, unbiased=False)

        ortho_loss = torch.tensor(0.0, device=inputfeats.device)
        if self.training:
            mask_bool = umask.bool()

            def masked_ortho_loss(feat_diff, feat_common, m):
                sim = torch.abs(F.cosine_similarity(feat_diff, feat_common, dim=-1))
                return sim[m].mean() if m.sum() > 0 else torch.tensor(0.0, device=feat_diff.device)

            sim_a = masked_ortho_loss(diff_a, consensus_feat, mask_bool)
            sim_t = masked_ortho_loss(diff_t, consensus_feat, mask_bool)
            sim_v = masked_ortho_loss(diff_v, consensus_feat, mask_bool)
            ortho_loss = (sim_a + sim_t + sim_v) / 3

        global_context = torch.cat(
            [consensus_feat, modality_variance, diff_a, diff_t, diff_v], dim=-1
        )

        weight_a = torch.softmax(self.router_a(global_context), dim=-1)
        weight_t = torch.softmax(self.router_t(global_context), dim=-1)
        weight_v = torch.softmax(self.router_v(global_context), dim=-1)

        if not self.training:
            weight_save.append(
                np.array(
                    [
                        weight_a.detach().cpu().numpy(),
                        weight_t.detach().cpu().numpy(),
                        weight_v.detach().cpu().numpy(),
                    ]
                )
            )

        w_a = weight_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        w_t = weight_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
        w_v = weight_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

        diff_a_drop = self.dropout_a(diff_a)
        diff_t_drop = self.dropout_t(diff_t)
        diff_v_drop = self.dropout_v(diff_v)

        x_a_experts = self.block(diff_a_drop, mask=attn_mask, modality="a")
        x_t_experts = self.block(diff_t_drop, mask=attn_mask, modality="t")
        x_v_experts = self.block(diff_v_drop, mask=attn_mask, modality="v")

        x_a_unweighted = x_a_experts.reshape(B, seq_len, 3, self.D_e)
        x_t_unweighted = x_t_experts.reshape(B, seq_len, 3, self.D_e)
        x_v_unweighted = x_v_experts.reshape(B, seq_len, 3, self.D_e)

        diff_out_a = torch.sum(w_a * x_a_unweighted, dim=2)
        diff_out_t = torch.sum(w_t * x_t_unweighted, dim=2)
        diff_out_v = torch.sum(w_v * x_v_unweighted, dim=2)

        gate_input = torch.cat([consensus_feat, modality_variance], dim=-1)
        alpha = self.conflict_gate(gate_input)

        refined_consensus = self.consensus_proj(consensus_feat)

        final_a = refined_consensus * alpha + diff_out_a
        final_t = refined_consensus * alpha + diff_out_t
        final_v = refined_consensus * alpha + diff_out_v

        x_fusion = torch.cat([final_a, final_t, final_v], dim=-1)

        res = x_fusion
        u = F.relu(self.proj1(x_fusion))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = (u + res) * mask_float

        return hidden, None, np.array(weight_save), ortho_loss

