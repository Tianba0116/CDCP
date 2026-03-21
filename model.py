import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from MOE import MoMKE
from torch.nn import Parameter


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if self.weight is None:
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) / torch.sum(
                self.weight[target] * mask_.squeeze()
            )
        return loss


def gelu(x):
    return 0.5 * x * (
        1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert model_dim % head_count == 0

        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(
            batch_size, -1, head_count * dim_per_head
        )
        output = self.linear(context)
        attn_mean = attn.mean(dim=1)
        return output, attn_mean


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        return x + pos_emb + speaker_emb


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)
            mask = mask.unsqueeze(1)
            context, attn = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if iter != 0:
                inputs_b = self.layer_norm(inputs_b)
            mask = mask.unsqueeze(1)
            context, attn = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(layers)]
        )

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == "MELD":
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        return z * a


class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)

        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat(
            [
                self.fc(a).unsqueeze(-2),
                self.fc(b).unsqueeze(-2),
                self.fc(c).unsqueeze(-2),
            ],
            dim=-2,
        )
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep


class mask_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(mask_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)

        utters = torch.cat([a_new, b_new], dim=-2)
        utters_fc = torch.cat(
            [self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2)],
            dim=-2,
        )
        utters_softmax = self.softmax(utters_fc)
        utters_two_model = utters_softmax * utters
        final_rep = torch.sum(utters_two_model, dim=-2, keepdim=False)
        return final_rep


class ConNet(nn.Module):
    def __init__(self, input_dim):
        super(ConNet, self).__init__()
        self.conf_net = nn.Linear(input_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conf_net(x))


class SGConv_Our(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SGConv_Our, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except Exception:
            pass

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class s_AttentionDiffusion(nn.Module):
    def __init__(self, hidden_dim, num_heads, steps=3, theta_hidden_dim=128):
        super(s_AttentionDiffusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.steps = steps

        self.theta_mlp = nn.Sequential(
            nn.Linear(hidden_dim, theta_hidden_dim),
            nn.ReLU(),
            nn.Linear(theta_hidden_dim, steps + 1),
        )
        self.pos_emb = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, node_features1, graph, speaker_emb):
        B, N, D = node_features.shape
        device = node_features.device

        x = self.pos_emb(node_features, speaker_emb)
        x = self.dropout(x)

        attn_out, attn_scores = self.self_attention(x, x, x)

        masked_attn_scores = attn_scores * graph.unsqueeze(1)
        row_sum = masked_attn_scores.sum(dim=-1, keepdim=True) + 1e-9
        P = masked_attn_scores / row_sum
        P = P.mean(dim=1)

        theta_logits = self.theta_mlp(x)
        theta = torch.softmax(theta_logits, dim=-1)

        final_features = torch.zeros_like(x)
        P_power = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

        for i in range(self.steps + 1):
            if i > 0:
                P_power = torch.bmm(P_power, P)

            weight = theta[:, :, i].unsqueeze(-1)
            weighted_P = weight * P_power
            final_features += torch.bmm(weighted_P, x)

        output = self.norm(final_features + node_features)
        return output


class LengthAdaptiveSemanticGraph(nn.Module):
    def __init__(self, input_dim, shared_dim=64, init_alpha=1.0):
        super(LengthAdaptiveSemanticGraph, self).__init__()
        self.proj = nn.Linear(input_dim, shared_dim)
        nn.init.xavier_normal_(self.proj.weight)
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, umask, qmask):
        B, L, D = x.shape
        device = x.device

        x_proj = F.normalize(self.proj(x), p=2, dim=-1)
        sim_matrix = torch.bmm(x_proj, x_proj.transpose(1, 2))
        sim_matrix = (sim_matrix + 1) / 2

        indices = torch.arange(L, device=device).float()
        dist_matrix = (indices.unsqueeze(1) - indices.unsqueeze(0)).abs()

        learned_alpha = torch.exp(self.raw_alpha)
        learned_alpha = torch.clamp(learned_alpha, min=0.1, max=5.0)
        sigma = learned_alpha * math.log2(max(L, 2))

        decay_matrix = torch.exp(-(dist_matrix ** 2) / (2 * (sigma ** 2)))
        decay_matrix = decay_matrix.unsqueeze(0).expand(B, -1, -1)

        if qmask.dim() == 3:
            qmask_ids = qmask.argmax(dim=-1)
        else:
            qmask_ids = qmask

        s_id = qmask_ids.unsqueeze(2)
        t_id = qmask_ids.unsqueeze(1)
        is_same = s_id == t_id
        is_diff = ~is_same

        scores_s = sim_matrix
        scores_c = sim_matrix * decay_matrix

        causal_mask = torch.tril(torch.ones(L, L, device=device)).bool().unsqueeze(0)
        valid_mask = (umask.unsqueeze(2) * umask.unsqueeze(1)).bool()
        base_mask = causal_mask & valid_mask

        all_valid_scores = (
            (scores_s * is_same.float() + scores_c * is_diff.float()) * base_mask.float()
        )
        mean_score = all_valid_scores.sum() / (base_mask.float().sum() + 1e-6)

        smask = (scores_s > mean_score) & is_same & base_mask
        cmask = (scores_c > mean_score) & is_diff & base_mask

        eye = torch.eye(L, device=device).bool().unsqueeze(0)
        smask = smask | (eye & valid_mask)

        return smask, cmask


class CDCP(nn.Module):
    def __init__(
        self,
        args,
        dataset,
        temp,
        D_text,
        D_visual,
        D_audio,
        n_head,
        n_classes,
        hidden_dim,
        n_speakers,
        dropout,
    ):
        super(CDCP, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers

        if self.n_speakers == 2:
            padding_idx = 2
        elif self.n_speakers == 9:
            padding_idx = 9
        else:
            padding_idx = n_speakers

        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)

        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)

        self.t_t = TransformerEncoder(
            d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout
        )
        self.a_a = TransformerEncoder(
            d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout
        )
        self.v_v = TransformerEncoder(
            d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout
        )

        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.t_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.a_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.v_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

        self.modal_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
        )

        self.features_reduce_t = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_a = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_v = nn.Linear(3 * hidden_dim, hidden_dim)

        steps = args.steps
        thera_hidden_dim = args.thera_hidden_dim

        self.t_t_s = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )
        self.t_t_c = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )
        self.a_a_s = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )
        self.a_a_c = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )
        self.v_v_s = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )
        self.v_v_c = s_AttentionDiffusion(
            hidden_dim, n_head, steps=steps, theta_hidden_dim=thera_hidden_dim
        )

        self.MOE = MoMKE(
            args=args,
            adim=hidden_dim,
            tdim=hidden_dim,
            vdim=hidden_dim,
            D_e=hidden_dim,
            n_classes=n_classes,
            depth=args.MOE_depth,
            num_heads=8,
            mlp_ratio=1.0,
        )

        self.fusion = nn.Linear(3 * hidden_dim, hidden_dim)

        self.t_k = LengthAdaptiveSemanticGraph(hidden_dim, shared_dim=64)
        self.a_k = LengthAdaptiveSemanticGraph(hidden_dim, shared_dim=64)
        self.v_k = LengthAdaptiveSemanticGraph(hidden_dim, shared_dim=64)

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len, label, args):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx

        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()

        spk_embeddings = self.speaker_embeddings(spk_idx)

        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        t_t_transformer_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        t_smask, t_cmask = self.t_k(t_t_transformer_out, u_mask, qmask)
        t_t_transformer_out_smask = self.t_t_s(textf, textf, t_smask, spk_embeddings)
        t_t_transformer_out_cmask = self.t_t_c(textf, textf, t_cmask, spk_embeddings)

        a_a_transformer_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        a_smask, a_cmask = self.a_k(a_a_transformer_out, u_mask, qmask)
        a_a_transformer_out_smask = self.a_a_s(acouf, acouf, a_smask, spk_embeddings)
        a_a_transformer_out_cmask = self.a_a_c(acouf, acouf, a_cmask, spk_embeddings)

        v_v_transformer_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)
        v_smask, v_cmask = self.v_k(v_v_transformer_out, u_mask, qmask)
        v_v_transformer_out_smask = self.v_v_s(visuf, visuf, v_smask, spk_embeddings)
        v_v_transformer_out_cmask = self.v_v_c(visuf, visuf, v_cmask, spk_embeddings)

        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.a_t_gate(t_t_transformer_out_smask)
        v_t_transformer_out = self.v_t_gate(t_t_transformer_out_cmask)

        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(a_a_transformer_out_smask)
        v_a_transformer_out = self.v_a_gate(a_a_transformer_out_cmask)

        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(v_v_transformer_out_smask)
        a_v_transformer_out = self.a_v_gate(v_v_transformer_out_cmask)

        t_transformer_out = self.features_reduce_t(
            torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1)
        )
        a_transformer_out = self.features_reduce_a(
            torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1)
        )
        v_transformer_out = self.features_reduce_v(
            torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1)
        )

        all_feature = torch.cat([t_transformer_out, a_transformer_out, v_transformer_out], dim=-1)
        out = self.MOE(all_feature, u_mask, spk_embeddings)
        all_transformer_out = self.fusion(out[0])
        loss_o = out[3]

        t_final_out = self.t_output_layer(t_transformer_out)
        a_final_out = self.a_output_layer(a_transformer_out)
        v_final_out = self.v_output_layer(v_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        t_log_prob = F.log_softmax(t_final_out, 2)
        a_log_prob = F.log_softmax(a_final_out, 2)
        v_log_prob = F.log_softmax(v_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        return t_log_prob, a_log_prob, v_log_prob, all_log_prob, all_prob, loss_o