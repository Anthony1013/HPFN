
"""
* @name: hpfn.py
* @description: HPFN model.
"""

import torch
from torch import nn
from .bert import BertTextEncoder
from .hpfn_blocks import ProjMLP, MonoCNN, interleave, CoMambaPair, Transformer


def _req(ns, key):
    if not hasattr(ns, key):
        raise ValueError(f"[HPFN] wrong：model.{key}")
    return getattr(ns, key)


class HPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        margs = args.model

        self.L = int(_req(margs, "token_length"))
        self.d = int(_req(margs, "proj_input_dim"))

        token_len = int(_req(margs, "token_len"))
        token_dim = int(_req(margs, "token_dim"))
        self.h_hyper = nn.Parameter(torch.ones(1, token_len, token_dim))

        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=_req(margs, "bert_pretrained")
        )

        l_in = int(_req(margs, "l_input_dim"))
        a_in = int(_req(margs, "a_input_dim"))
        v_in = int(_req(margs, "v_input_dim"))

        cnn_depth = int(_req(margs, "cnn_depth"))
        cnn_ks    = int(_req(margs, "cnn_kernel_size"))
        cnn_dil   = bool(_req(margs, "cnn_dilated"))

        proj_dropout = float(_req(margs, "proj_dropout"))

        self.proj_l = nn.Sequential(
            MonoCNN(dim=l_in, depth=cnn_depth, kernel_size=cnn_ks, dilated=cnn_dil),
            ProjMLP(l_in, self.d, dropout=proj_dropout),
        )
        self.proj_a = nn.Sequential(
            MonoCNN(dim=a_in, depth=cnn_depth, kernel_size=cnn_ks, dilated=cnn_dil),
            ProjMLP(a_in, self.d, dropout=proj_dropout),
        )
        self.proj_v = nn.Sequential(
            MonoCNN(dim=v_in, depth=cnn_depth, kernel_size=cnn_ks, dilated=cnn_dil),
            ProjMLP(v_in, self.d, dropout=proj_dropout),
        )

        self.mamba_use     = bool(_req(margs, "mamba_use"))
        mamba_depth        = int(_req(margs, "mamba_depth"))
        mamba_state_dim    = int(_req(margs, "mamba_state_dim"))
        mamba_expand       = int(_req(margs, "mamba_expand"))
        fb_heads           = int(_req(margs, "mamba_fallback_heads"))
        fb_mlp_ratio       = int(_req(margs, "mamba_fallback_mlp_ratio"))

        self.pair_AT = CoMambaPair(
            dim=self.d, depth=mamba_depth, use_mamba=self.mamba_use,
            d_state=mamba_state_dim, expand=mamba_expand,
            fallback_heads=fb_heads, fallback_mlp_ratio=fb_mlp_ratio
        )
        self.pair_VT = CoMambaPair(
            dim=self.d, depth=mamba_depth, use_mamba=self.mamba_use,
            d_state=mamba_state_dim, expand=mamba_expand,
            fallback_heads=fb_heads, fallback_mlp_ratio=fb_mlp_ratio
        )

        if not self.mamba_use:
            self.pos_AT = nn.Parameter(torch.zeros(1, 2 * self.L, self.d))
            self.pos_VT = nn.Parameter(torch.zeros(1, 2 * self.L, self.d))
        else:
            self.pos_AT = None
            self.pos_VT = None

        fusion_depth   = int(_req(margs, "fusion_layer_depth"))
        fusion_heads   = int(_req(margs, "fusion_heads"))
        fusion_mlp_dim = int(_req(margs, "fusion_mlp_dim"))
        fusion_dropout = float(_req(margs, "fusion_dropout"))
        emb_dropout    = float(_req(margs, "emb_dropout"))

        self.fusion_layer = Transformer(
            num_frames=4 * self.L,
            save_hidden=False,
            token_len=1,
            dim=self.d,
            depth=fusion_depth,
            heads=fusion_heads,
            mlp_dim=fusion_mlp_dim,
            dropout=fusion_dropout,
            emb_dropout=emb_dropout
        )


        self.alpha = float(_req(margs, "alpha"))

        in_dim = int(_req(margs, "token_dim"))
        self.regression_layer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1)
        )

        if not hasattr(self, "_printed_cfg"):
            print("[HPF cfg] L=", self.L, "d=", self.d,
                  "mamba_use=", self.mamba_use,
                  "mamba_depth=", mamba_depth,
                  "mamba_state_dim=", mamba_state_dim,
                  "mamba_expand=", mamba_expand,
                  "fusion_depth=", fusion_depth,
                  "fusion_heads=", fusion_heads,
                  "fusion_mlp_dim=", fusion_mlp_dim,
                  "alpha=", self.alpha)
            self._printed_cfg = True

    def forward(self, x_visual, x_audio, x_text):
        x_text = self.bertmodel(x_text)

        L = self.h_hyper.shape[1]
        h_v = self.proj_v(x_visual)[:, :L]
        h_a = self.proj_a(x_audio)[:,  :L]
        h_t = self.proj_l(x_text)[:,   :L]

        AT = interleave(h_t, h_a)   # [B, 2L, d]
        VT = interleave(h_t, h_v)   # [B, 2L, d]

        if self.pos_AT is not None:
            AT = AT + self.pos_AT
            VT = VT + self.pos_VT

        ATp = self.pair_AT(AT)      # [B, 2L, d]
        VTp = self.pair_VT(VT)      # [B, 2L, d]

        X = torch.cat([ATp, VTp], dim=1)   # [B, 4L, d]
        Z = self.fusion_layer(X)           # [B, 1 + 4L, d]

        cls  = Z[:, 0]
        mean = Z[:, 1:].mean(dim=1)
        feat = self.alpha * cls + (1 - self.alpha) * mean

        return self.regression_layer(feat)


def build_model(args):
    return HPFN(args)
