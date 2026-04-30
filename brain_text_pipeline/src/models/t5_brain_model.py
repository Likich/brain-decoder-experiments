"""T5 wrapper that consumes brain sequences via encoder inputs_embeds."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from brain_text_pipeline.src.models.brain_encoder import BrainEncoder


class T5BrainModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        brain_dim: int,
        brain_encoder_layers: int = 2,
        brain_encoder_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        d_model = self.t5.config.d_model
        self.brain_encoder = BrainEncoder(
            brain_dim=brain_dim,
            d_model=d_model,
            num_layers=brain_encoder_layers,
            nhead=brain_encoder_heads,
            dropout=dropout,
        )

    def forward(
        self,
        brain_seq: torch.Tensor,
        brain_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        assert brain_seq.ndim == 3, "brain_seq must be [B,T,D]"
        enc_out = self.brain_encoder(brain_seq, brain_mask)
        return self.t5(
            inputs_embeds=enc_out,
            attention_mask=brain_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def freeze_t5(self) -> None:
        for p in self.t5.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n: int) -> None:
        # unfreeze last N encoder + decoder blocks
        if n <= 0:
            return
        enc_blocks = self.t5.encoder.block
        dec_blocks = self.t5.decoder.block
        for layer in enc_blocks[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for layer in dec_blocks[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        # final layer norms
        for p in self.t5.encoder.final_layer_norm.parameters():
            p.requires_grad = True
        for p in self.t5.decoder.final_layer_norm.parameters():
            p.requires_grad = True
        if hasattr(self.t5, "lm_head"):
            for p in self.t5.lm_head.parameters():
                p.requires_grad = True

    def unfreeze_decoder_cross_attention(self, last_n: int = 0) -> None:
        """Unfreeze only decoder encoder-decoder attention blocks.

        This is useful for brain-conditioned decoding: the language model and
        LM head stay fixed, while cross-attention learns how to read the
        brain-derived encoder states.
        """
        blocks = self.t5.decoder.block
        if last_n and last_n > 0:
            blocks = blocks[-last_n:]
        for block in blocks:
            # T5 decoder block layout:
            # layer[0] = self-attention, layer[1] = encoder-decoder attention,
            # layer[2] = feed-forward.
            if len(block.layer) < 2:
                continue
            for p in block.layer[1].parameters():
                p.requires_grad = True


class T5FixedAuxResidualMEGModel(nn.Module):
    """Frozen aux-conditioned T5 with a trainable residual MEG branch.

    The base T5 weights and the auxiliary encoder are loaded from a standalone
    text-aux run and kept fixed. A separate MEG encoder produces a residual
    encoder-state update that is added on top of the frozen auxiliary branch.
    This makes ``MEG ZERO`` exactly the frozen auxiliary baseline, rather than
    a separately trained joint model with a different backbone.
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        aux_dim: int,
        meg_dim: int,
        aux_encoder_layers: int = 2,
        aux_encoder_heads: int = 4,
        meg_encoder_layers: int = 2,
        meg_encoder_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model_name_or_path)
        d_model = self.t5.config.d_model
        self.aux_encoder = BrainEncoder(
            brain_dim=aux_dim,
            d_model=d_model,
            num_layers=aux_encoder_layers,
            nhead=aux_encoder_heads,
            dropout=dropout,
        )
        self.meg_encoder = BrainEncoder(
            brain_dim=meg_dim,
            d_model=d_model,
            num_layers=meg_encoder_layers,
            nhead=meg_encoder_heads,
            dropout=dropout,
        )

    def forward(
        self,
        aux_seq: torch.Tensor,
        meg_seq: torch.Tensor,
        brain_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        *,
        use_meg: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        assert aux_seq.ndim == 3, "aux_seq must be [B,T,D]"
        assert meg_seq.ndim == 3, "meg_seq must be [B,T,D]"
        with torch.no_grad():
            aux_enc = self.aux_encoder(aux_seq, brain_mask)
        if use_meg:
            enc_out = aux_enc + self.meg_encoder(meg_seq, brain_mask)
        else:
            enc_out = aux_enc
        return self.t5(
            inputs_embeds=enc_out,
            attention_mask=brain_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def load_aux_encoder(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        self.aux_encoder.load_state_dict(state)

    def load_meg_encoder(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        self.meg_encoder.load_state_dict(state)

    def freeze_base(self) -> None:
        for p in self.t5.parameters():
            p.requires_grad = False
        for p in self.aux_encoder.parameters():
            p.requires_grad = False
