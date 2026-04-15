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
