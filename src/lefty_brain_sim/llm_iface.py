from __future__ import annotations
import numpy as np
from dataclasses import dataclass
# --- OpenAI-compatible (vLLM) backend ---
from dataclasses import dataclass
import os, re, json
import numpy as np
from openai import OpenAI
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass


@dataclass
class LLMOut:
    evidence: dict
    report: str

class MockLLM:
    """
    Placeholder LLM evidence provider. Given a feature vector z and optional
    memory hits, emits a soft evidence distribution and a short report.
    Replace with an actual LLM call (RAG) as needed.
    """
    def __init__(self, vocab=("APPLE", "PEAR")):
        self.vocab = vocab

    def infer(self, z: np.ndarray, memory_hits: list[dict]) -> LLMOut:
        # Toy rule: mean(z) > 0 → A; else B; modulated by memory
        bias = float(np.tanh(z.mean()))
        mem_bias = 0.1 * sum([m.get("bias", 0.0) for m in memory_hits])
        a = 0.5 + 0.4 * (bias + mem_bias)
        a = float(np.clip(a, 0.01, 0.99))
        b = 1.0 - a
        token = self.vocab[0] if a > b else self.vocab[1]
        report = f"I saw '{token}'. Confidence {max(a,b):.2f}"
        return LLMOut(evidence={"A": a, "B": b}, report=report)
    
    


class LocalDecoderNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LocalDecoderLLM:
    """
    Tiny local 'LLM' that reads brain state vectors and
    outputs a class label + confidence.
    """
    def __init__(
        self,
        ckpt_path: str,
        meta_path: str = "models/brain_decoder_meta.json",
        device: str = "cpu",
    ):
        ckpt = torch.load(ckpt_path, map_location=device)
        in_dim = ckpt["in_dim"]
        num_classes = ckpt.get("num_classes", 2)

        self.model = LocalDecoderNet(in_dim=in_dim, num_classes=num_classes)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device

        meta = json.loads(Path(meta_path).read_text())
        self.class_names = meta["class_names"]

    def infer(self, z: np.ndarray, memory_hits: list[dict]) -> LLMOutput:
        """
        z: brain state vector (encoder output or workspace readout)
        memory_hits: unused for now, but kept for API compatibility
        """
        x = torch.from_numpy(z.astype("float32")).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(probs.argmax())
        label = self.class_names[idx]
        p = float(probs[idx])

        # map [0, 1] → [-1, 1] as “evidence” scalar
        evidence = 2.0 * (p - 0.5)

        report = f"I saw '{label}' (p={p:.2f})"
        return LLMOut(evidence=evidence, report=report)


class OpenAICompatLLM:
    def __init__(self, model, max_tokens, temperature,
                 api_key="EMPTY", base_url="http://10.180.132.23:8188/v1"):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)



    def infer(self, z: np.ndarray, hits: list[dict]) -> LLMOut:
        mean_z = float(np.mean(z))

        prompt = f"""
You receive a 1D brain-state summary z with mean value {mean_z:+.3f}.
If z is NEGATIVE, the stimulus was APPLE.
If z is POSITIVE, the stimulus was PEAR.

Classify which stimulus is more likely.

Return ONLY valid JSON of the form:
{{"label": "APPLE" or "PEAR", "p": number between 0.50 and 0.99}}
""".strip()

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful classifier. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (resp.choices[0].message.content or "").strip()

        # --- Robust JSON extraction ---
        # Try to grab the first {...} block from the reply
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if m:
            raw_json = m.group(0)
        else:
            raw_json = raw  # hope it's already just JSON

        try:
            data = json.loads(raw_json)
        except Exception:
            # Fallback: if the model misbehaves, default to neutral PEAR
            data = {"label": "PEAR", "p": 0.5}

        label = data.get("label", "PEAR")
        p = float(data.get("p", 0.5))

        # Signed drift: PEAR positive, APPLE negative
        drift = p if label == "PEAR" else -p

        evidence = {"A": 0.0, "B": drift}
        report = f"I saw '{label}' (p={p:.2f})"
        return LLMOut(evidence=evidence, report=report)

