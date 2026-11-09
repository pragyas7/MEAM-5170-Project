"""LLM backends.

`OpenAICompatible` talks to any chat-completions endpoint (OpenAI, Groq,
Together, vLLM) using the stdlib, so the baseline does not need the openai
package. `HuggingFaceBackend` loads a local transformers model when the
optional train dependencies are installed. Neither is required to run the
classical baseline.
"""

from __future__ import annotations

import json
import os
import urllib.request


class LLMError(RuntimeError):
    pass


class OpenAICompatible:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY is not set")

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": "You output only JSON action lists."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode())
        return payload["choices"][0]["message"]["content"]


class HuggingFaceBackend:
    """Local causal LM. Optional adapter_path is a LoRA checkpoint."""

    def __init__(self, model_name: str, adapter_path: str | None = None, max_new_tokens: int = 400):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        import torch

        messages = [
            {"role": "system", "content": "You output only JSON action lists."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 1e-6
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-3) if do_sample else None,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_client(backend: str = "auto", model: str | None = None, adapter: str | None = None):
    """Return a client, or raise LLMError if the requested backend is unavailable."""
    choice = backend
    if choice == "auto":
        if model or os.environ.get("TGL_HF_MODEL"):
            choice = "hf"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            raise LLMError(
                "No LLM is configured. Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, "
                "OPENAI_MODEL) or pass --model for a local Hugging Face checkpoint. "
                "The classical planners naive, repair, and oracle do not need a model."
            )
    if choice in ("openai", "api"):
        return OpenAICompatible(model=model)
    if choice in ("hf", "transformers"):
        name = model or os.environ.get("TGL_HF_MODEL")
        if not name:
            raise LLMError("Hugging Face backend requires a model name")
        return HuggingFaceBackend(name, adapter_path=adapter)
    raise LLMError(f"unknown LLM backend {backend!r}")
