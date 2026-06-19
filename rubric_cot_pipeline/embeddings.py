from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION = (
    "Given a user's past item interactions and optional recommendation reasoning, "
    "retrieve items the user is likely to prefer next."
)


def resolve_torch_dtype(name: str):
    import torch

    name = (name or "auto").lower()
    if name == "auto":
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def first_device(model):
    return next(model.parameters()).device


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def format_qwen3_query(text: str, instruction: str = DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION) -> str:
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def append_recommendation_reasoning(user_history: str, reasoning: str) -> str:
    reasoning = str(reasoning or "").strip()
    if not reasoning:
        return str(user_history or "").strip()
    return f"{str(user_history or '').strip()}\n\nRecommendation reasoning:\n{reasoning}"


@dataclass
class Qwen3TextEmbedder:
    model_path: str
    max_length: int = 8192
    batch_size: int = 8
    torch_dtype: str = "bfloat16"
    device: str = "auto"
    query_instruction: str = DEFAULT_RECOMMENDATION_QUERY_INSTRUCTION
    output_dim: int = 0

    def __post_init__(self) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.F = F
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"trust_remote_code": True, "torch_dtype": resolve_torch_dtype(self.torch_dtype)}
        if self.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()

    def _prepare(self, texts: Iterable[str], *, is_query: bool) -> list[str]:
        if is_query:
            return [format_qwen3_query(str(text or ""), self.query_instruction) for text in texts]
        return [str(text or "") for text in texts]

    def encode(self, texts: Iterable[str], *, is_query: bool = False):
        prepared = self._prepare(texts, is_query=is_query)
        if not prepared:
            return self.torch.empty((0, 0), dtype=self.torch.float32)

        batches = []
        with self.torch.no_grad():
            for start in range(0, len(prepared), self.batch_size):
                batch_texts = prepared[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                device = first_device(self.model)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.model(**batch)
                embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                if self.output_dim and self.output_dim > 0:
                    embeddings = embeddings[:, : self.output_dim]
                embeddings = self.F.normalize(embeddings.float(), p=2, dim=1).cpu()
                batches.append(embeddings)
        return self.torch.cat(batches, dim=0)

    def encode_queries(self, texts: Iterable[str]):
        return self.encode(texts, is_query=True)

    def encode_documents(self, texts: Iterable[str]):
        return self.encode(texts, is_query=False)

    def pairwise_cosine(self, queries: Iterable[str], documents: Iterable[str]) -> list[float]:
        query_emb = self.encode_queries(queries)
        doc_emb = self.encode_documents(documents)
        if query_emb.shape[0] != doc_emb.shape[0]:
            raise ValueError("queries and documents must have the same length")
        return (query_emb * doc_emb).sum(dim=1).tolist()
