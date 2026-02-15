"""
Semantic Entropy-based Curriculum (SENT) calculator.

Calculates semantic entropy per query by sampling M responses, extracting
final answers using the provided RuleBasedVerifier, clustering responses by
numeric equivalence, and computing H_SE = -sum(P(C_i) * log(P(C_i))).

Implements checkpointing and caching to resume long-running dataset
processing jobs.
"""
from __future__ import annotations

import os
import time
import json
import math
import hashlib
from typing import Any, Dict, List, Tuple

import torch

from src.grpo.verifier import RuleBasedVerifier
from src.core.memory_manager import MemoryManager
from src.utils.config import Config


def cluster_by_answer(
    verifier: RuleBasedVerifier,
    responses: List[str],
) -> List[Dict[str, Any]]:
    """Cluster responses by final answer using verifier.normalize_number().

    Returns list of clusters, each cluster is dict:
        { 'answer': normalized_value_or_raw, 'indices': [i,...] }
    Unextractable answers are each treated as unique clusters (high entropy).
    """
    clusters: List[Dict[str, Any]] = []

    for idx, resp in enumerate(responses):
        extracted = verifier.extract_final_answer(resp)
        norm = verifier.normalize_number(extracted)

        placed = False
        if norm is not None:
            for c in clusters:
                if isinstance(c["answer"], (int, float)) and abs(c["answer"] - norm) < 1e-3:
                    c["indices"].append(idx)
                    placed = True
                    break

        if not placed:
            clusters.append({
                "answer": norm if norm is not None else extracted,
                "indices": [idx],
            })

    return clusters


def compute_semantic_entropy(clusters: List[Dict[str, Any]], total_samples: int) -> float:
    """Compute H_SE using Monte Carlo frequency estimation.

    P(C_i) = |C_i| / M, where M = total_samples.
    H_SE = -sum(P(C_i) * log(P(C_i)))
    """
    if not clusters or total_samples == 0:
        return 0.0

    entropy = 0.0
    for c in clusters:
        p = len(c["indices"]) / total_samples
        if p > 0:
            entropy -= p * math.log(p)

    return float(entropy)


def save_sent_cache(path: str, data: Dict[str, Any]) -> None:
    """Atomic save (write to .tmp then rename)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    if path.endswith('.json'):
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2, default=lambda o: o if o is not None else None)
    else:
        torch.save(data, tmp)
    os.replace(tmp, path)


def load_sent_cache(path: str) -> Dict[str, Any]:
    """Load a SENT cache file (.json or .pt)."""
    if path.endswith('.json'):
        with open(path, "r") as fh:
            return json.load(fh)
    else:
        return torch.load(path, weights_only=False)


def make_sent_metadata(config, status: str = "in_progress", backend: str = "hf") -> Dict[str, Any]:
    """Create metadata block for cache files."""
    cfg = config.to_dict() if hasattr(config, "to_dict") else {}
    cfg_ser = json.dumps(cfg, sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_ser.encode()).hexdigest()
    return {
        "version": "sent_v1",
        "config_hash": cfg_hash,
        "created_at": time.time(),
        "status": status,
        "model_id": getattr(config.model, "model_id", None),
        "backend": backend,
    }


class SemanticEntropyCalculator:
    """Calculator for semantic entropy (SENT) per query.

    Usage:
        calc = SemanticEntropyCalculator(model, tokenizer, verifier, config)
        entropy, meta = calc.compute_entropy_for_query("What is 2+2?", num_samples=4)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        verifier: RuleBasedVerifier,
        config: Config,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory_manager = MemoryManager(device=self.device)

        # Checkpoint settings
        self.checkpoint_interval = getattr(self.config.sent, "checkpoint_interval", 100)

    def compute_entropy_for_query(self, question: str, num_samples: int = 4) -> Tuple[float, Dict[str, Any]]:
        """Generate `num_samples` responses to `question`, cluster by final answer,
        and compute semantic entropy using Monte Carlo frequency estimation.

        P(C_i) = |C_i| / M, where M = num_samples.
        H_SE = -sum(P(C_i) * log(P(C_i)))

        Returns:
            entropy: float
            metadata: dict with clusters and counts (not full responses)
        """
        # Prepare prompt/tokenize
        self.model.eval()

        prompt = question
        # Use tokenizer to prepare input ids
        enc = self.tokenizer([prompt], return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Generation settings
        temperature = getattr(self.config.sent, "temperature", 1.0)
        gen_kwargs = dict(
            max_new_tokens=self.config.training.max_response_length,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        responses: List[str] = []

        # Expand to num_samples by repeating input — generate all at once
        batch_ids = input_ids.repeat(num_samples, 1)
        batch_mask = attention_mask.repeat(num_samples, 1)

        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            sequences = self.model.generate(
                input_ids=batch_ids, attention_mask=batch_mask, **gen_kwargs
            )

            for j in range(num_samples):
                resp_ids = sequences[j, prompt_len:]
                text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                responses.append(text)

            del sequences
            self.memory_manager.clear_cache()

        # Cluster by extracted final answers
        clusters = self._cluster_by_answer(responses)

        entropy = self._compute_semantic_entropy(clusters, num_samples)

        metadata = {
            "num_samples": num_samples,
            "num_clusters": len(clusters),
            "clusters": [
                {
                    "answer": c["answer"], 
                    "count": len(c["indices"]),
                }
                for c in clusters
            ],
        }

        # Free VRAM
        torch.cuda.empty_cache()

        return float(entropy), metadata

    def compute_entropy_for_batch(self, questions: List[str], num_samples: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        """Generate `num_samples` responses for a BATCH of questions, and compute
        semantic entropy using Monte Carlo frequency estimation.
        
        Args:
            questions: List of question strings
            num_samples: Number of samples per question (M)
            
        Returns:
            List of (entropy, metadata) tuples, one per question.
        """
        batch_size = len(questions)
        if batch_size == 0:
            return []

        self.model.eval()

        # Tokenize batch
        enc = self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        
        prompt_len = input_ids.shape[1]

        # Generation settings
        temperature = getattr(self.config.sent, "temperature", 1.0)
        gen_kwargs = dict(
            max_new_tokens=self.config.training.max_response_length,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        responses_per_q: List[List[str]] = [[] for _ in range(batch_size)]

        # Expand inputs: repeat_interleave repeats generic structure like [A, B] -> [A, A, B, B]
        batch_ids = input_ids.repeat_interleave(num_samples, dim=0)
        batch_mask = attention_mask.repeat_interleave(num_samples, dim=0)

        with torch.no_grad():
            sequences = self.model.generate(
                input_ids=batch_ids, attention_mask=batch_mask, **gen_kwargs
            )

            total_samples = batch_size * num_samples
            
            for k in range(total_samples):
                resp_ids = sequences[k, prompt_len:]
                text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                q_idx = k // num_samples
                responses_per_q[q_idx].append(text)
            
            del sequences
            self.memory_manager.clear_cache()
            
        # Cluster and compute entropy
        results = []
        for q_idx in range(batch_size):
            clusters = self._cluster_by_answer(responses_per_q[q_idx])
            entropy = self._compute_semantic_entropy(clusters, num_samples)
            metadata = {
                "num_samples": num_samples,
                "num_clusters": len(clusters),
                "clusters": [
                    {
                        "answer": c["answer"], 
                        "count": len(c["indices"]),
                    }
                    for c in clusters
                ],
            }
            results.append((entropy, metadata))
            
        torch.cuda.empty_cache()
        return results

    def process_dataset(self, dataset: List[Dict[str, Any]], cache_path: str, resume: bool = True, progress_callback=None, batch_size: int = 1) -> None:
        """Process an entire dataset (list of examples with 'question' key), computing
        entropy per query and saving cache periodically.

        Args:
            dataset: list of examples, each must have 'question' and optionally 'id'
            cache_path: path to write cache JSON
            resume: if True, load partial results and continue
            progress_callback: optional callable(current_idx, total) for progress reporting
            batch_size: Number of queries to process in parallel (default 1)
        """
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        start_idx = 0
        entropies: List[float] = []
        clusters_list: List[Any] = []
        indices: List[Any] = []

        # Try resume
        if resume and os.path.exists(cache_path):
            try:
                data = self.load_cache(cache_path)
                entropies = data.get("entropies", [])
                clusters_list = data.get("clusters", [])
                indices = data.get("indices", [])
                start_idx = len(entropies)
            except Exception:
                start_idx = 0

        total = len(dataset)

        for i in range(start_idx, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch = dataset[i : batch_end]

            # Prepare batch questions
            questions = []
            valid_indices_in_batch = []
            
            for b_idx, ex in enumerate(batch):
                q = ex.get("question") or ex.get("questions") or ex.get("prompt")
                if q is None:
                    continue
                questions.append(q)
                valid_indices_in_batch.append(b_idx)
            
            num_samples = getattr(self.config.sent, "num_samples", 4)
            
            batch_results = []
            if questions:
                try:
                    batch_results = self.compute_entropy_for_batch(questions, num_samples=num_samples)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # Check for OOM
                    msg = str(e).lower()
                    if ("out of memory" in msg or "allocate" in msg) and len(questions) > 1:
                        # Fallback to serial processing
                        self.memory_manager.clear_cache(aggressive=True)
                        torch.cuda.empty_cache()
                        
                        # Process one by one
                        for q in questions:
                            try:
                                res = self.compute_entropy_for_batch([q], num_samples=num_samples)
                                batch_results.extend(res)
                            except Exception as e2:
                                # If single query fails, we can't do much. 
                                # Maybe skip it? Or fail hard.
                                raise e2
                    else:
                        raise

            # Map results back to dataset indices
            current_result_idx = 0
            for b_idx in range(len(batch)):
                global_idx = i + b_idx
                ex = batch[b_idx]
                
                if b_idx in valid_indices_in_batch:
                    if current_result_idx < len(batch_results):
                        h, meta = batch_results[current_result_idx]
                        current_result_idx += 1
                        
                        entropies.append(h)
                        clusters_list.append(meta.get("clusters", []))
                    else:
                        # Should not happen if logic is correct
                        entropies.append(float('nan'))
                        clusters_list.append([])
                else:
                    # Was skipped
                    entropies.append(float('nan'))
                    clusters_list.append([])
                
                indices.append(ex.get("id", global_idx))

            if progress_callback is not None:
                progress_callback(i + len(batch), total)

            # Checkpoint
            if (i + len(batch)) % self.checkpoint_interval < batch_size:
                cache = {
                    "metadata": self._make_metadata(status="in_progress"),
                    "indices": indices,
                    "entropies": entropies,
                    "clusters": clusters_list,
                }
                self.save_cache(cache_path, cache)

        # Finalize: sort by entropy and save
        paired = list(zip(indices, entropies, clusters_list))
        # Sort increasing entropy (easy -> hard), but keep NaNs at end
        paired_sorted = sorted(paired, key=lambda x: (math.isnan(x[1]), x[1]))

        final_cache = {
            "metadata": self._make_metadata(status="complete"),
            "indices": [p[0] for p in paired_sorted],
            "entropies": [p[1] for p in paired_sorted],
            "clusters": [p[2] for p in paired_sorted],
        }

        self.save_cache(cache_path, final_cache)

    def _cluster_by_answer(self, responses: List[str]) -> List[Dict[str, Any]]:
        """Delegate to module-level cluster_by_answer."""
        return cluster_by_answer(self.verifier, responses)

    def _compute_semantic_entropy(self, clusters: List[Dict[str, Any]], total_samples: int) -> float:
        """Delegate to module-level compute_semantic_entropy."""
        return compute_semantic_entropy(clusters, total_samples)

    def save_cache(self, path: str, data: Dict[str, Any]) -> None:
        """Delegate to module-level save_sent_cache."""
        save_sent_cache(path, data)

    def load_cache(self, path: str) -> Dict[str, Any]:
        """Delegate to module-level load_sent_cache."""
        return load_sent_cache(path)

    def _make_metadata(self, status: str = "in_progress") -> Dict[str, Any]:
        """Delegate to module-level make_sent_metadata."""
        return make_sent_metadata(self.config, status, backend="hf")
