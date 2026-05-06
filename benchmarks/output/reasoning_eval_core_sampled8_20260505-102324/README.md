# Reasoning Evaluation Run

Command:

```bash
scripts/evaluate_reasoning_vllm.py --models-config configs/models.gsmplus_actual.yaml --models base,sent_best,sent_final,no_sent_final --datasets gsm8k,gsm-plus,math500,aime24 --tier maximal --output-dir benchmarks/output/reasoning_eval_core_sampled8_20260505-102324 --protocol sampled --n-samples 8 --max-new-tokens 8192 --limit-per-dataset 100 --gpu-memory-utilization 0.90 --max-num-seqs 8 --max-num-batched-tokens 8192 --batch-size 8 --seed 46
```

Models: base, sent_final, sent_best, no_sent_final

Datasets: gsm8k, gsm-plus, math500, aime24

The best-checkpoint labels are accepted from `models.yaml` only when marked as
`selection_source: in_training_validation`; this script does not select best
checkpoints from benchmark results.

Optional local dataset overrides:

- `REASONING_EVAL_GSM8K_PATH`: optional local JSON/JSONL/CSV override for `gsm8k`
- `REASONING_EVAL_GSM_PLUS_PATH`: optional local JSON/JSONL/CSV override for `gsm-plus`
- `REASONING_EVAL_MATH500_PATH`: optional local JSON/JSONL/CSV override for `math500`
- `REASONING_EVAL_AIME24_PATH`: optional local JSON/JSONL/CSV override for `aime24`
