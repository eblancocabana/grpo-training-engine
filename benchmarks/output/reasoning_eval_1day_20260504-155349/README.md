# Reasoning Evaluation Run

Command:

```bash
scripts/evaluate_reasoning_vllm.py --models-config configs/models.gsmplus_actual.yaml --datasets gsm8k,gsm-plus,svamp,math500,aime24,aime25,amc23,minerva,gpqa-diamond,arc-challenge,humaneval,mbpp --tier maximal --output-dir benchmarks/output/reasoning_eval_1day_20260504-155349 --protocol deterministic --max-new-tokens 8192 --limit-per-dataset 100 --gpu-memory-utilization 0.90 --max-num-seqs 8 --max-num-batched-tokens 8192 --batch-size 8 --seed 42
```

Models: base, sent_final, sent_best, no_sent_final, no_sent_best

Datasets: gsm8k, gsm-plus, svamp, math500, aime24, aime25, amc23, minerva, gpqa-diamond, arc-challenge, humaneval, mbpp

The best-checkpoint labels are accepted from `models.yaml` only when marked as
`selection_source: in_training_validation`; this script does not select best
checkpoints from benchmark results.

Optional local dataset overrides:

- `REASONING_EVAL_GSM8K_PATH`: optional local JSON/JSONL/CSV override for `gsm8k`
- `REASONING_EVAL_GSM_PLUS_PATH`: optional local JSON/JSONL/CSV override for `gsm-plus`
- `REASONING_EVAL_SVAMP_PATH`: optional local JSON/JSONL/CSV override for `svamp`
- `REASONING_EVAL_MATH500_PATH`: optional local JSON/JSONL/CSV override for `math500`
- `REASONING_EVAL_AIME24_PATH`: optional local JSON/JSONL/CSV override for `aime24`
- `REASONING_EVAL_AIME25_PATH`: optional local JSON/JSONL/CSV override for `aime25`
- `REASONING_EVAL_AMC23_PATH`: optional local JSON/JSONL/CSV override for `amc23`
- `REASONING_EVAL_MINERVA_PATH`: optional local JSON/JSONL/CSV override for `minerva`
- `REASONING_EVAL_GPQA_DIAMOND_PATH`: optional local JSON/JSONL/CSV override for `gpqa-diamond`
- `REASONING_EVAL_ARC_CHALLENGE_PATH`: optional local JSON/JSONL/CSV override for `arc-challenge`
- `REASONING_EVAL_HUMANEVAL_PATH`: optional local JSON/JSONL/CSV override for `humaneval`
- `REASONING_EVAL_MBPP_PATH`: optional local JSON/JSONL/CSV override for `mbpp`
