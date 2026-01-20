import json
import torch
from pathlib import Path

import sys
sys.path.append(str(Path('.').resolve()))
sys.path.append(str(Path("..").resolve()))

from summarization.metrics_utils import compute_custom_metrics

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

test = load_jsonl("summarization_data/exp_1_test.json")
preds = load_jsonl("summarization_results/gpt4_task1.jsonl")

texts = [ex["text"] for ex in test]
gold = [ex["summary"] for ex in test]
model_preds = [ex.get("summary", "") for ex in preds]

metrics = compute_custom_metrics(
    texts, gold, model_preds,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
print(metrics)
