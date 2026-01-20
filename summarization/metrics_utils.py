import numpy as np
from collections import defaultdict

import evaluate
from rouge_score import rouge_scorer


def get_rouge_score(gold, pred):
    """Return rouge1/2/3/4/L F1 scores for a single pair."""
    rouge_scores = ["rouge1", "rouge2", "rouge3", "rouge4", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_scores, use_stemmer=True)
    scores = scorer.score(gold, pred)
    return {k: scores[k].fmeasure * 100 for k in rouge_scores}


def compute_custom_metrics(srcs, golds, preds, device):
    """Replicates the custom metrics used in the paper."""
    scores = defaultdict(list)
    bertscore = evaluate.load("bertscore")
    sari = evaluate.load("sari")

    for gold, pred in zip(golds, preds):
        for k, v in get_rouge_score(gold, pred).items():
            scores[k].append(v)
        scores["words"].append(len(pred.split(" ")))
    for k, v in scores.items():
        scores[k] = np.mean(v)

    scores["bert_score"] = np.mean(
        bertscore.compute(predictions=preds, references=golds, lang="en", device=device)["f1"]
    ) * 100
    scores["bert_score_deberta-large"] = np.mean(
        bertscore.compute(
            predictions=preds, references=golds, device=device, model_type="microsoft/deberta-large-mnli"
        )["f1"]
    ) * 100
    scores["sari"] = sari.compute(sources=srcs, predictions=preds, references=[[g] for g in golds])["sari"]
    return scores


def print_metrics_as_latex(metrics):
    order = [
        "rouge1",
        "rouge2",
        "rouge3",
        "rouge4",
        "rougeL",
        "bert_score",
        "bert_score_deberta-large",
        "sari",
        "words",
    ]
    print(" & ".join([f"${metrics[k]:.2f}$" for k in order]))
