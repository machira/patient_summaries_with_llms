#!/usr/bin/env python
"""
End-to-end GPT ablation pipeline that reuses the Guidance-based prompting
infrastructure from the summarization scripts. For each example we:
  1. Generate an initial summary S0.
  2. Detect hallucinated spans in S0.
  3. Produce a corrected summary S1 using the hallucination report.
  4. Run hallucination detection again on S1.

The script records per-example details plus aggregate stats.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import guidance
import yaml
try:
    import tiktoken
    from tiktoken import model as _tiktoken_model

    for _alias in ("gpt-5.1", "gpt-5.1-mini", "gpt-5.1-preview"):
        _tiktoken_model.MODEL_TO_ENCODING.setdefault(_alias, "o200k_base")
        _tiktoken_model.MODEL_PREFIX_TO_ENCODING.setdefault(
            _alias, "o200k_base"
        )
except Exception:
    pass


# ---------------------------------------------------------------------------
# Config / LLM helpers
# ---------------------------------------------------------------------------
def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())

def init_guidance_llm(
    *, model_name: str, max_calls_per_min: int, config: Dict[str, Any]
):
    """Create a Guidance OpenAI/Azure client matching run_summarization.py."""
    common_kwargs = {"max_calls_per_min": max_calls_per_min}
    mode = config.get("openai_api_mode", "openai")

    if mode == "openai":
        os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
        org = config.get("openai_organization")
        llm = guidance.llms.OpenAI(
            model_name,
            organization=org,
            chat_mode=True,
            **common_kwargs,
        )
    elif mode == "azure":
        deployment_id = model_name
        # Historic deployments sometimes rename gpt-3.5 models.
        if model_name == "gpt-3.5-turbo":
            deployment_id = "gpt-35-turbo"
        elif model_name == "gpt-3.5-turbo-16k":
            deployment_id = "gpt-35-turbo-16k"
        llm = guidance.llms.OpenAI(
            model_name,
            api_type="azure",
            api_key=config["azure_api_key"],
            api_base=config["azure_api_base"],
            api_version=config["azure_api_version"],
            deployment_id=deployment_id,
            chat_mode=True,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unsupported openai_api_mode='{mode}'")

    if "gpt-5" in model_name.lower():
        original_caller = llm.caller

        async def caller_with_conversion(**kwargs):
            if "max_tokens" in kwargs:
                value = kwargs.pop("max_tokens")
                # prefer explicit value already provided
                kwargs.setdefault("max_completion_tokens", value)
            if "stop" in kwargs:
                kwargs.pop("stop")
            return await original_caller(**kwargs)

        llm.caller = caller_with_conversion

    return llm


def _token_kw_for_model(model_name: str) -> str:
    return "max_completion_tokens" if "gpt-5" in model_name.lower() else "max_tokens"


# ---------------------------------------------------------------------------
# Guidance prompts
# ---------------------------------------------------------------------------
SUMMARY_PROMPT_TEMPLATE = """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given some doctor's notes and you will need to summarize the patient's brief hospital course in one paragraph. Please only include key events and findings and avoid using medical jargons, and you MUST start the summary with "You were admitted".

{{#if icl_examples}}
Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

SUMMARY: 
{{this.summary}}
{{/each}}
{{/if}}

DOCUMENT: {{final_text}}
{{~/user}}

{{#assistant~}}
{{gen 'summary' __MAX_TOKENS__=600 temperature=0}}
{{~/assistant}}
"""

HALLUCINATION_PROMPT_TEMPLATE = """
{{#system~}}
You are a helpful assistant that helps patients understand their medical records.
{{~/system}}

{{#user~}}
You will be given a doctor's notes and a summary with potentially incorrectness. Your task is to identify spans with erroneous, contradictory, or unsupported facts in the summary, and label them using the <error> tag (e.g. <error>incorrect fact</error>). There could be more than one error in the summary. 

### Allowed General Medical Knowledge and Advice
We allow general medical knowledge or advice that patients commonly receive (for example, “take your medications as prescribed” or “call your doctor if symptoms worsen”). Do not flag those statements as errors even if the doctor's note does not explicitly state them, unless they contradict the information provided in the document.

{{#if icl_examples}}
Here are some examples:

{{#each icl_examples}}
DOCUMENT: 
{{this.text}}

ORIGINAL SUMMARY: 
{{this.summary}}

SUMMARY WITH LABELED ERRORS:
{{this.labeled_summary}}
{{/each}}
{{/if}}
 

Can you identify the errors for the following document and summary?
DOCUMENT: 
{{final_text}}

ORIGINAL SUMMARY: 
{{final_summary}}

SUMMARY WITH LABELED ERRORS:
{{~/user}}

{{#assistant~}}
{{gen 'labeled_errors' __MAX_TOKENS__=600 temperature=0}}
{{~/assistant}}
"""

CORRECTION_PROMPT_TEMPLATE = """
{{#system~}}
You correct clinical summaries so that they only contain facts supported by
the provided context.
{{~/system}}

{{#user~}}
CONTEXT:
{{context}}

INITIAL SUMMARY (S0):
{{summary}}

HALLUCINATION SPANS:
{{hallucination_text}}

Write a corrected summary (S1) that removes or fixes unsupported statements
while preserving accurate details.
{{~/user}}

{{#assistant~}}
{{gen 'summary' __MAX_TOKENS__=600 temperature=0}}
{{~/assistant}}
"""

def build_program(template: str, token_kw: str, **kwargs):
    prompt = template.replace("__MAX_TOKENS__", token_kw)
    return guidance(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ERROR_SPAN_RE = re.compile(r"<error>(.*?)</error>", re.DOTALL)


def extract_error_spans(labeled_summary: str) -> List[str]:
    spans = []
    for match in ERROR_SPAN_RE.findall(labeled_summary):
        span = match.strip()
        if span:
            spans.append(span)
    return spans


def safe_gen(program, *, llm, **kwargs) -> Optional[Any]:
    try:
        return program(llm=llm, **kwargs)
    except Exception as exc:  # pragma: no cover - guidance surfaces many errors
        print(f"Generation failed: {exc}")
        return None


def generate_initial_summary(program, llm, context: str) -> str:
    result = safe_gen(program, llm=llm, final_text=context)
    if result is None:
        return ""
    return result.get("summary", "").strip()


def detect_hallucinations(program, llm, context: str, summary: str) -> List[str]:
    result = safe_gen(
        program,
        llm=llm,
        final_text=context,
        final_summary=summary,
        icl_examples=[],
    )
    if result is None:
        return []
    labeled = result.get("labeled_errors", "").strip()
    return extract_error_spans(labeled)


def format_spans_for_prompt(spans: List[str]) -> str:
    if not spans:
        return "None detected."
    return "\n".join(f"- {s}" for s in spans)


def correct_summary(
    program, llm, context: str, summary: str, hallucinations: List[str]
) -> str:
    result = safe_gen(
        program,
        llm=llm,
        context=context,
        summary=summary,
        hallucination_text=format_spans_for_prompt(hallucinations),
    )
    if result is None:
        return ""
    return result.get("summary", "").strip()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_ablation(args: argparse.Namespace):
    config = load_config(Path(args.config))
    summary_program = build_program(
        SUMMARY_PROMPT_TEMPLATE,
        _token_kw_for_model(args.model_name),
        icl_examples=[],
    )
    correction_program = build_program(
        CORRECTION_PROMPT_TEMPLATE,
        _token_kw_for_model(args.model_name),
    )
    hallucination_program = build_program(
        HALLUCINATION_PROMPT_TEMPLATE,
        _token_kw_for_model(args.hallucination_model),
        icl_examples=[],
    )
    summary_llm = init_guidance_llm(
        model_name=args.model_name,
        max_calls_per_min=args.max_calls_per_min,
        config=config,
    )
    if args.hallucination_model == args.model_name:
        hallucination_llm = summary_llm
    else:
        hallucination_llm = init_guidance_llm(
            model_name=args.hallucination_model,
            max_calls_per_min=args.max_calls_per_min,
            config=config,
        )

    # Load evaluation examples.
    examples = []
    with open(args.examples_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if args.num_examples and len(examples) >= args.num_examples:
                break

    results = []
    for idx, ex in enumerate(examples, 1):
        ctx = ex.get("text") or ex.get("context")
        if ctx is None:
            raise ValueError("Each example must include a 'text' field.")
        ex_id = ex.get("id", str(idx))
        print(f"\n=== Example {idx}/{len(examples)} (id={ex_id}) ===")
        print(" - Generating initial summary (S0)")
        s0 = generate_initial_summary(summary_program, summary_llm, ctx)
        print(" - Detecting hallucinations in S0")
        h0 = detect_hallucinations(hallucination_program, hallucination_llm, ctx, s0)
        print(" - Correcting summary (S1)")
        s1 = correct_summary(correction_program, summary_llm, ctx, s0, h0)
        print(" - Detecting hallucinations in S1")
        h1 = detect_hallucinations(hallucination_program, hallucination_llm, ctx, s1)

        results.append(
            {
                "id": ex_id,
                "text": ctx,
                "summary_0": s0,
                "summary_1": s1,
                "hallucinations_0": h0,
                "hallucinations_1": h1,
                "delta": len(h1) - len(h0),
            }
        )

    total = len(results)
    improved = sum(1 for r in results if r["delta"] < 0)
    worsened = sum(1 for r in results if r["delta"] > 0)
    unchanged = total - improved - worsened
    avg_h0 = (
        sum(len(r["hallucinations_0"]) for r in results) / total if total else 0
    )
    avg_h1 = (
        sum(len(r["hallucinations_1"]) for r in results) / total if total else 0
    )
    avg_delta = sum(r["delta"] for r in results) / total if total else 0

    print("\n=== Ablation Summary ===")
    print(f"Total examples: {total}")
    print(f"Improved (delta < 0): {improved}")
    print(f"Unchanged: {unchanged}")
    print(f"Worsened (delta > 0): {worsened}")
    print(f"Avg hallucinations in S0: {avg_h0:.2f}")
    print(f"Avg hallucinations in S1: {avg_h1:.2f}")
    print(f"Avg delta (H1 - H0): {avg_delta:.2f}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Saved per-example results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT ablation pipeline")
    parser.add_argument(
        "--examples_file", type=str, required=True, help="Path to examples.jsonl"
        )
    parser.add_argument(
        "--config",
        type=str,
        default="gpt-4/config.yaml",
        help="OpenAI/Azure credential config",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model for generation/correction",
    )
    parser.add_argument(
        "--hallucination_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for hallucination detection",
    )
    parser.add_argument(
        "--max_calls_per_min", type=int, default=3, help="API rate limit"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Optional cap on processed examples",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gpt-4/summarization_results/ablation_results.json",
    )
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
