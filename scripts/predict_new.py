#!/usr/bin/env python3
# scripts/predict_new.py

import difflib, re
from pick_best_example import infer_substitutions

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Phi-3 uses custom code, so trust_remote_code is recommended
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Causal LMs often have no pad token; use EOS as pad if needed
if TOKENIZER.pad_token is None:
    TOKENIZER.pad_token = TOKENIZER.eos_token

# Use half precision on GPU to save VRAM; full precision on CPU
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,  # stream weights to reduce peak RAM
)

MODEL = MODEL.to(DEVICE)

MODEL.eval()

METRICS_FILE = "work/metrics.csv"
RESIDUAL_FILE = "work/residuals.jsonl"

PREFIX_TEXT = None


def _chat_user(content: str) -> str:
    return f"<|user|>\n{content}<|end|>\n"

def _chat_system(content: str) -> str:
    return f"<|system|>\n{content}<|end|>\n"

def _chat_assistant(content: str) -> str:
    return f"<|assistant|>\n{content}<|end|>\n"

def _chat_assistant_gen() -> str:
    return "<|assistant|>\n"


def build_rule_extraction_prompt(prev_old: str, prev_new: str) -> str:
    system = (
        "You are a substitution rule extractor.\n"
        "Given a BEFORE and AFTER document, output ONLY the substitution rules "
        "that transform BEFORE into AFTER.\n"
        "\n"
        "Requirements:\n"
        "- For each changed phrase, list ALL case variants that appear in the text\n"
        "- NEVER output a rule where before and after are identical\n"
        "- Only include rules evidenced by the diff\n"
        "- Order from most specific (longest) to least specific\n"
        "- Format each rule exactly as: \"BEFORE\" -> \"AFTER\"\n"
        "- Output nothing else — no explanation, no preamble, no BEFORE/AFTER labels\n"
    )

    user = (
        f"BEFORE:\n{prev_old}\n\n"
        f"AFTER:\n{prev_new}\n\n"
        "Rules:"
    )

    return (
        _chat_system(system)
        + _chat_user(user)
        + _chat_assistant_gen()
    )


def parse_rules(llm_output: str) -> list[tuple[str, str]]:
    """
    Parse lines of the form: "BEFORE" -> "AFTER"
    Returns list of (before, after) tuples, longest first.
    """
    rules = []
    for line in llm_output.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            line = line.split('.', 1)[-1].strip()
        m = re.match(r'^"?([^"]+)"?\s*->\s*"?([^"]+)"?$', line)
        if m:
            before = m.group(1).strip()
            after  = m.group(2).strip()
            if before and after and before != after:
                rules.append((before, after))

    # Sort longest source phrase first — prevents partial matches
    # e.g. "Department of Defense" must be applied before "Defense"
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules


def build_rewrite_prompt(rules: list[tuple[str, str]], old: str) -> str:
    rules_str = "\n".join(f'"{a}" -> "{b}"' for a, b in rules)

    system = (
        "You are a deterministic rewrite engine.\n"
        "Apply ONLY the substitutions listed below.\n"
        "Every character not covered by a rule must be copied exactly.\n"
        "DO NOT paraphrase, reorder, summarize, or add content.\n"
        "\n"
        "Substitutions (apply every occurrence, most specific first):\n"
        f"{rules_str}\n"
    )

    return (
        _chat_system(system)
        + _chat_user(old)
        + _chat_assistant_gen()
    )


def apply_rules_deterministic(text: str, rules: list[tuple[str, str]]) -> str:
    """Fast deterministic pre-pass before LLM. Longest rules applied first."""
    for before, after in rules:  # already sorted longest-first
        text = text.replace(before, after)
    return text


def init_prefix_kv(prev_old, prev_new):
    global PREFIX_TEXT

    # Stage 1: extract rules via LLM
    extraction_prompt = build_rule_extraction_prompt(prev_old, prev_new)
    enc = TOKENIZER(extraction_prompt, return_tensors="pt",
                    add_special_tokens=False).to(DEVICE)

    with torch.no_grad():
        out = MODEL.generate(
            **enc,
            max_new_tokens=256,   # rules list is short
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    prompt_len = enc["input_ids"].size(1)
    raw_rules = TOKENIZER.decode(out[0, prompt_len:], skip_special_tokens=True)
    print(f"[Stage 1] Extracted rules:\n{raw_rules}")

    rules = parse_rules(raw_rules)
    print(f"[Stage 1] Parsed {len(rules)} rules: {rules}")

    # Validation: apply rules back to prev_old, check similarity to prev_new
    test_output = apply_rules_deterministic(prev_old, rules)
    sim = difflib.SequenceMatcher(a=test_output, b=prev_new).ratio()
    print(f"[Stage 1] Rule validation similarity: {sim:.3f}")
    if sim < 0.85:
        print("[Stage 1] Warning: low similarity, rules may be incomplete.")
        #print("[Stage 1] Warning: low similarity, rules may be incomplete. "
        #      "Falling back to infer_substitutions.")
        #rules = infer_substitutions(prev_old, prev_new)

    # Stage 2 prompt is built per-document in predict(), using these rules
    if rules:
        PREFIX_TEXT = rules
    else:
        print("[Stage 1] No rules extracted — keeping existing PREFIX_TEXT")
    return rules


# Given old data, predict new
def predict(old, target_len=None):
    if PREFIX_TEXT is None:
        raise RuntimeError("Call init_prefix_kv() first.")

    rules = PREFIX_TEXT  # list of (before, after) tuples

    # Deterministic pre-pass — handles the easy cases instantly, no LLM cost
    pre_applied = apply_rules_deterministic(old, rules)

    # If deterministic pass already produced a perfect result, skip LLM entirely
    # (won't happen often but worth short-circuiting)
    if pre_applied == old and not rules:
        return old

    # Build Stage 2 prompt: LLM only needs to fix what deterministic pass missed
    full_text = build_rewrite_prompt(rules, pre_applied)

    max_ctx = _model_max_ctx()

    # Estimate decode length based on pre_applied (closer to final than raw old)
    if target_len is not None:
        approx = target_len + 64
    else:
        approx = len(TOKENIZER(pre_applied).input_ids) + 64
    max_new = max(32, approx)

    enc = TOKENIZER(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,  # chat tokens already embedded by build_rewrite_prompt
        truncation=True,
        max_length=max_ctx,
    ).to(DEVICE)

    prompt_len = enc["input_ids"].size(1)
    available = max_ctx - prompt_len
    if available <= 0:
        print("Warning: no room left for generation; returning pre-applied deterministic result.")
        return pre_applied  # still useful, better than empty string

    max_new = min(max_new, available)

    with torch.no_grad():
        out = MODEL.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    pred = TOKENIZER.decode(out[0, prompt_len:], skip_special_tokens=True)
    return pred


def _model_max_ctx():
    """Get a reasonable max context length for this model."""
    max_ctx = getattr(MODEL.config, "max_position_embeddings", None)
    if max_ctx is None or max_ctx <= 0:
        max_ctx = getattr(TOKENIZER, "model_max_length", 2048)
    if max_ctx is None or max_ctx <= 0:
        max_ctx = 2048
    return int(max_ctx)
    
    

if __name__ == "__main__":
    raise SystemExit("predict_new.py is a module — import predict() and init_prefix_kv() directly.")