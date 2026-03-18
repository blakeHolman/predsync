#!/usr/bin/env python3
# scripts/pick_best_example.py

# Responsible for:
#   - Scoring (old, new) pairs as exemplar candidates
#   - Reading and writing prompt.json (rules, rules_hash, rules_score, rules_version)
#   - Deciding whether predict_new needs to extract new rules
#   - Applying server rule updates to prompt.json

import json
import difflib
import hashlib
import re
from math import log1p
from collections import Counter
from pathlib import Path
from collections import Counter

# ----- Constants -----

PROMPT_JSON = "data/prompt.json"
VERSIONS_JSON = "data/versions.json"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]|\s+", re.UNICODE)

# ----- I/O -----

def _empty_prompt():
    """Canonical empty state — used by initialize.py to create prompt.json."""
    return {
        "rules_hash":    None,
        "rules_score":   0.0,
        "rules_version": 0,
        "rules":         [],
    }


def _load_prompt(prompt_path):
    """
    Load prompt.json.

    initialize.py guarantees this file exists before any client or server
    starts, so a missing file is always an error here.

    Raises:
        OSError:    if the file cannot be read.
        ValueError: if the JSON is malformed or a required field is missing.
    """
    try:
        text = Path(prompt_path).read_text(encoding="utf-8")
    except OSError as e:
        raise OSError(f"[pick_best] could not read {prompt_path}: {e}") from e

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"[pick_best] malformed prompt.json at {prompt_path}: {e}") from e


def _save_prompt(data, prompt_path):
    """Write prompt.json. initialize.py guarantees the directory exists."""
    Path(prompt_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _hash_rules(rules: list):
    """SHA-256 of JSON-serialized rules"""
    content = json.dumps(rules)
    return hashlib.sha256(content.encode()).hexdigest() #full hash


def bump_version(chunk_id: str, versions_path: str = VERSIONS_JSON) -> int:
    """
    Increment the version counter for chunk_id in versions.json.
    Creates the file if missing. Returns the new version number.
    """
    path = Path(versions_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"chunks": {}}
    else:
        data = {"chunks": {}}
 
    current = data["chunks"].get(chunk_id, 0)
    data["chunks"][chunk_id] = current + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data["chunks"][chunk_id]
 
 
def get_version(chunk_id: str, versions_path: str = VERSIONS_JSON) -> int:
    """Return the current version of chunk_id, or 0 if not yet tracked."""
    path = Path(versions_path)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("chunks", {}).get(chunk_id, 0)
    except json.JSONDecodeError:
        return 0

# ----- Scoring and exemplar selection ------

def _substitution_coverage_score(old, new):
    """
    Reward exemplars that have both long-form and short-form (acronym) changes.
    High variance in changed span lengths = good exemplar.
    """
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    changed_spans = [
        old[i1:i2]
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag == "replace" and i2 - i1 > 1
    ]
    if not changed_spans:
        return 0.0

    lengths = [len(s) for s in changed_spans]
    avg_len = sum(lengths) / len(lengths)
    diversity = len(set(changed_spans))

    # Variance in lengths rewards having both short (acronyms) and long (full phrases)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

    return diversity * (1.0 + variance ** 0.5)


def score_example(old, new, tokenizer=None):
    """
    Score an (old, new) pair as a rule-extraction exemplar.

    Higher = better teaching example for the LLM.
    Uses only stdlib — no LLM, no file I/O.

    Args:
        old:       Source text.
        new:       Target text.
        tokenizer: Optional HuggingFace tokenizer. If provided, size is
                   computed as token count (more accurate for LLM context
                   budgeting). Falls back to character count if None.

    Returns:
        Float score >= 0.0.
    """
    if tokenizer is not None:
        size = len(tokenizer(old + "\n" + new).input_ids)
    else:
        size = len(old) + len(new)

    if size <= 0:
        return 0.0

    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    change_ratio = 1.0 - sm.ratio()
    coverage = _substitution_coverage_score(old, new)

    return (change_ratio * coverage) / log1p(size)


# ----- Main interface -----

def check(old, new, tokenizer=None, prompt_path = PROMPT_JSON):
    """
    Score (old, new) and decide whether new rule extraction is needed.

    Reads current rules_score from prompt.json. If the candidate scores
    higher, signals predict_new to extract new rules. Otherwise returns
    the current rules as-is.

    Args:
        old:          Source text of the edited chunk.
        new:          Target text of the edited chunk.
        tokenizer:    Optional HuggingFace tokenizer for accurate size scoring.
        prompt_path:  Path to prompt.json.

    Returns:
        (rules, needs_extraction, candidate_score) where:
          - rules is the current best rules list (may be empty on first run)
          - needs_extraction is True if predict_new should run init_prefix_kv
            and then call save_rules() with the result
          - candidate_score is the score of this (old, new) pair, to be passed
            directly to save_rules() after init_prefix_kv so it isn't recomputed
    """
    prompt = _load_prompt(prompt_path)
    current_score = prompt["rules_score"]
    current_rules = prompt["rules"]

    candidate_score = score_example(old, new, tokenizer)
    needs_extraction = candidate_score > current_score or not current_rules

    if needs_extraction:
        print(f"[pick_best] candidate_score={candidate_score:.4f} > "
              f"current={current_score:.4f} — extraction needed")
    else:
        print(f"[pick_best] candidate_score={candidate_score:.4f} <= "
              f"current={current_score:.4f} — keeping current rules")

    return current_rules, needs_extraction, candidate_score


def save_rules(rules, rules_score, prompt_path = PROMPT_JSON):
    """
    Persist newly extracted rules to prompt.json.

    Called by predict_new after init_prefix_kv produces a new rule set.
    Computes hash and increments version automatically.

    Args:
        rules:        List of (before, after) tuples from rule extraction.
        rules_score:  Exemplar score that triggered this extraction.
        prompt_path:  Path to prompt.json.

    Returns:
        Updated prompt dict (rules, rules_hash, rules_score, rules_version).
    """
    prompt = _load_prompt(prompt_path)

    updated = {
        "rules_hash":    _hash_rules(rules),
        "rules_score":   rules_score,
        "rules_version": prompt["rules_version"] + 1,
        "rules":         rules,
    }

    _save_prompt(updated, prompt_path)
    print(f"[pick_best] saved rules v{updated['rules_version']} "
          f"hash={updated['rules_hash']} score={rules_score:.4f}")
    return updated


def update_from_server(rules, rules_hash, rules_score, rules_version, prompt_path = PROMPT_JSON):
    """
    Overwrite local prompt.json with rules received from the server.

    Called by predict_new when transport.verify_rules() returns better
    rules from the server. Trusts the server's hash, score, and version
    without recomputing.

    Args:
        rules:         Rule list from server.
        rules_hash:    Hash as reported by server.
        rules_score:   Score as reported by server.
        rules_version: Version as reported by server.
        prompt_path:   Path to prompt.json.
    """
    updated = {
        "rules_hash":    rules_hash,
        "rules_score":   rules_score,
        "rules_version": rules_version,
        "rules":         rules,
    }
    _save_prompt(updated, prompt_path)
    print(f"[pick_best] adopted server rules v{rules_version} "
          f"hash={rules_hash} score={rules_score:.4f}")


def load_rules(prompt_path = PROMPT_JSON):
    """
    Load current rules state from prompt.json.

    Convenience for predict_new to read rules/hash/score/version
    without going through check().

    Returns:
        Dict with rules, rules_hash, rules_score, rules_version.
    """
    return _load_prompt(prompt_path)


# ----- Example inference logic (not currently used) -----

def infer_substitutions(prev_old, prev_new, top_k = 20, max_span_tokens = 6, min_len_chars = 2):
    """
    Infer likely substitution pairs from one exemplar (prev_old -> prev_new).
    Used as fallback in predict_new when LLM extraction similarity is too low.

    Returns a list of (before, after) tuples, most-frequent first.
    """
    a = _tokenize_preserve(prev_old)
    b = _tokenize_preserve(prev_new)

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)

    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue

        span_a = a[i1:i2]
        span_b = b[j1:j2]

        span_a_nw = [t for t in span_a if _is_meaningful_token(t)]
        span_b_nw = [t for t in span_b if _is_meaningful_token(t)]

        if not span_a_nw or not span_b_nw:
            continue
        if len(span_a_nw) > max_span_tokens or len(span_b_nw) > max_span_tokens:
            continue

        phrase_a = "".join(span_a_nw).strip()
        phrase_b = "".join(span_b_nw).strip()

        # Filter out tiny / single-char / trivial substitutions like "D" -> "W"
        if len(phrase_a) < min_len_chars or len(phrase_b) < min_len_chars:
            continue
        if len(phrase_a) == 1 or len(phrase_b) == 1:
            continue

        pairs.append((phrase_a, phrase_b))

    ctr = Counter(pairs)
    return [p for p, _ in ctr.most_common(top_k)]

# ----- Helpers (not used) -----

def _tokenize_preserve(text):
    """
    Tokenize into: words/numbers, punctuation, and whitespace tokens.
    Keeping whitespace tokens lets the matcher align better, but we’ll filter
    them out when building substitution candidates.
    """
    return _TOKEN_RE.findall(text)

def _is_meaningful_token(tok):
    # reject whitespace-only
    if tok.isspace():
        return False
    # reject pure punctuation (single punctuation token)
    if re.fullmatch(r"[^\w\s]+", tok):
        return False
    return True