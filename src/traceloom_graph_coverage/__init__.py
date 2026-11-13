"""Capture shapes chosen from realized query-token batches. Uses stock vLLM settings; no runtime patch."""
from __future__ import annotations
from copy import deepcopy
import argparse
import json
from pathlib import Path

__version__ = "0.1.0"

def _integer(value, expected, name):
    if type(value) is not int or value not in expected:
        raise ValueError(f"{name} must be one of {sorted(expected)}")

def configure(engine_kwargs, *, enabled=False):
    """Return independent LLM kwargs. Installation and the default are inert."""
    result = deepcopy(engine_kwargs)
    if not enabled:
        return result
    _integer(result.get("max_num_seqs"), {4}, "max_num_seqs")
    if result.get("enforce_eager", False):
        raise ValueError("the qualified recipe requires graph execution")
    if result.get("enable_chunked_prefill") is not True:
        raise ValueError("enable_chunked_prefill=True is required")
    if result.get("enable_prefix_caching") is not False:
        raise ValueError("enable_prefix_caching=False is required")
    _integer(result.get("max_num_batched_tokens"), {32, 128}, "max_num_batched_tokens")
    budget = result["max_num_batched_tokens"]
    sizes = [1 << n for n in range(budget.bit_length())]
    compilation = result.setdefault("compilation_config", {})
    if not isinstance(compilation, dict):
        raise ValueError("compilation_config must be an explicit dict")
    if compilation.get("cudagraph_mode") in (0, "NONE"):
        raise ValueError("graph execution is disabled")
    existing = compilation.get("cudagraph_capture_sizes")
    if existing is not None and existing != sizes:
        raise ValueError("remove the conflicting capture sizes before applying the recipe")
    compilation["cudagraph_capture_sizes"] = sizes
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="JSON object containing existing LLM kwargs")
    p.add_argument("--enable", action="store_true", help="explicitly apply this recipe")
    a = p.parse_args()
    print(json.dumps(configure(json.loads(a.input.read_text()), enabled=a.enable), indent=2))
