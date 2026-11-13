# TraceLoom Graph Capture Coverage

Expand graph capture coverage to match realized query-token batches, using stock vLLM configuration.

This is an independently installable **configuration MOD**, implemented with
existing vLLM controls. Installing it changes no engine behavior. TraceLoom
supported the diagnosis and selection of the setting; the package adds a
checked, explicit application of that setting.

## Install and use

```bash
git clone https://github.com/vLLM-HUST/vllm-hust-graph-coverage.git
cd vllm-hust-graph-coverage
python -m pip install .
traceloom-graph-coverage examples/base.json --enable
```

The command prints LLM kwargs; it does not start an engine. In an application:

```python
import json
from traceloom_graph_coverage import configure
from vllm import LLM

base = json.load(open("examples/base.json"))
base["model"] = "/path/to/Qwen3-0.6B"
llm = LLM(**configure(base, enabled=True))
```

`configure(base)` returns an unchanged independent copy. Unsupported qualified
contexts and conflicting settings raise an error before engine construction.
The original dict remains available for rollback. After obtaining a device
allocation, `examples/run.py --model PATH --enable` runs an explicit smoke
example; it is not the historical performance measurement.

## Result and compatibility

At a fixed 32-token step budget, expanding [1,2,4] to [1,2,4,8,16,32] reduced median batch completion by approximately 57.0% in the reference workload. A separate 128-token-budget factorial campaign measured 38.7%; the two campaigns are distinct.

Four simultaneous prompts of 8/48/96/160 tokens, 16 output tokens each; Qwen3-0.6B FP16, TP1, Ascend 910B2. Ten fresh engine processes and 80 unprofiled batches in the 32-token reference campaign; corresponding output tokens matched.

Historical engine revisions: vLLM `0fc695f` and vLLM-Ascend `f4a08bd`,
TorchNPU `2.10.0.post4`. The setting API is checked by this package;
performance on a different engine revision requires a new paired run.
See [evidence](docs/EVIDENCE.md), [protocol](evidence/protocol.json), and
[provenance](docs/PROVENANCE.md).

Larger capture coverage uses more startup time and graph memory. At budget128 the separate campaign observed about +1.30 s capture time and +10 MiB graph memory with compiler caches retained. Measure both startup and service behavior for a new workload.

## Disable and verify

Restart with the original kwargs (`enabled=False`), then optionally run
`python -m pip uninstall vllm-hust-graph-coverage`. No source files are patched.

```bash
python -m pip install '.[test]'
python -m pytest -q
python -m build
```

Maintainer: [Shuhao Zhang](https://github.com/ShuhaoZhangTony).
