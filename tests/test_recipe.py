import json
from pathlib import Path
import pytest
from traceloom_graph_coverage import configure

def base():
    return json.loads((Path(__file__).parents[1]/"examples/base.json").read_text())

def test_disabled_is_inert_and_copy_is_independent():
    x=base();y=configure(x)
    assert y==x and y is not x
    y.setdefault("compilation_config",{})["test"]=1
    assert "test" not in x.get("compilation_config",{})

@pytest.mark.parametrize("field,value",[("max_num_seqs",8),("enforce_eager",True),("enable_chunked_prefill",False),("enable_prefix_caching",True)])
def test_incompatible_mode(field,value):
    x=base();x[field]=value
    with pytest.raises(ValueError):configure(x,enabled=True)

def test_enabled_changes_capture_only():
    x=base(); y=configure(x,enabled=True)
    assert y.pop("compilation_config")=={"cudagraph_capture_sizes":[1,2,4,8,16,32]}
    assert y==x
    x["max_num_batched_tokens"]=128
    assert configure(x,enabled=True)["compilation_config"]["cudagraph_capture_sizes"][-1]==128

def test_conflict_and_nonqualified_budget():
    x=base();x["max_num_batched_tokens"]=64
    with pytest.raises(ValueError):configure(x,enabled=True)
    x=base();x["compilation_config"]={"cudagraph_capture_sizes":[1,2,4]}
    with pytest.raises(ValueError):configure(x,enabled=True)

