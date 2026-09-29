import json
import pytest
from sebrain.training_config import TrainingConfig
from sebrain.training_engine import TrainingDataError,load_eligible_examples
from sebrain.training_registry import TrainingRunRegistry

def test_training_config_round_trip(tmp_path):
    cfg=TrainingConfig(base_model="example/model"); cfg.write(tmp_path/"config.json")
    assert TrainingConfig.from_file(tmp_path/"config.json").to_dict()==cfg.to_dict()

def test_training_dataset_requires_eligible_record(tmp_path):
    p=tmp_path/"data.jsonl"; p.write_text(json.dumps({"record_id":"x","training_eligible":False,"execution_status":"NOT_EXECUTED","validation_status":"ILLUSTRATIVE"})+"\n")
    with pytest.raises(TrainingDataError,match="no training-eligible"): load_eligible_examples(p)

def test_training_dataset_manifest_is_traceable(tmp_path):
    p=tmp_path/"data.jsonl"; p.write_text(json.dumps({"record_id":"x","training_eligible":True,"execution_status":"EXECUTED","validation_status":"VERIFIED","instruction":"q","output":"a","split":"train"})+"\n")
    rows,manifest=load_eligible_examples(p); assert rows[0]["record_id"]=="x"; assert manifest["eligible_count"]==1; assert len(manifest["source_sha256"])==64

def test_registry_never_overwrites_run(tmp_path):
    registry=TrainingRunRegistry(tmp_path); registry.create("run-1",{"x":1},{"eligible_count":1})
    with pytest.raises(FileExistsError): registry.create("run-1",{"x":2},{"eligible_count":2})
