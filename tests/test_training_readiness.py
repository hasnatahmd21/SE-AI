import json

from sebrain.training_config import TrainingConfig


def test_smoke_config_is_real_and_valid():
    data = json.loads(open("configs/training.smoke.json", encoding="utf-8").read())
    cfg = TrainingConfig.from_dict(data)
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B"
    assert cfg.max_steps == 2
    assert cfg.lora.r == 8
