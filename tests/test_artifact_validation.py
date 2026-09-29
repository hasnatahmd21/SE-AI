from pathlib import Path

def test_artifact_validator_is_present():
    text=Path('scripts/validate_training_artifact.py').read_text(encoding='utf-8')
    assert 'PeftModel.from_pretrained' in text
    assert 'adapter_model.safetensors' in text