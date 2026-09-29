"""Runtime validation for a completed SE Brain LoRA adapter."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--max-new-tokens', type=int, default=8)
    p.add_argument('--generate', action='store_true')
    args=p.parse_args()
    adapter=args.adapter
    if not adapter.is_dir(): raise SystemExit(f'adapter directory not found: {adapter}')
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (adapter/name).is_file(): raise SystemExit(f'missing adapter file: {name}')
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install requirements-training.txt before adapter validation') from exc
    tokenizer=AutoTokenizer.from_pretrained(adapter, use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(args.base_model)
    model=PeftModel.from_pretrained(model, adapter)
    model.eval()
    result={'adapter':str(adapter.resolve()),'base_model':args.base_model,'load_ok':True,'generated':False}
    if args.generate:
        prompt='Write one concise Python function that returns 2 + 2.'
        inputs=tokenizer(prompt, return_tensors='pt')
        with torch.no_grad(): output=model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        result['generated']=True
        result['output']=tokenizer.decode(output[0], skip_special_tokens=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

if __name__=='__main__': raise SystemExit(main())