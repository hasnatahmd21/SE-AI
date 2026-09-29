"""Optional Hugging Face backup for a completed SE Brain adapter."""
from __future__ import annotations
import argparse, os
from pathlib import Path

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--repo-id', default=os.getenv('HF_REPO_ID'))
    p.add_argument('--path-in-repo', default=None)
    args=p.parse_args()
    if not args.repo_id: raise SystemExit('HF_REPO_ID or --repo-id is required')
    if not args.adapter.is_dir(): raise SystemExit(f'adapter not found: {args.adapter}')
    token=os.getenv('HF_TOKEN')
    if not token: raise SystemExit('HF_TOKEN is required for Hugging Face upload')
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit('Install huggingface_hub before using Hugging Face backup') from exc
    api=HfApi(token=token)
    path=args.path_in_repo or args.adapter.name
    info=api.upload_folder(folder_path=args.adapter, repo_id=args.repo_id, repo_type='model', path_in_repo=path, commit_message='Upload SE Brain LoRA adapter artifact')
    print(info)
    return 0

if __name__=='__main__': raise SystemExit(main())