import argparse
from dataclasses import dataclass, field
from huggingface_hub import snapshot_download

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser("Convenience script for downloading models from HF hub.")

    parser.add_argument('repo_id', type=str, help='Repo id')
    parser.add_argument('--cache_dir', type=str, default='./run_local/test_dir/', help='cache directory')
    parser.add_argument('--revision', type=str, default='main', help='revision')
    parser.add_argument('--allow_regex', type=str, nargs='+', default=None, help='space delim string')
    parser.add_argument('--ignore_regex', type=str, nargs='+', default=None, help='space delim string')

    args = parser.parse_args()
    
    print(args) 
    
    snapshot_download(
        repo_id=args.repo_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
        allow_regex=args.allow_regex,
        ignore_regex=args.ignore_regex
    )
