# # Copyright (c) Microsoft Corporation.
# # Licensed under the MIT license.

# import argparse
# import os
# import sys
# import torch
# from pathlib import Path

# from Models.SDNetTrainer import SDNetTrainer
# from Utils.Arguments import Arguments


# def is_file(p: str) -> bool:
#     try:
#         return Path(p).is_file()
#     except Exception:
#         return False


# def resolve_datadir(conf_path: str) -> str:
#     """
#     If conf_file is a file -> datadir is its parent.
#     If conf_file is a directory -> datadir is that directory.
#     """
#     p = Path(conf_path)
#     return str(p.parent if p.is_file() else p)


# def maybe_set_seed(seed: int | None):
#     if seed is None:
#         return
#     import random
#     import numpy as np
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     # more reproducible at slight speed cost
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def parse_args():
#     parser = argparse.ArgumentParser(description="SDNet")
#     parser.add_argument("command", choices=["train"], help="Command to run.")
#     parser.add_argument("conf_file", help="Path to conf file or directory.")

#     # Quality-of-life flags for modern stack
#     parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
#     parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
#     parser.add_argument("--bert_model", type=str, default=None,
#                         help="HF model name or local path for BERT (e.g., 'bert-large-uncased').")
#     parser.add_argument("--hf_cache_dir", type=str, default=None,
#                         help="Optional HuggingFace cache directory.")

#     # passthrough/overrides: any extra unknown args still get forwarded to opt
#     args, unknown = parser.parse_known_args()

#     # preserve unknowns in case your Arguments/Trainer expects them
#     args._unknown = unknown
#     return args


# def main():
#     args = parse_args()

#     # Read config from file (if a folder is provided, your Arguments class should handle it;
#     # otherwise pass the exact file path).
#     conf_args = Arguments(args.conf_file)
#     opt = conf_args.readArguments()

#     # Device selection
#     use_cuda = (torch.cuda.is_available() and not args.cpu)
#     opt["cuda"] = use_cuda

#     # datadir depends on whether conf_file is a file or directory
#     opt["confFile"] = args.conf_file
#     opt["datadir"] = resolve_datadir(args.conf_file)

#     # Optional HF settings/overrides for the modern Bert wrapper
#     if args.bert_model:
#         # The modern Bert wrapper looks for these keys as hints
#         if "BERT_LARGE" in opt:
#             opt["BERT_large_hf_name"] = args.bert_model
#         else:
#             opt["BERT_base_hf_name"] = args.bert_model
#     if args.hf_cache_dir:
#         os.environ["HF_HOME"] = args.hf_cache_dir
#         os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

#     # Reproducibility
#     maybe_set_seed(args.seed)

#     # Fold in any additional CLI overrides (if present)
#     for key, val in vars(args).items():
#         if key in ["command", "conf_file", "_unknown"]:
#             continue
#         if val is not None:
#             opt[key] = val

#     # Also append unknown switches as simple flags (e.g., --FOO) -> opt["FOO"]=True
#     for token in getattr(args, "_unknown", []):
#         if token.startswith("--"):
#             opt[token.lstrip("-")] = True

#     # Echo selection
#     print(f"Select command: {args.command}")
#     print(f"CUDA available: {torch.cuda.is_available()} | Using CUDA: {opt['cuda']}")
#     print(f"Config path: {args.conf_file} | datadir: {opt['datadir']}")

#     # Run
#     trainer = SDNetTrainer(opt)
#     if args.command == "train":
#         trainer.train()
#     else:
#         raise ValueError(f"Unsupported command: {args.command}")


# if __name__ == "__main__":
#     # Ensure project root is on sys.path if running from a subdir
#     sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
#     main()


# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import sys
import torch
from pathlib import Path

from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments


def is_file(p: str) -> bool:
    try:
        return Path(p).is_file()
    except Exception:
        return False


def resolve_datadir(conf_path: str) -> str:
    """
    If conf_file is a file -> datadir is its parent.
    If conf_file is a directory -> datadir is that directory.
    """
    p = Path(conf_path)
    return str(p.parent if p.is_file() else p)


def _clean(val: str) -> str:
    return val.strip().strip('"').strip("'")


def _looks_like_path(p: str) -> bool:
    # Treat it as a filesystem path if it has a slash or a common file extension.
    return ("/" in p or "\\" in p or p.endswith(".txt") or p.endswith(".json"))


def normalize_config_paths(opt: dict) -> dict:
    """
    Make relative file paths absolute (based on opt['datadir']),
    but leave HF model IDs like 'bert-large-uncased' untouched.
    """
    base = opt["datadir"]
    keys = [
        "INIT_WORD_EMBEDDING_FILE",
        "CoQA_TRAIN_FILE",
        "CoQA_DEV_FILE",
        "CoQA_TEST_FILE",

    ]

    for key in keys:
        if key in opt and isinstance(opt[key], str) and opt[key].strip():
            val = _clean(opt[key])
            if _looks_like_path(val):
                val = os.path.abspath(os.path.join(base, val))
            opt[key] = val
    return opt


def maybe_set_seed(seed: int | None):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # more reproducible at slight speed cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="SDNet")

    parser.add_argument("conf_file", help="Path to conf file or directory.")
    parser.add_argument("command", choices=["train","evaluate","test"], help="Command to run.")
    parser.add_argument("--MODEL_PATH", type=str, default=None,
                    help="Path to checkpoint, e.g. coqa/conf~/run_X/best_model.pt")
    parser.add_argument("--EVAL_SPLIT", type=str, choices=["train", "dev", "test"], default="dev",
                    help="Which preprocessed split to evaluate on.")

    # Quality-of-life flags for modern stack
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--bert_model", type=str, default=None,
                        help="HF model name or local path for BERT (e.g., 'bert-large-uncased').")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Optional HuggingFace cache directory.")
    
    # passthrough/overrides: any extra unknown args still get forwarded to opt
    args, unknown = parser.parse_known_args()

    # preserve unknowns in case your Arguments/Trainer expects them
    args._unknown = unknown
    
    return args


def main():
    args = parse_args()

    # Read config
    conf_args = Arguments(args.conf_file)
    opt = conf_args.readArguments()

    # Device selection
    use_cuda = (torch.cuda.is_available() and not args.cpu)
    opt["cuda"] = use_cuda

    # datadir depends on whether conf_file is a file or directory
    opt["confFile"] = args.conf_file
    opt["datadir"] = resolve_datadir(args.conf_file)

    # Normalize config paths (files become absolute; HF ids remain as-is)
    opt = normalize_config_paths(opt)

    # Optional HF settings/overrides for the modern Bert wrapper
    if args.bert_model:
        # The modern Bert wrapper can optionally read these hints
        if "BERT_LARGE" in opt:
            opt["BERT_large_hf_name"] = args.bert_model
        else:
            opt["BERT_base_hf_name"] = args.bert_model
    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    # Reproducibility
    maybe_set_seed(args.seed)

    # Fold in any additional CLI overrides (if present)
    for key, val in vars(args).items():
        if key in ["command", "conf_file", "_unknown"]:
            continue
        if val is not None:
            opt[key] = val




    # Echo selection
    print(f"Select command: {args.command}")
    print(f"CUDA available: {torch.cuda.is_available()} | Using CUDA: {opt['cuda']}")
    print(f"Config path: {args.conf_file} | datadir: {opt['datadir']}")


    trainer = SDNetTrainer(opt)
    if args.command == "train":
        trainer.train()
    elif args.command == "evaluate":
        # expects two env/CLI values:
        #   opt["MODEL_PATH"]  -> path to your checkpoint (e.g., coqa/conf~/run_10/best_model.pt)
        #   opt["EVAL_SPLIT"]  -> which preprocessed split name: 'dev' (default), 'test', or 'train'
        model_path = opt.get("MODEL_PATH") or os.environ.get("MODEL_PATH")
        which = opt.get("EVAL_SPLIT", "dev")
        if not model_path:
            raise ValueError("Please provide MODEL_PATH in conf or as --MODEL_PATH path/to/best_model.pt")
        trainer.evaluate_preprocessed(model_path, which=which)

    
    elif args.command == "test":
        # expects two env/CLI values:
        #   opt["MODEL_PATH"]  -> path to your checkpoint (e.g., coqa/conf~/run_10/best_model.pt)
        #   opt["EVAL_SPLIT"]  -> which preprocessed split name: 'dev' (default), 'test', or 'train'
        model_path = opt.get("MODEL_PATH") or os.environ.get("MODEL_PATH")
        which = opt.get("EVAL_SPLIT", "test")
        if not model_path:
            raise ValueError("Please provide MODEL_PATH in conf or as --MODEL_PATH path/to/best_model.pt")
        trainer.evaluate_preprocessed(model_path, which=which)
    else:
        raise ValueError(f"Unsupported command: {args.command}")



if __name__ == "__main__":
    # Ensure project root is on sys.path if running from a subdir
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    main()
