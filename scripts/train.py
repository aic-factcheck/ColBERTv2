from argparse import ArgumentParser
from pathlib import Path
from pprint import pformat
import shutil
from typing import Callable, Dict, Optional, Union

import torch
import wandb

from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["root"], cfg["experiment"])
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    NRANKS = torch.cuda.device_count()
    print(f"Detected #GPUs: {NRANKS}")

    with Run().context(RunConfig(nranks=NRANKS, experiment=cfg["experiment"].as_posix(), root=cfg["root"].as_posix())):
        config = ColBERTConfig(
            wandb_name=cfg["wandb_name"],
            wandb_project=cfg["wandb_project"],
            bsize=cfg["bsize"],
            eval_bsize=cfg["eval_bsize"],
            nway=cfg["nway"],
            use_ib_negatives=cfg["use_ib_negatives"],
            accumsteps=cfg["accumsteps"],
            lr=cfg["lr"],
            maxsteps=cfg["maxsteps"],
            similarity=cfg.get("similarity", "cosine"),
            checkpoint=cfg["checkpoint"],
            resume=cfg.get("resume", False),
            max_eval_triples=cfg.get("max_eval_triples"),
            batches_to_eval=cfg["batches_to_eval"],
            early_patience=cfg["early_patience"],
            auto_score=cfg["auto_score"],
            seed=cfg.get("seed", 12345),
        )
        trainer = Trainer(
            triples=cfg["triples"].as_posix(),
            collection=cfg["collection"].as_posix(),
            queries=cfg["queries"].as_posix(),
            eval_triples=cfg["eval_triples"].as_posix(),
            eval_queries=cfg["eval_queries"].as_posix(),
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint=cfg["checkpoint"])

        print(f"Saved checkpoint to {checkpoint_path}...")

if __name__ == "__main__":
    main()