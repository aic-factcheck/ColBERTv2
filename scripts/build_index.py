import numpy as np
import torch
import json

from collections import defaultdict, OrderedDict, Counter
from dataclasses import dataclass
import datetime as dt
from itertools import chain
import os
import pathlib
from pathlib import Path
import string
import unicodedata as ud
from time import time
from typing import Dict, Type, Callable, List
import sys

from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Collection


def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return Path(cfg["index_path"])
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    NRANKS = torch.cuda.device_count()
    print(f"Detected #GPUs: {NRANKS}")

    collection = Collection(path=str(cfg["collection"]))

    with Run().context(RunConfig(nranks=NRANKS, experiment=str(cfg["index_name"]))):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=cfg["doc_maxlen"], nbits=cfg["nbits"])

        indexer = Indexer(checkpoint=str(cfg["checkpoint"]), config=config)
        indexer.index(name=str(cfg["index_path"]), collection=collection, overwrite=True)


if __name__ == "__main__":
    main()

