from argparse import ArgumentParser
from collections import defaultdict, Counter
from copy import deepcopy
import numpy as np
from pathlib import Path
from pprint import pformat
import shutil
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from aic_nlp_utils.batch import batch_apply
from aic_nlp_utils.encoding import nfc
from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.json import read_json, write_json, read_jsonl, write_jsonl
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return cfg["dst_predictions_root"]
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    print(f'reading source predictions: {cfg["src_predictions"]}')
    src_predictions = read_jsonl(cfg["src_predictions"])
    print(f'reading filtering predictions: {cfg["filter_predictions"]}')
    filter_predictions = read_jsonl(cfg["filter_predictions"])
    assert len(src_predictions) == len(filter_predictions), (len(src_predictions), len(filter_predictions))

    for k in cfg["k_values"]:
        k = int(k)
        dst_predictions = []
        failure_count = 0
        for s, f in zip(src_predictions, filter_predictions):
            s = deepcopy(s)
            if len(f["predicted_pages"]) < k:
                print(f'warning only {f["predicted_pages"]} for k={k}')
                print(f)
                failure_count += 1
            k_bid_set= set(f["predicted_pages"][:k])
            if "predicted_scores" in s:
                s["predicted_scores"] = [score for bid, score in zip(s["predicted_pages"], s["predicted_scores"]) if bid in k_bid_set]
            s["predicted_pages"] = [bid for bid in s["predicted_pages"] if bid in k_bid_set]
            dst_predictions.append(s)

        out_fname = str(cfg["dst_predictions_prefix"]) + f"_k{k}.jsonl"
        print(f"filtered for k = {k} ({failure_count}/{len(dst_predictions)} failures), saving to: {out_fname}")
        write_jsonl(out_fname, dst_predictions)
            

main()

