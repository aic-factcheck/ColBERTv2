from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
from pathlib import Path
from pprint import pformat
import shutil
from typing import Callable, Dict, Optional, Union

from aic_nlp_utils.files import create_parent_dir
from aic_nlp_utils.json import read_json, read_jsonl, write_jsonl
from aic_nlp_utils.pycfg import parse_pycfg_args, read_pycfg

from colbert import Searcher
from colbert.infra import Run, RunConfig
from tqdm import tqdm

class ColBERTv2Retriever:
    def __init__(self, index_name, original_id2pid_file):
            with Run().context(RunConfig(experiment='FEVER retrieval')):
                self.searcher = Searcher(index=index_name)
            self.original_id2pid = read_json(original_id2pid_file)
            self.pid2original_id = {v: k for k, v in self.original_id2pid.items()}

    def retrieve(self, query: str, k: int):
        results = self.searcher.search(query, k=k)
        pids, ranks, scores = results
        ids = [self.pid2original_id[pid] for pid in pids]
        return ids, scores

    def retrieve_full(self, query: str, k: int, factor:int = 2, init_factor:int = 4, replace_underscore:Optional[str] = None):
        # retrieves full document ids (dids) if blocks (bids) are indexed
        # if there are multible blocks retrieved, the score is the score of document is computed as max score over blocks
        # `replace_underscore` replaces underscores in bid for other string (most likely space)

        def get_did(pid):
            bid = self.pid2original_id[pid]
            last = bid.rfind("_")
            assert last != -1, "Underscore not found! Did you want to use 'block' level instead?"
            if replace_underscore:
                return bid[:last].replace("_", replace_underscore)
            else:
                return bid[:last]
        
        curk = init_factor * k
        while True:
            pids, ranks, scores = self.searcher.search(query, k=curk)
            dids = [get_did(pid) for pid in pids]
           
            if len(set(dids)) < k:
                curk *= factor
                print(f"extending current k: {curk}")
                continue

            selected = set()
            ids = []
            for did in dids:
                if did in selected:
                    continue
                ids.append(did)
                selected.add(did)

            did2scores = defaultdict(list)
            for pid, score in zip(pids, scores):
                did = get_did(pid)
                did2scores[did].append(score)

            scores = [np.max(did2scores[did]) for did in ids]

            return ids, scores
    

def generate_fever_predictions(claims_jsonl, predictions_jsonl, retriever: ColBERTv2Retriever, k:int=500, level:str="document", replace_underscore:Optional[str] = None):
    assert level in ["document", "block"]
    test_data = read_jsonl(claims_jsonl)
    for r in tqdm(test_data[:]):
        if level == "document":
            ids, scores = retriever.retrieve_full(r["claim"], k=k, replace_underscore=replace_underscore)
        else:
            ids, scores = retriever.retrieve(r["claim"], k=k)
        r["predicted_pages"] = ids
        r["predicted_scores"] = scores
    write_jsonl(predictions_jsonl, test_data, mkdir=True)

def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return cfg["root"]
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    retriever = ColBERTv2Retriever(cfg["index_path"], cfg["id2pid"])
    
    generate_fever_predictions(cfg["claims"], 
                               Path(cfg["root"], "predictions.jsonl"), 
                               retriever, k=cfg["k"], 
                               level=cfg["level"], 
                               replace_underscore=cfg.get("replace_underscore", None))

main()
