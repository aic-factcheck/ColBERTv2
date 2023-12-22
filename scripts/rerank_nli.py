from argparse import ArgumentParser
from collections import defaultdict, Counter
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


from transformers import AutoModelForSequenceClassification, AutoTokenizer

def import_corpus(corpus_file):
    # it already has correct format
    raw = read_jsonl(corpus_file, show_progress=True)
    for e in raw:
        e["id"] = nfc(e["id"])
        e["did"] = nfc(e["did"])
        e["text"] = nfc(e["text"])
    return raw


def generate_original_id2pid_mapping(corpus):
    original_id2pid = {}
    for pid, r in enumerate(corpus):
        original_id = r["id"]
        # assert original_id not in original_id2pid, f"original ID not unique! {original_id}"
        if original_id in original_id2pid:
            print(f"original ID not unique! {pid} {original_id}, previous pid: {original_id2pid[original_id]}")
        original_id2pid[original_id] = pid
    return original_id2pid


def prepare_nli_data(evidence, corpus, original_id2pid, max_k):
    recs = []
    counts = Counter()
    for cid, sample in enumerate(evidence):
        claim = sample["claim"]
        if "id" in sample:
            cid = sample["id"]

        label = sample["label"]
        evidence_bids = sample["predicted_pages"]
        for bid in evidence_bids[:max_k]:
            context = corpus[original_id2pid[bid]]["text"]
            recs.append({"claim": claim, "context": context, "label": label, "claim_id": cid, "bid": bid})
            counts[label] += 1
    print(f"nli samples: {len(recs)}, label counts: {counts}")
    return recs


def split_predict(model, tokenizer, split, batch_size=128, device="cuda", max_length=128, apply_softmax=False):
    def predict(inputs):
        X = tokenizer(inputs, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids = X["input_ids"].to(device)
        attention_mask = X["attention_mask"].to(device)
        with torch.no_grad():
            Y = model(input_ids=input_ids, attention_mask=attention_mask).logits
            return Y
        
    # inputs = [[claim, context] for claim, context in zip(split["claim"],  split["context"])] # for datasets.Dataset
    inputs = [[e["claim"], e["context"]] for e in split] # for list of dicts
    Ys = batch_apply(predict, inputs, batch_size=batch_size, show_progress=True)
    Y = torch.vstack(Ys)
    if apply_softmax:
        Y = F.softmax(Y, dim=1)
    label_rename = {'NOT ENOUGH INFO': 'NEI', 'n': 'NEI', 's': 'SUP', 'r': 'REF', 'SUPPORTS': 'SUP', 'REFUTES': 'REF'}
    id2label = {id_: label_rename.get(label, label) for id_, label in model.config.id2label.items()}
    C = [id2label[id_.item()] for id_ in Y.argmax(dim=1)]
    C = [label_rename.get(c, c) for c in C]
    # T = [l for l in split["label"]] # for datasets.Dataset
    T = [label_rename.get(e["label"], e["label"]) for e in split] # for list of dicts
    Y = Y.detach().cpu().numpy()
    evidence = []
    rank = 1
    last_id = None
    for e, y, c, t in zip(split, Y, C, T):
        r = e.copy()
        if last_id != r["claim_id"]:
            last_id = r["claim_id"]
            rank = 1
        else:
            rank += 1

        r["probs"] = {id2label[id_]: y_ for id_, y_ in enumerate(y)}
        r["pred"] = c
        r["target"] = t
        r['rank'] = rank # original rank assigned by the evidence retrieval
        del r['label']

        evidence.append(r)
    claim2evidence = defaultdict(list)
    for e in evidence:
        claim2evidence[e["claim_id"]].append(e)
    # sort evidence for each claim by decreasing maximum confidence
    # for c, e in claim2evidence.items():
    #     e.sort(key=lambda e: -np.max(e["probs"]))
    return claim2evidence, id2label


def sort_by_nli(claim2evidence, k):
    # sorts by NLI confidence (SUP an REF only)
    claim2evidence_nli_sorted = {}
    for cid, evidence_list in claim2evidence.items():
        evidence_list = evidence_list[:k]
        for e in evidence_list:
            e["max_confidence"] = max(float(e["probs"]['SUP']), float(e["probs"]['REF']))
        evidence_list = sorted(evidence_list, key=lambda e: -e["max_confidence"])
        claim2evidence_nli_sorted[cid] = evidence_list
    return claim2evidence_nli_sorted


def main():
    args = parse_pycfg_args()

    def save_dir_fn(cfg):
        return cfg["dst_predictions_root"]
    
    cfg = read_pycfg(args.pycfg, save_dir_fn=save_dir_fn)

    assert np.max(cfg["k_values"]) <= cfg["max_k"]

    print(f"loading source predictions: {cfg['src_predictions']}")
    src_predictions = read_jsonl(cfg["src_predictions"])

    if Path(cfg["claim2evidence"]).is_file():
        print(f"NLI predictions already computed: {cfg['claim2evidence']}, loading...")
        claim2evidence = read_json(cfg['claim2evidence'])
        # id2label = read_jsonl(cfg['id2label'])
        claim2evidence = {int(k): v for k, v in claim2evidence.items()}

    else:
        print(f'loading corpus: {cfg["corpus_file"]}')
        corpus = import_corpus(cfg["corpus_file"])
        original_id2pid = generate_original_id2pid_mapping(corpus)

        print(f"preparing NLI data")
        nli_data = prepare_nli_data(src_predictions, corpus=corpus, original_id2pid=original_id2pid, k=cfg["max_k"])
        
        model_name = cfg["nli_model_path"]
        print(f"loading NLI model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

        claim2evidence, id2label = split_predict(model, tokenizer, nli_data, batch_size=128, 
                                                 device="cuda", max_length=cfg["nli_max_length"], apply_softmax=True)
        print(f"saving NLI predictions: {cfg['claim2evidence']}")
        write_json(cfg['claim2evidence'], claim2evidence)
        
    for k in cfg["k_values"]:
        out_fname = str(cfg["dst_predictions_prefix"]) + f"_k{k}.jsonl"
        claim2evidence_nli_sorted = sort_by_nli(claim2evidence, k=k)

        # fixing in place!
        for cid, sample in enumerate(src_predictions):
            if "id" in sample:
                cid = sample["id"]
            # cid = str(cid)
            # print(claim2evidence_nli_sorted.keys())
            # print(type(list(claim2evidence_nli_sorted.keys())[0]))
            sample["predicted_pages"] = [p["bid"] for p in claim2evidence_nli_sorted[cid]]
            sample["predicted_scores"] = [p["max_confidence"] for p in claim2evidence_nli_sorted[cid]]

        print(f"sorted for k = {k}, saving to: {out_fname}")
        write_jsonl(out_fname, src_predictions)

main()