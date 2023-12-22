
import numpy as np
from pathlib import Path
from tqdm import tqdm

from typing import Dict, Type, Callable, List, Optional, Union
import unicodedata

import torch

from pyserini.search import LuceneSearcher
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from aic_nlp_utils.json import read_jsonl, read_json, write_json, write_jsonl, process_to_jsonl
from aic_nlp_utils.encoding import nfc

def generate_original_id2pid_mapping(corpus):
    original_id2pid = {}
    for pid, r in enumerate(corpus):
        original_id = r["id"]
        # assert original_id not in original_id2pid, f"original ID not unique! {original_id}"
        if original_id in original_id2pid:
            print(f"original ID not unique! {pid} {original_id}, previous pid: {original_id2pid[original_id]}")
        original_id2pid[original_id] = pid
    return original_id2pid


def import_qacg_split(split_files: Dict[str, Union[str,Path]]):
    # Imports data for a single split of a Zero-shot dataset based on multiple `split_files`
    # Each split file should correspond to one of the output classes (SUPPORT, REFUTE, NEI).
    data = []
    for label, split_file in split_files.items():
        print(f"reading: {split_file}")
        raw = read_json(split_file)
        ids = list(raw.keys())
        for id_ in tqdm(ids):
            claims = raw[id_]
            for _, claim in claims.items():
                claim_ = nfc(claim)
                if claim != claim_:
                    print("WARN>> claim not NFC, fixing...")
                data.append({"claim": claim_, "label": label, "evidence": [id_]})
    return data


def import_qacg_split_subsample(split_files: Dict[str, Union[str,Path]], subsample:int, seed:int=1234):
    # Imports data for a single split of a Zero-shot dataset.
    # Each split file should correspond to one of the output classes (SUPPORT, REFUTE, NEI).
    # This is meant to get balanced data where each class has exactly `subsample` instances.
    # Total output size is `subsample * len(split_files)`.
    # The claims are selected randomly, it is done by chosing alternatively between all randomly shuffled blocks (paragraphs).
    all_data = []
    for label, split_file in split_files.items():
        rng = np.random.RandomState(seed)
        seed += 1
        data = []
        print(f"reading: {split_file}")
        raw = read_json(split_file)

        ids = list(raw.keys()) # block ids (bids)
        rng.shuffle(ids) # shuffle blocks
        level = 0 # level marks the ith
        while len(data) < subsample:
            once = False
            for id_ in ids:
                claims = list(raw[id_].values())
                if level < len(claims): # are there enough claims for this block?
                    once = True
                    claim = claims[level]
                    claim_ = nfc(claim)
                    if claim != claim_:
                        print("WARN>> claim not NFC, fixing...")
                    data.append({"claim": claim_, "label": label, "evidence": [id_]})
                    if len(data) == subsample:
                        break
            if not once:
                raise ValueError(f"too few claims in data, len(data)= {len(data)}, expected samples = {len(split_files) * subsample}")
            level += 1
        all_data += data
    assert len(all_data) == len(split_files) * subsample, (len(all_data), subsample)
    return all_data


def export_as_anserini_collection(corpus, output_dir):
    data = []
    for r in corpus:
        assert r["id"] == nfc(r["id"])
        assert r["text"] == nfc(r["text"])
        data.append({"id": r["id"], "contents": r["text"]})
    Path(output_dir).mkdir(parents=True)
    write_jsonl(Path(output_dir, "docs.json"), data)


def anserini_retrieve_claims(anserini_index, data, k):
    searcher = LuceneSearcher(anserini_index)
    for r in tqdm(data):
        hits = searcher.search(r["claim"], k)
        r["retrieved"] = []
        for h in hits:
            h = h.docid
            assert h == nfc(h), h
            r["retrieved"].append(h)


def sbert_CE_rerank(data, corpus, out_jsonl, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512):
    id2txt = {doc["id"]: doc["text"] for doc in corpus}

    model = CrossEncoder(model_name, max_length=max_length)

    def process_element(r):
        claim = r["claim"]
        docs = [id2txt[id_] for id_ in r["retrieved"]]
        X = [(claim, doc) for doc in docs]
        scores = model.predict(X)
        idxs = np.argsort(scores, kind='stable')[::-1]
        rec = r.copy()
        rec["retrieved"] = [r["retrieved"][idx] for idx in idxs]
        return rec

    total = len(data) # upper bound for data element count, some may be filtered out...
    # which may happen for exact retrieval like Anserini (typos in claim)
    data = filter(lambda r: len(r["retrieved"])  > 0, data)
    
    process_to_jsonl(data, process_element, out_jsonl, total=total, show_progress=True, bufsize=1000)


def sbert_BI_rerank(data, corpus, out_jsonl, model_name='paraphrase-multilingual-mpnet-base-v2'):

    id2txt = {doc["id"]: doc["text"] for doc in corpus}
    print(f'sbert_BI_rerank>> loading: "{model_name}"')
    model = SentenceTransformer(model_name)

    data = list(filter(lambda r: len(r["retrieved"])  > 0, data))
    print(f"sbert_BI_rerank>> embedding claims")
    claim_embs = model.encode([r["claim"] for r in data], batch_size=128, show_progress_bar=True)

    doc_ids = [id_ for r in data for id_ in r["retrieved"]]
    doc_id_set = set(doc_ids)
    docid2idx = {id_: i for i, id_ in enumerate(doc_id_set)}
    docs = [id2txt[id_] for id_ in docid2idx.keys()]
    print(f"sbert_BI_rerank>> embedding {len(docs)} unique documents (out of {len(doc_id_set)})")
    doc_embs = model.encode(docs, batch_size=128, show_progress_bar=True)

    new_data = []
    for i, r in enumerate(data):
        ce = claim_embs[i]
        doc_idxs = [docid2idx[id_] for id_ in r["retrieved"]] # indices in embedding
        des = doc_embs[doc_idxs]
        scores = util.cos_sim(ce, des)[0,:].detach().cpu().numpy()
        idxs = np.argsort(scores, kind='stable')[::-1]
        # print(r["claim"])
        # print(r["retrieved"])
        rec = r.copy()
        rec["retrieved"] = [r["retrieved"][idx] for idx in idxs]
        # print(rec["retrieved"])
        new_data.append(rec)
    write_jsonl(out_jsonl, new_data)


def generate_triples_by_retrieval(data, corpus, original_id2pid, k, offset=0):
    id2txt = {doc["id"]: doc["text"] for doc in corpus}
    failures = 0
    triples = []
    for qid, r in enumerate(data):
        # those retrieved but not in the annotated evidence will become hard negatives 
        # retrieved = set(r["retrieved"][offset:]).difference(r["evidence"]) # error messes order!
        retrieved = [e for e in r["retrieved"][offset:] if e not in r["evidence"]]
        for pos in r["evidence"]:
            if pos not in id2txt:
                # may happen for EnFEVER when the snapshot does not exactly match 
                failures += 1
                continue
            for neg in retrieved[:k]:
                triples.append((qid, original_id2pid[pos], original_id2pid[neg]))
    print(f"generated {len(triples)} triples with {failures} failures")
    return triples


def generate_triples_by_retrieval_nway(data, corpus, original_id2pid, nway: int, seed=1234, use_evidence=False):
    # sligthly differs from FEVER version where nway was derived from data (but was not ensured to be constant)
    
    id2txt = {doc["id"]: doc["text"] for doc in corpus}
    ids = [doc["id"] for doc in corpus]

    def update_retrieved_by_evidence(retrieved, evidence, claim):
        # if any retrieved document is in the set of evidence, move it to the front, preserving order relative to retrieved
        # example: [x1, x2, A, x3, x4, B, C, x5, x6] -> [A, B, C, D, E, x1, x2, x3, x4, x5, x6] where A, B, C, D, E are documents in evidence set 
        # the rank of D, E documents is arbitrary
        n = len(retrieved)
        if len(evidence) > 0:
            retrieved = retrieved.copy()
            evidence = set(evidence)
            j = 0
            for i in range(len(retrieved)):
                e = retrieved[i]
                if e in evidence:
                    del retrieved[i]
                    retrieved.insert(j, e)
                    j += 1
                    evidence.remove(e)
            # insert pieces evidence not retrieved
            for e in evidence:
                assert e in id2txt, e
                retrieved.insert(j, e)

        retrieved = retrieved[:n] # shorten retrieved to preserve length
        return retrieved

    rng = np.random.RandomState(seed) # just to fix for not enough retrieved documents
    nway_skips = 0
    failures = 0
    triples = []
    for qid, r in enumerate(tqdm(data)):
        # those retrieved but not in the annotated evidence will become hard negatives 
        # retrieved = set(r["retrieved"][offset:]).difference(r["evidence"]) # error messes order!
        retrieved = r["retrieved"]
        # for pos in r["evidence"]:
            # if pos not in id2txt:
            #     # may happen for EnFEVER when the snapshot does not exactly match 
            #     failures += 1
            #     continue

        if use_evidence:
            retrieved = update_retrieved_by_evidence(retrieved, r["evidence"], r["claim"])

        if len(retrieved) < nway:
            nway_skips += 1
            if nway_skips <= 3:
                print(f"WARNING: not enough retrieved documents {nway} => {len(retrieved)} fixing by random")
            elif nway_skips == 4:
                print(f"WARNING: more than 3 occurences of too few retrieved documents, fixing by random...")
            m = nway - len(retrieved) # documents to retrieve
            max_tries = 10
            while max_tries > 0:
                rand_ids = rng.choice(ids, m, replace=False)
                rand_ids_set = set(rand_ids)
                if len(rand_ids_set.intersection(set(retrieved))) == 0:
                    # print(f"pre retrieved {retrieved}")
                    retrieved += list(rand_ids)
                    # print(f"post retrieved {retrieved}")
                    break
                max_tries -= 1
            if max_tries == 0:
                # This should not happen for large corpora
                assert False, f"Too many tries to select random {m} documents from corpus of {len(ids)}."
        examples = []
        for example in retrieved:
            if example not in id2txt:
                failures += 1
                continue
            examples.append(original_id2pid[example])
        triples_lst = [qid] + examples
        triples.append(tuple(triples_lst))
    print(f"generated {len(triples)} triples with {failures} failures and {nway_skips} random fixes")
    if failures > 0:
        print("NOTE that generated triples won't match the queries! This should be fixed for training!")
    return triples