from datetime import datetime
from pathlib import Path

def config():
    # TRAIN_DATA = "fever" # for original FEVER splits
    # TRAIN_DATA = "qacg" # for QACG generated splits (Zero-shot-Fact-Verification-by-Claim-Generation, Question Answering for Claim Generation)
    # TRAIN_DATA = "fever+qacg"
    TRAIN_DATA = "qacg-r"

    COLBERT_ROOT = "/mnt/data/factcheck/fever/data_full_nli-filtered-cs/colbertv2"
    EXPERIMENT_ROOT = Path(COLBERT_ROOT, "experiments")

    # CHECKPOINT = "bert-base-uncased"
    CHECKPOINT = "bert-base-multilingual-cased"
    # CHECKPOINT = "deepset/xlm-roberta-large-squad2"
    CHECKPOINT_NAME = CHECKPOINT.replace("/", "_")


    NWAY = 32
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    TRIPLES = Path(COLBERT_ROOT, TRAIN_DATA, "triples", f"train_triples_nway128_anserini.jsonl")
    QUERIES = Path(COLBERT_ROOT, TRAIN_DATA, "queries", "train_anserini_queries.jsonl")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    EXPERIMENT = Path(TRAIN_DATA, CHECKPOINT_NAME, f"nway{NWAY}_anserini_{DATESTR}")

    NOTE = "QACG based on random Wiki pages. prepare_data_fever.ipynb"

    return {
        "bsize": 16,
        "nway": NWAY,
        "use_ib_negatives": False,
        "accumsteps": 8,
        "lr": 3e-06,
        "maxsteps": 500000,
        "root": EXPERIMENT_ROOT,
        "checkpoint": CHECKPOINT,
        "triples": TRIPLES,
        "collection": COLLECTION,
        "queries": QUERIES,
        "experiment": EXPERIMENT,
        "note": NOTE,
    }
