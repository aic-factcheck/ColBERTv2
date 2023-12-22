from datetime import datetime
from pathlib import Path

def config():
    # TRAIN_DATA = "fever" # for original FEVER splits
    # TRAIN_DATA = "qacg" # for QACG generated splits (Zero-shot-Fact-Verification-by-Claim-Generation, Question Answering for Claim Generation)
    TRAIN_DATA = "fever+qacg"

    COLBERT_ROOT = "/mnt/data/factcheck/fever/data-en-lrev/colbertv2"
    EXPERIMENT_ROOT = Path(COLBERT_ROOT, "experiments")

    CHECKPOINT = "bert-base-uncased"
    # CHECKPOINT = "bert-base-multilingual-cased"
    # CHECKPOINT = "deepset/xlm-roberta-large-squad2"
    CHECKPOINT_NAME = CHECKPOINT.replace("/", "_")

    # CHECKPOINT = "/mnt/data/factcheck/models/ColBERT/colbertv2.0"
    # CHECKPOINT_NAME = "colbertv2.0_msmarco"

    NWAY = 32
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    TRIPLES = Path(COLBERT_ROOT, TRAIN_DATA, "triples", f"train_triples_nway128_anserini+minilm.jsonl")
    QUERIES = Path(COLBERT_ROOT, TRAIN_DATA, "queries", "train_anserini+minilm_queries.jsonl")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    EXPERIMENT = Path(TRAIN_DATA, CHECKPOINT_NAME, f"nway{NWAY}_anserini+minilm_{DATESTR}")

    NOTE = "Combination of FEVER and QACG (generated) data. Data combined in prepare_data_fever.ipynb"

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
