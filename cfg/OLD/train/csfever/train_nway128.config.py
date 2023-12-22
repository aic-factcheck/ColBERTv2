from datetime import datetime
from pathlib import Path

def config():
    FEVER_ROOT = "/mnt/data/factcheck/fever/data-cs-lrev"
    COLBERT_ROOT = f"{FEVER_ROOT}/colbertv2"

    QUERIES_ROOT = Path(COLBERT_ROOT, "queries")
    TRIPLES_ROOT = Path(COLBERT_ROOT, "triples")

    EXPERIMENT_ROOT = Path(COLBERT_ROOT, "experiments")


    CHECKPOINT = "bert-base-multilingual-cased"
    # CHECKPOINT = "deepset/xlm-roberta-large-squad2"
    CHECKPOINT_NAME = CHECKPOINT.replace("/", "_")

    NWAY = 128
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    TRIPLES = Path(TRIPLES_ROOT, f"train_triples_nway128_anserini.jsonl")
    QUERIES = Path(QUERIES_ROOT, f"train_anserini_queries.jsonl")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    EXPERIMENT = Path(COLBERT_ROOT, "checkpoints", CHECKPOINT_NAME, f"nway{NWAY}_anserini_{DATESTR}")

    NOTE = "LREV CsFEVER LREV. Maximum nway (128)."

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
