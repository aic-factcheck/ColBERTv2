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

    NWAY = 32
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    TRIPLES = Path(TRIPLES_ROOT, f"train_triples_nway128_evidence+anserini.jsonl")
    QUERIES = Path(QUERIES_ROOT, f"train_anserini_queries.jsonl")
    DEV_TRIPLES = Path(TRIPLES_ROOT, f"dev_triples_nway128_evidence+anserini.jsonl")
    DEV_QUERIES = Path(QUERIES_ROOT, f"dev_anserini_queries.jsonl")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    EXPERIMENT = Path(COLBERT_ROOT, "checkpoints", CHECKPOINT_NAME, f"nway{NWAY}_anserini_{DATESTR}")

    WANDB_PROJECT="ColBERTv2"
    WANDB_NAME="CsFEVER EV IBN"
    NOTE = "LREV CsFEVER. Anserini and evidence."

    return {
        "bsize": 16,
        "nway": NWAY,
        "use_ib_negatives": True,
        "accumsteps": 8,
        "lr": 3e-06,
        "maxsteps": 500000,
        "root": EXPERIMENT_ROOT,
        "checkpoint": CHECKPOINT,
        "triples": TRIPLES,
        "collection": COLLECTION,
        "queries": QUERIES,
        "eval_bsize": 64,
        "eval_triples": DEV_TRIPLES,
        "eval_queries": DEV_QUERIES,
        "batches_to_eval": 100,
        "early_patience": 20,
        "auto_score": False,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": WANDB_NAME,
        "experiment": EXPERIMENT,
        "note": NOTE,
    }
