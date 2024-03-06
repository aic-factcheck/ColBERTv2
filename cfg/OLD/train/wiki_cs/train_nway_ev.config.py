from datetime import datetime
from pathlib import Path

def config():
    APPROACH = "balanced"

    LANG_SHORT = "cs"
    
    DATE = "20230801"
    WIKI_ROOT = f"/mnt/data/factcheck/wiki/{LANG_SHORT}/{DATE}"

    NER_DIR = "PAV-ner-CNEC"
    QG_DIR = "mt5-large_all-cp126k"
    QACG_DIR = "mt5-large_all-cp156k"

    CLAIM_DIR = Path(NER_DIR, QG_DIR, QACG_DIR)

    COLBERT_ROOT = Path(WIKI_ROOT, "colbertv2", "qacg")
    EXPERIMENT_ROOT = Path(COLBERT_ROOT, "experiments", CLAIM_DIR)

    CHECKPOINT = "bert-base-multilingual-cased"
    # CHECKPOINT = "deepset/xlm-roberta-large-squad2"
    CHECKPOINT_NAME = CHECKPOINT.replace("/", "_")

    NWAY = 32
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    TRIPLES = Path(COLBERT_ROOT, "triples", CLAIM_DIR, f"trn_triples_nway128_evidence+anserini_{APPROACH}.jsonl")
    QUERIES = Path(COLBERT_ROOT, "queries", CLAIM_DIR, f"train_qacg_queries_{APPROACH}.jsonl")
    DEV_TRIPLES = Path(COLBERT_ROOT, "triples", CLAIM_DIR, f"dev_triples_nway128_evidence+anserini_{APPROACH}.jsonl")
    DEV_QUERIES = Path(COLBERT_ROOT, "queries", CLAIM_DIR, f"dev_qacg_queries_{APPROACH}.jsonl")

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    EXPERIMENT = Path(COLBERT_ROOT, "checkpoints", CHECKPOINT_NAME, f"nway{NWAY}_anserini_{APPROACH}_{DATESTR}")

    WANDB_PROJECT="ColBERTv2"
    WANDB_NAME="QACG-CS EV"
    NOTE = "QACG based on Wiki pages. Anserini and evidence. prepare_data_wiki.ipynb"

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
        "eval_bsize": 64,
        "eval_triples": DEV_TRIPLES,
        "eval_queries": DEV_QUERIES,
        "max_eval_triples": 10000,
        "batches_to_eval": 100,
        "early_patience": 5,
        "auto_score": False,
        "wandb_project": WANDB_PROJECT,
        "wandb_name": WANDB_NAME,
        "experiment": EXPERIMENT,
        "note": NOTE,
    }