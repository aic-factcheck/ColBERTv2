from datetime import datetime
from pathlib import Path

def config():
    # LEVEL = "document"
    LEVEL = "block"

    TEST_NAME = "sum_cs_en_pl_sk"

    FEVER_ROOT = "/mnt/data/factcheck/fever/data-en-lrev/fever-data"
    SPLIT = "paper_test"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = Path("/mnt/data/factcheck/fever/data-en-lrev/colbertv2")

    MODEL_ROOT = Path("/mnt/data/factcheck/wiki/sum_cs_en_pl_sk/20230801/colbertv2/qacg")
    MODEL_NAME = "bert-base-multilingual-cased"
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_shuf_231029_200426/colbert-best_eval")
    CHECKPOINT = Path(MODEL_ROOT, "checkpoints", CHECKPOINT_NAME)

    NBITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices", TEST_NAME)
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{NBITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on EnFEVER for QACG-SUM model."

    return {
        "root": PREDICTIONS_ROOT,
        "index_path": INDEX_PATH,
        "id2pid": ID2PID,
        "split": SPLIT,
        "claims": FEVER_CLAIMS,
        "k": 500,
        "level": LEVEL,
        "replace_underscore": " ", 
        "note": NOTE,
        "date": DATESTR,
    }
