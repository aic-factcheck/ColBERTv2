from datetime import datetime
from pathlib import Path

def config():
    # LEVEL = "document"
    LEVEL = "block"

    TEST_NAME = "csfever"

    FEVER_ROOT = "/mnt/data/factcheck/fever/data-cs-lrev/fever-data"
    SPLIT = "test_deepl"
    FEVER_CLAIMS = Path(FEVER_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = Path("/mnt/data/factcheck/fever/data-cs-lrev/colbertv2")

    MODEL_ROOT = Path("/mnt/data/factcheck/fever/data-cs-lrev/colbertv2")
    MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway128_anserini_230915_203640/colbert")
    CHECKPOINT = Path(MODEL_ROOT, "checkpoints", CHECKPOINT_NAME)

    NBITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices", TEST_NAME)
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{NBITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"CsFEVER ColBERT. Model trained on CsFEVER LREV data with NWAY 128."


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
