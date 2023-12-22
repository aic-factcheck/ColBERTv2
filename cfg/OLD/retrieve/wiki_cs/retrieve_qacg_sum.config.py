from datetime import datetime
from pathlib import Path

def config():
    # LEVEL = "document"
    LEVEL = "block"

    TEST_NAME = "qacg"
    SPLIT_ROOT = "/mnt/data/factcheck/wiki/cs/20230801/qacg/splits/PAV-ner-CNEC/mt5-large_all-cp126k/mt5-large_all-cp156k"
    SPLIT = "test_balanced"
    FEVER_CLAIMS = Path(SPLIT_ROOT, f"{SPLIT}.jsonl")

    COLBERT_ROOT = "/mnt/data/factcheck/wiki/cs/20230801/colbertv2/qacg"

    MODEL_NAME = "bert-base-multilingual-cased"
    MODEL_NAME = MODEL_NAME.replace("/", "_")
    CHECKPOINT_NAME = Path("sum_cs_en_pl_sk", MODEL_NAME, "nway32_anserini_balanced_230915_122018/colbert") # QACG

    INDEX_BITS = 2
    INDEX_ROOT = Path(COLBERT_ROOT, "indices")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{INDEX_BITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    ID2PID = Path(COLBERT_ROOT, "original_id2pid.json")

    PREDICTIONS_ROOT = Path(COLBERT_ROOT, "predictions", TEST_NAME, INDEX_NAME)

    DATESTR = datetime.now().strftime("%y%m%d_%H%M%S")
    NOTE = f"Predictions on summed languages QACG test data for QACG trained ColBERTv2."

    return {
        "root": PREDICTIONS_ROOT,
        "index_path": INDEX_PATH,
        "id2pid": ID2PID,
        "split": SPLIT,
        "claims": FEVER_CLAIMS,
        "k": 500,
        "level": LEVEL,
        "note": NOTE,
        "date": DATESTR,
    }
