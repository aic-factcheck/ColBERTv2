from datetime import datetime
from pathlib import Path

def config():
    NBITS = 2   # encode each dimension with 2 bits
    DOC_MAXLEN = 300   # truncate passages at 300 tokens

    COLBERT_ROOT = Path("/mnt/data/factcheck/fever/data-cs-lrev/colbertv2")
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    
    MODEL_ROOT = Path("/mnt/data/factcheck/fever/data-cs-lrev/colbertv2")
    MODEL_NAME = "bert-base-multilingual-cased"
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_231025_133608/colbert")
    CHECKPOINT = Path(MODEL_ROOT, "checkpoints", CHECKPOINT_NAME)

    INDEX_ROOT = Path(COLBERT_ROOT, "indices", "csfever")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{NBITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    NOTE = "CsFEVER ColBERT index. Model trained on CsFEVER LREV data, Anserini with evidence."

    return {
        "nbits": NBITS, 
        "doc_maxlen": DOC_MAXLEN,
        "collection": COLLECTION,
        "checkpoint": CHECKPOINT,
        "index_name": INDEX_NAME,
        "index_path": INDEX_PATH,
        "note": NOTE,
    }
