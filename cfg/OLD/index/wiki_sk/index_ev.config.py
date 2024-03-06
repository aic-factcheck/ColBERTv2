from datetime import datetime
from pathlib import Path

def config():
    NBITS = 2   # encode each dimension with 2 bits
    DOC_MAXLEN = 300   # truncate passages at 300 tokens

    TRAIN_DATA = "qacg"

    COLBERT_ROOT = Path("/mnt/data/factcheck/wiki/sk/20230801/colbertv2", TRAIN_DATA)
    COLLECTION = Path(COLBERT_ROOT, "collection.jsonl")
    
    MODEL_NAME = "bert-base-multilingual-cased"
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231026_104345/colbert")
    
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_shuf_231026_222231/colbert")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_shuf_231026_222231/colbert-10000")

    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231026_201121/colbert")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231026_201121/colbert-10000")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231026_201121/colbert-BKP")

    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231027_155142/colbert-best_eval")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231027_172701/colbert-best_eval")
    # CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231027_192240/colbert-best_eval")
    CHECKPOINT_NAME = Path(MODEL_NAME, "nway32_anserini_balanced_231027_194555/colbert-best_eval")

    CHECKPOINT = Path(COLBERT_ROOT, "checkpoints", CHECKPOINT_NAME)

    INDEX_ROOT = Path(COLBERT_ROOT, "indices")
    INDEX_NAME = Path(CHECKPOINT_NAME, f"{NBITS}bits")
    INDEX_PATH = Path(INDEX_ROOT, INDEX_NAME)

    NOTE = "QACG-based ColBERT index, Anserini with evidence."

    return {
        "nbits": NBITS, 
        "doc_maxlen": DOC_MAXLEN,
        "collection": COLLECTION,
        "checkpoint": CHECKPOINT,
        "index_name": INDEX_NAME,
        "index_path": INDEX_PATH,
        "note": NOTE,
    }