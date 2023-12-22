import argparse
import json
import logging
from time import time
import ujson

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColBERTv2Retriever:
    def __init__(self, index_name, fever_id2pid):
            with Run().context(RunConfig(experiment='FEVER predictions')):
                self.searcher = Searcher(index=index_name)
            with open(cfg["lineno2id_mapping"], "r") as f:
                 self.lineno2id = ujson.load(f)
                 self.lineno2id = {int(k): v for k, v in self.lineno2id.items()}
                 assert len(self.lineno2id) == len(self.searcher.collection.data), f"not maching collection size: {len(self.lineno2id)} != {len(self.searcher.collection.data)}"

    def retrieve(self, query: str, k: int):
        results = self.searcher.search(query, k=k)
        pids, ranks, scores = results
        ids = [self.lineno2id[pid] for pid in pids]
        return ids, scores

if __name__ == "__main__":
    cparser = argparse.ArgumentParser()
    cparser.add_argument(
        'cfgfile', help="JSON configuration file, such as cfg/colbertv2_wiki_en_20230220.json")
    # cparser.add_argument('--device', default='cpu',
                        #  choices=["cpu", "cuda"], help="target device CPU or CUDA", type=str)
    cargs = cparser.parse_args()
    
    with open(cargs.cfgfile, "r") as f:
        cfg = json.load(f)
    
    logger.info(json.dumps(cfg, indent=4))

    retriever = ColBERTv2Retriever(cfg)

