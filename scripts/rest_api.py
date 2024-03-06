import argparse
import json
import logging
from flask import Flask, Blueprint
from flask_restx import Api, Resource, fields
from time import time
import ujson

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


api_v1 = Blueprint("api", __name__, url_prefix="/api/v1")

api = Api(
    api_v1,
    version="1.0",
    title="ColBERTv2 API",
    description="Late interaction document retrieval",
)

ns = api.namespace("retrieve", description="retrieve document ids")

parser = api.parser()
parser.add_argument(
    "query", type=str, required=True, help="query for retrieval", location="form"
)
parser.add_argument(
    "k", type=int, required=True, help="number of documents to retrieve", location="form"
)

apimodel = api.model('RetrieveModel', {
    'ids': fields.List(fields.String, description="Retrieved document ids sorted by decreasing score"),
    'scores': fields.List(fields.Float, description="Retrieved document scores"),
    'duration_s': fields.Float(description="Retrieval duration in seconds"),
})


@ns.route("/")
class Retrieve(Resource):
    """Retrieve documents"""

    @api.marshal_with(apimodel, envelope='retrieved')
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        st = time()
        query, k = args["query"], args["k"]
        logger.info(f'RETRIEVE (k={k}): "{query}"')
        ids, scores = retriever.retrieve(query, k)
        duration_s = time()-st
        ret = {"ids": ids, "scores": scores, "duration_s": duration_s}
        return ret, 200

class ColBERTv2Retriever:
    def __init__(self, cfg: dict):
            with Run().context(RunConfig(experiment='REST api')):
                self.searcher = Searcher(index=str(cfg["index_name"]))
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

    app = Flask(__name__)
    app.register_blueprint(api_v1)
    app.run(debug=True, host='0.0.0.0', port=cfg["port"], use_reloader=False)
