import numpy as np
import os
import ujson

from functools import partial
from colbert.infra.config.config import ColBERTConfig
from colbert.utils.utils import print_message, zipstar
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
from colbert.evaluation.loaders import load_collection

from colbert.data.collection import Collection
from colbert.data.queries import Queries
from colbert.data.examples import Examples

# from colbert.utils.runs import Run


class EvalBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection):
        self.bsize = config.eval_bsize
        print(f"EvalBatcher: bsize = {self.bsize}")
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist()
        if config.max_eval_triples is not None:
            assert len(self.triples) >= config.max_eval_triples, (len(self.triples), config.max_eval_triples)
            rng = np.random.RandomState(1234)
            rng.shuffle(self.triples)
            self.triples = self.triples[:config.max_eval_triples]
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

        self.auto_score = config.auto_score

        print(f"EvalBatcher: batches = {len(self.triples)/self.bsize}")

        self.first_batch = True
        print(f"EvalBatcher: initialized")

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        # print(f"START EvalBatcher: __next__ --------------------------------------------------------")
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        # print(f"EvalBatcher: offset = {offset}")
        # print(f"EvalBatcher: endpos = {endpos}")
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            # print(f"EvalBatcher: stopping iteration")
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            query, *pids = self.triples[position]
            assert len(pids) >= self.nway, f'EvalBatcher.__next__: Not enough pids ({len(pids)} < {self.nway}) for query at position {position}!' #dhonza
            pids = pids[:self.nway]
            # print(f"LazyBatcher: pids = {pids}")
            # print(f"LazyBatcher: len(pids) = {len(pids)}")

            query = self.queries[query]
            # print(f"LazyBatcher: query = {query}")

            try:
                pids, scores = zipstar(pids)
            except:
                scores = []

            if self.first_batch and self.auto_score:
                scores = [1.0/(i+1) for i in range(len(pids))]

            passages = [self.collection[pid] for pid in pids]

            all_queries.append(query)
            all_passages.extend(passages)
            all_scores.extend(scores)
        
        assert len(all_scores) in [0, len(all_passages)], len(all_scores)

        ret = self.collate(all_queries, all_passages, all_scores)
        if self.first_batch:
            if len(all_scores) > 0:
                print(f'EvalBatcher: size={len(all_queries)}')
                print(f'EvalBatcher: query[0]="{all_queries[0]}", scores[0]={all_scores[0]}')
                self.first_batch = False
        return ret

    def collate(self, queries, passages, scores):
        assert len(passages) == self.nway * self.bsize, (len(queries), len(passages), self.nway, self.bsize)
        return self.tensorize_triples(queries, passages, scores, self.bsize, self.nway)
