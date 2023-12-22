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


class LazyBatcher():
    def __init__(self, config: ColBERTConfig, triples, queries, collection, rank=0, nranks=1):
        self.bsize, self.accumsteps = config.bsize, config.accumsteps
        self.nway = config.nway

        self.query_tokenizer = QueryTokenizer(config)
        self.doc_tokenizer = DocTokenizer(config)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)
        self.position = 0

        self.triples = Examples.cast(triples, nway=self.nway).tolist(rank, nranks)
        self.queries = Queries.cast(queries)
        self.collection = Collection.cast(collection)

        self.auto_score = config.auto_score

        self.first_batch = True

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.triples)

    def __next__(self):
        # print(f"START LazyBatcher: __next__ --------------------------------------------------------")
        offset, endpos = self.position, min(self.position + self.bsize, len(self.triples))
        # print(f"LazyBatcher: offset = {offset}")
        # print(f"LazyBatcher: endpos = {endpos}")
        self.position = endpos

        if offset + self.bsize > len(self.triples):
            # print(f"LazyBatcher: stopping iteration")
            raise StopIteration

        all_queries, all_passages, all_scores = [], [], []

        for position in range(offset, endpos):
            # print(f"LazyBatcher: position = {position}")
            query, *pids = self.triples[position]
            assert len(pids) >= self.nway, f'LazyBatcher.__next__: Not enough pids ({len(pids)} < {self.nway}) for query at position {position}!' #dhonza
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
                print(f'LazyBatcher: size={len(all_queries)}')
                print(f'LazyBatcher: query[0]="{all_queries[0]}", scores[0]={all_scores[0]}')
                self.first_batch = False

        return ret

    def collate(self, queries, passages, scores):
        assert len(passages) == self.nway * self.bsize, (len(queries), len(passages), self.nway, self.bsize)
        assert self.bsize >= self.accumsteps, f"batch size: {self.bsize} smaller than # of accumulation steps: {self.accumsteps}" #dhonza
        return self.tensorize_triples(queries, passages, scores, self.bsize // self.accumsteps, self.nway)

    # def skip_to_batch(self, batch_idx, intended_batch_size):
    #     Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')
    #     self.position = intended_batch_size * batch_idx
