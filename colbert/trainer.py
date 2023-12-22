from colbert.infra.run import Run
from colbert.infra.launcher import Launcher
from colbert.infra.config import ColBERTConfig, RunConfig

from colbert.training.training import train


class Trainer:
    def __init__(self, triples, queries, eval_triples, eval_queries, collection, config=None):
        print(f"Trainer Run().config:\n{Run().config}")
        self.config = ColBERTConfig.from_existing(config, Run().config)
        print(f"Trainer self.config:\n{self.config}")

        self.triples = triples
        self.queries = queries
        self.eval_triples = eval_triples
        self.eval_queries = eval_queries
        self.collection = collection

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def train(self, checkpoint='bert-base-uncased'):
        """
            Note that config.checkpoint is ignored. Only the supplied checkpoint here is used.
        """

        # Resources don't come from the config object. They come from the input parameters.
        # TODO: After the API stabilizes, make this "self.config.assign()" to emphasize this distinction.
        self.configure(triples=self.triples, queries=self.queries, 
                       eval_triples=self.eval_triples, eval_queries=self.eval_queries, 
                       collection=self.collection)
        self.configure(checkpoint=checkpoint)

        launcher = Launcher(train)

        self._best_checkpoint_path = launcher.launch(self.config, 
                                                     self.triples, self.queries, 
                                                     self.eval_triples, self.eval_queries, 
                                                     self.collection)

    def best_checkpoint_path(self):
        return self._best_checkpoint_path

