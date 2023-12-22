import time
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import wandb

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eval_batcher import EvalBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints

def prepare_evaluate_batches(reader):
    # dhonza
    encodings = []
    target_scores_lst = []
    start_time = time.time()
    for i, BatchSteps in enumerate(reader):
        # if i >= 2:
            # break
        for j, batch in enumerate(BatchSteps):
            try:
                queries, passages, target_scores = batch
                encoding = [queries, passages]
            except: 
                encoding, target_scores = batch
                encoding = [encoding.to(DEVICE)]

            encodings.append(encoding)
            target_scores_lst.append(target_scores)

    print(f"evaluation batch preparation time: {time.time()-start_time}, #encodings: {len(encodings)}")
    return encodings, target_scores_lst

def evaluate_loss(config, encodings, target_scores_lst, colbert, amp, labels):
    # dhonza
    start_time = time.time()
    colbert.eval()
    losses = []
    for encoding, target_scores in zip(encodings, target_scores_lst):
        with amp.context():
            with torch.no_grad():
                scores = colbert(*encoding)
            if config.use_ib_negatives:
                scores, ib_loss = scores

            scores = scores.view(-1, config.nway)

            if len(target_scores) and not config.ignore_scores:
                target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                target_scores = target_scores * config.distillation_alpha
                target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
            else:
                loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

            if config.use_ib_negatives:
                loss += ib_loss

            losses.append(loss.item())
    colbert.train()
    return np.mean(losses), time.time()-start_time

def train(config: ColBERTConfig, 
          triples, queries=None, 
          eval_triples=None, eval_queries=None, 
          collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        wandb.init(name=config.wandb_name, project=config.wandb_project)
        config.help()

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
            print("RerankBatcher initialized")
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
            if config.rank < 1:
                eval_reader = EvalBatcher(config, eval_triples, eval_queries, reader.collection)
            print("LazyBatcher initialized")
    else:
        raise NotImplementedError()

    if not config.reranker:
        print("ColBERT initializing")
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        print("ElectraReranker initializing")
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)
    eval_labels = torch.zeros(config.eval_bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    if config.rank < 1:
        print(f"#> encoding evaluation data")
        eval_encodings, eval_target_scores = prepare_evaluate_batches(eval_reader)
        print(f"eval_target_scores={eval_target_scores}")
        best_loss, secs = evaluate_loss(config, eval_encodings, eval_target_scores, colbert, amp, eval_labels)
        best_batch_idx = 0
        print(f"#> initial evaluation loss={best_loss} in {secs}s")
        wandb.log({"eval_loss": best_loss, "step": 0})

    first_batch = True
    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                    if first_batch: print(f"try branch, len(target_scores)={len(target_scores)}")
                except: 
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]
                    if first_batch:  f"except branch, len(target_scores)={len(target_scores)}"

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)
                if first_batch: print(f"scores.size()={scores.size()}")
                if first_batch and config.use_ib_negatives: print(f"ib_loss={ib_loss}")

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    if first_batch: print(f"KLDivLoss: target_scores.size()={target_scores.size()}, log_scores.size()={log_scores.size()}")
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    if first_batch: print("CrossEntropyLoss")
                    # all labels are zero as the zero-th output is actually the POSITIVE sample
                    # all others are NEGATIVE
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps
                first_batch = False

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            wandb.log({"train_loss": train_loss}, commit=False)
            if (batch_idx + 1) % config.batches_to_eval == 0:
                loss, secs = evaluate_loss(config, eval_encodings, eval_target_scores, colbert, amp, eval_labels)
                print(f"#> evaluation for batch {batch_idx + 1} loss = {loss} in {secs}s")
                wandb.log({"eval_loss": loss}, commit=False)
                batch_patience = config.early_patience * config.batches_to_eval
                if loss < best_loss:
                    best_loss = loss
                    best_batch_idx = batch_idx
                    ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, best_eval=True)
                elif best_batch_idx + batch_patience < batch_idx:
                    print(f"#> EARLY STOPPING for {batch_idx + 1}")
                    ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, early_stop=True)
                    return ckpt_path
            wandb.log({"step": batch_idx+1})
            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.



def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
