# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from command import configs
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MemDepConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="mem_dep",
        metadata={"help": "name of the classification head to use"},
    )


@register_criterion('mem_dep', dataclass=MemDepConfig)
class MemDepCriterion(FairseqCriterion):

    def __init__(self, cfg: MemDepConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.fields = configs.fields

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
                hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads
        ), 'model must provide classification head for --criterion=mem_dep'

        inst_pair = sample['inst_pair']
        inst1_idx = inst_pair.eq(self.task.inst_pair_dictionary.index('1'))
        inst2_idx = inst_pair.eq(self.task.inst_pair_dictionary.index('2'))

        func_embedding = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=None
        )[0]

        # pool the embedding
        inst1_embedding_mean = torch.mean(func_embedding * inst1_idx.unsqueeze(-1), dim=1)
        inst2_embedding_mean = torch.mean(func_embedding * inst2_idx.unsqueeze(-1), dim=1)

        # concatenate mean(func), u, v, |u-v|, u*v
        concat_in = torch.cat((torch.mean(func_embedding, dim=1),
                               inst1_embedding_mean,
                               inst2_embedding_mean,
                               torch.abs(inst1_embedding_mean - inst2_embedding_mean),
                               inst1_embedding_mean * inst2_embedding_mean),
                              dim=-1)

        # concat_in = [func_embedding[:, 0, :], inst1_embedding_mean, inst2_embedding_mean]

        # predict
        logits = model.classification_heads[self.classification_head_name](concat_in)

        # target, 0 - not memory dependent, 1 - may alias
        targets = model.get_targets(sample, [func_embedding]).view(-1)

        sample_size = targets.numel()

        # loss = F.cosine_embedding_loss(caller_embedding, callee_embedding, targets,
        #                                margin=configs.cosine_embedding_loss_margin,
        #                                reduction='sum')

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        # preds = torch.cosine_similarity(caller_embedding, callee_embedding, dim=1)
        # preds_positive = preds > configs.cosine_embedding_loss_margin
        # targets_positive = targets > configs.cosine_embedding_loss_margin
        #
        # logging_output['ncorrect'] = ((preds_positive == targets_positive) * targets_positive).sum().item()
        # logging_output['ncorrect_total'] = (preds_positive == targets_positive).sum().item()
        # logging_output['mem_dep_pred'] = preds_positive.sum().item()
        # logging_output['mem_dep'] = targets_positive.sum().item()
        # print(logging_output['ncorrect'], logging_output['ncorrect_total'], logging_output['mem_dep_pred'],
        #       logging_output['mem_dep'], preds, targets)
        # logging_output['preds'] = preds.detach().cpu().numpy().tolist()
        # logging_output['targets'] = targets.detach().cpu().numpy().tolist()

        preds = logits.argmax(dim=1)

        if random.random() < 0.01:
            print('groundtruth:', targets)
            print('prediction:', preds)

        logging_output['ncorrect_total'] = (preds == targets).sum().item()
        logging_output['ncorrect'] = ((preds == targets) * (targets == 1)).sum().item()

        logging_output['mem_dep'] = (targets == 1).sum().item()
        logging_output['mem_dep_pred'] = (preds == 1).sum().item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        # if sample_size != ntokens:
        #     metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect_total' in logging_outputs[0]:
            ncorrect_total = sum(log.get('ncorrect_total', 0) for log in logging_outputs)
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            mem_dep = sum(log.get('mem_dep', 0) for log in logging_outputs)
            mem_dep_pred = sum(log.get('mem_dep_pred', 0) for log in logging_outputs)

            precision = ncorrect / (mem_dep_pred + 1e-5)
            recall = ncorrect / (mem_dep + 1e-5)
            F1 = 2 * (precision * recall) / (precision + recall + 1e-5)
            metrics.log_scalar('precision', 100.0 * precision, sample_size, round=1)
            metrics.log_scalar('recall', 100.0 * recall, sample_size, round=1)
            metrics.log_scalar('F1', 100.0 * F1, sample_size, round=1)
            metrics.log_scalar('accuracy', 100.0 * ncorrect_total / sample_size, sample_size, round=1)

            # preds = list(itertools.chain.from_iterable([log.get('preds', 0) for log in logging_outputs]))
            # targets = list(itertools.chain.from_iterable([log.get('targets', 0) for log in logging_outputs]))
            # if len(set(targets)) != 2:
            #     auc = 1.
            # else:
            #     auc = roc_auc_score(targets, preds)
            # metrics.log_scalar('AUC', auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
