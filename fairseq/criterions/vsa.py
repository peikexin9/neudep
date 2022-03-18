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
class VSAConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="vsa",
        metadata={"help": "name of the classification head to use"},
    )


@register_criterion('vsa', dataclass=VSAConfig)
class VSACriterion(FairseqCriterion):

    def __init__(self, cfg: VSAConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.fields = configs.fields
        self.cfg = cfg

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
        ), 'model must provide classification head for --criterion=vsa'

        logits = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )[0]
        targets = model.get_targets(sample, [logits])
        sample_size = targets.size(0)

        real_idx = sample['target'] >= 0

        logits = logits[real_idx]
        targets = targets[real_idx]

        lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        loss = F.nll_loss(lprobs, targets, reduction='sum')

        # atcoder/avatar
        logging_output = {
            'loss': loss.data,
            'ntokens': targets.size(0),
            'nsentences': sample_size,
            'sample_size': sample_size
        }

        preds = logits.argmax(dim=1)

        if random.random() < 0.0001:
            print(preds[:10], targets[:10])

        logging_output['ncorrect_total'] = (preds == targets).sum()
        logging_output[f'ncorrect'] = ((preds == targets) * (targets == 1)).sum()
        logging_output[f'vsa'] = (targets == 1).sum().item()
        logging_output[f'vsa_pred'] = (preds == 1).sum().item()

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect_total' in logging_outputs[0]:
            ncorrect_total = sum(log.get('ncorrect_total', 0) for log in logging_outputs)

            ncorrect = sum(log.get(f'ncorrect', 0) for log in logging_outputs)
            vsa_pred = sum(log.get(f'vsa_pred', 0) for log in logging_outputs)
            vsa = sum(log.get(f'vsa', 0) for log in logging_outputs)

            precision = ncorrect / (vsa_pred + 1e-5)
            recall = ncorrect / (vsa + 1e-5)
            F1 = 2 * (precision * recall) / (precision + recall + 1e-5)

            metrics.log_scalar(f'precision', 100.0 * precision, ntokens, round=1)
            metrics.log_scalar(f'recall', 100.0 * recall, ntokens, round=1)
            metrics.log_scalar(f'F1', 100.0 * F1, ntokens, round=1)
            metrics.log_scalar('accuracy', 100.0 * ncorrect_total / ntokens, ntokens, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
