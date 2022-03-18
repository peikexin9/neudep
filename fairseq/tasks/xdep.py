# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING, II

from command import configs
from fairseq import utils
from fairseq.data import (
    BytevalueDataset,
    Dictionary,
    IdDataset,
    MaskCodeDataset,
    MaskValueDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from .language_modeling import SAMPLE_BREAK_MODE_CHOICES, SHORTEN_METHOD_CHOICES

logger = logging.getLogger(__name__)


@dataclass
class XdepConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
                    'tokens. If set to "complete", splits samples only at the end '
                    "of sentence, but may include multiple sentences per sample. "
                    '"complete_doc" is similar but respects doc boundaries. '
                    'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
                    'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")

    # xdep specific

    # training loss balance between code and value prediction
    code_value_loss_alpha: float = field(
        default=10,
        metadata={'help': "hyperparameter to balance the pretraining loss for code and value prediction"}
    )

    # no curriculum learning
    no_curriculum: bool = field(
        default=False,
        metadata={
            'help': "do not train the model with curriculum learning, e.g., random shuffle samples, fix learning rate"}
    )


@register_task("xdep", dataclass=XdepConfig)
class XdepTask(FairseqTask):
    cfg: XdepConfig

    """Adapted from Task for training masked language models (e.g., BERT, RoBERTa)."""

    def __init__(self, cfg: XdepConfig, dictionary_dict):
        super().__init__(cfg)
        self.dictionary_dict = dictionary_dict
        self.seed = cfg.seed

        # add mask token
        self.mask_idx_dict = {}
        for field in configs.maskable_fields:
            self.mask_idx_dict[field] = dictionary_dict[field].add_symbol("<mask>")

    @classmethod
    def setup_task(cls, cfg: XdepConfig, **kwargs):
        # paths = utils.split_paths(cfg.data)

        paths = os.listdir(cfg.data)
        assert len(paths) > 0

        if len(paths) != len(configs.fields):
            print('ERROR: invalid paths:', paths)
            raise ValueError()

        dictionary_dict = {}
        for field in configs.fields:
            dictionary_dict[field] = Dictionary.load(os.path.join(cfg.data, field, "dict.txt"))
            logger.info(f"{field} dictionary: {len(dictionary_dict[field])} types")
        return cls(cfg, dictionary_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0

        # curriculum learning by gradually increasing masking rate
        # up to masking 80% - assuming training at most 20 epochs
        if self.cfg.no_curriculum:
            mask_prob = self.cfg.mask_prob
        else:
            mask_prob = self.cfg.mask_prob + (.8 - self.cfg.mask_prob) * (epoch - 1) / 20

        src_tokens = {}
        tgt_tokens = {}
        for field in configs.fields:
            split_path = os.path.join(self.cfg.data, field, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.cfg.shorten_data_split_list,
                self.cfg.shorten_method,
                self.cfg.tokens_per_sample,
                self.cfg.seed,
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.cfg.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary[field].pad(),
                eos=self.source_dictionary[field].eos(),
                break_mode=self.cfg.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary[field].bos())

            if field == configs.static_field:
                src_dataset_code, tgt_dataset_code = MaskCodeDataset.apply_mask(
                    dataset,
                    self.source_dictionary[field],
                    pad_idx=self.source_dictionary[field].pad(),
                    mask_idx=self.mask_idx_dict[field],
                    seed=self.cfg.seed,
                    mask_prob=mask_prob,
                    leave_unmasked_prob=self.cfg.leave_unmasked_prob,
                    random_token_prob=self.cfg.random_token_prob,
                    freq_weighted_replacement=self.cfg.freq_weighted_replacement,
                )
                src_tokens[field] = RightPadDataset(
                    src_dataset_code,
                    pad_idx=self.source_dictionary[field].pad()
                )
                tgt_tokens[field] = RightPadDataset(
                    tgt_dataset_code,
                    pad_idx=self.source_dictionary[field].pad()
                )
            elif field in configs.byte_fields:
                src_dataset_value, tgt_dataset_value = MaskValueDataset.apply_mask(
                    dataset,
                    self.source_dictionary[field],
                    pad_idx=self.source_dictionary[field].pad(),
                    mask_idx=self.mask_idx_dict[field],
                    seed=self.cfg.seed,
                    mask_prob=mask_prob,
                    leave_unmasked_prob=self.cfg.leave_unmasked_prob,
                    random_token_prob=self.cfg.random_token_prob,
                    freq_weighted_replacement=self.cfg.freq_weighted_replacement,
                )

                # implemented auto-padding and normalizing bytes
                # dummy tokens are treated as 1, padded tokens are treated as 1 too
                src_tokens[field] = BytevalueDataset(src_dataset_value, self.source_dictionary[field])
                tgt_tokens[field] = BytevalueDataset(tgt_dataset_value, self.source_dictionary[field])
            elif field in configs.mem_fields:
                src_tokens[field] = BytevalueDataset(dataset, self.source_dictionary[field])
            else:
                src_tokens[field] = RightPadDataset(
                    dataset,
                    pad_idx=self.source_dictionary[field].pad()
                )

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset_code))

        if self.cfg.no_curriculum:
            sort_order = shuffle
        else:
            sort_order = [
                shuffle,
                src_dataset_code.sizes,
            ]

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": NumelDataset(src_dataset_code, reduce=False),
                    },
                    "target": tgt_tokens,
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset_code, reduce=True),
                },
                sizes=[src_dataset_code.sizes],
            ),
            sort_order=sort_order,
        )

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict
