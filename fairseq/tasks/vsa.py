# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING, open_dict, II, OmegaConf

from command import configs
from fairseq import utils
from fairseq.data import (
    BytevalueDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    SortDataset,
    TruncateDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class VSAConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    label: str = field(
        default='other',
        metadata={"help": "label of memory region"}
    )
    num_classes: int = field(
        default=4,
        metadata={"help": "number of classes or regression targets"},
    )
    no_shuffle: bool = field(
        default=False,
    )
    max_positions: int = field(
        default=512,
        metadata={"help": "max tokens per example"},
    )
    classification_head_name: str = II("criterion.classification_head_name")
    seed: int = II("common.seed")


@register_task('vsa', dataclass=VSAConfig)
class VSATask(FairseqTask):

    def __init__(self, cfg, data_dictionary_dict, label_dictionary):
        super().__init__(cfg)
        self.dictionary_dict = data_dictionary_dict
        self._label_dictionary = label_dictionary

        # All field of each token
        self.fields = configs.fields

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, 'Must set --num-classes'

        data_dictionary_dict = {}
        for i, field in enumerate(configs.fields):
            data_dictionary_dict[field] = Dictionary.load(os.path.join(cfg.data, field, 'dict.txt'))
            if field in configs.maskable_fields:
                data_dictionary_dict[field].add_symbol('<mask>')  # to align with the dictionary used in pretraining

            logger.info(f'| [input] {field} dictionary: {len(data_dictionary_dict[field])} types')

        # load label dictionary
        label_dict = Dictionary.load(os.path.join(cfg.data, f'label_{cfg.label}', 'dict.txt'))
        print('| [label] dictionary: {} types'.format(len(label_dict)))

        return cls(cfg, data_dictionary_dict, label_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0

        src_tokens = {}
        target = {}
        for field in self.fields:
            split_path = os.path.join(self.cfg.data, field, split)

            src_dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                combine=combine,
            )
            if src_dataset is None:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

            # src_dataset = TruncateDataset(src_dataset, self.cfg.max_positions)

            if field in configs.byte_fields + configs.mem_fields:
                src_tokens[field] = BytevalueDataset(src_dataset, self.source_dictionary[field])
            else:
                src_tokens[field] = RightPadDataset(src_dataset, pad_idx=self.source_dictionary[field].pad())

        net_input = dict()
        net_input['src_tokens'] = src_tokens
        net_input['src_lengths'] = NumelDataset(src_dataset, reduce=False)

        with data_utils.numpy_seed(self.cfg.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        # Net input has multiple fields
        dataset = {
            'id': IdDataset(),
            'net_input': net_input,
            'target': target,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_dataset, reduce=True),
        }

        label_path = os.path.join(self.cfg.data, f'label_{self.cfg.label}', split)
        label_dataset = data_utils.load_indexed_dataset(
            label_path,
            self.label_dictionary,
            combine=combine,
        )

        if label_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, label_path))

        dataset.update(
            target=RightPadDataset(
                OffsetTokensDataset(
                    label_dataset,
                    offset=-self.label_dictionary.nspecial,
                ),
                pad_idx=self.label_dictionary.pad() - self.label_dictionary.nspecial
            )
        )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_dataset.sizes],
        )

        if self.cfg.no_shuffle:
            self.datasets[split] = nested_dataset
        else:
            self.datasets[split] = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(self.datasets[split])))
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_positions = self.cfg.max_positions

        model = models.build_model(cfg, self)

        model.register_classification_list_head(
            self.cfg.classification_head_name,
            num_classes=self.cfg.num_classes
        )

        return model

    def max_positions(self):
        return self.cfg.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict

    @property
    def label_dictionary(self):
        return self._label_dictionary
