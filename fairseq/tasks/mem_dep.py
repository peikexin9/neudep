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
    PrependTokenDataset,
    SortDataset,
    TruncateDataset,
    RawLabelDataset
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task

logger = logging.getLogger(__name__)


@dataclass
class MemDepConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    num_classes: int = field(
        default=2,
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


@register_task('mem_dep', dataclass=MemDepConfig)
class MemDepTask(FairseqTask):

    def __init__(self, cfg, data_dictionary_dict, inst_pair_dict):
        super().__init__(cfg)
        self.dictionary_dict = data_dictionary_dict
        self._inst_pair_dictionary = inst_pair_dict

        # All field of each token
        self.fields = configs.fields

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        # paths = utils.split_paths(args.data)
        # paths = os.listdir(args.data)
        # assert len(paths) > 0
        # assert len(paths) == len(params.fields)
        assert cfg.num_classes > 0, 'Must set --num-classes'

        data_dictionary_dict = {}
        for i, field in enumerate(configs.fields):
            data_dictionary_dict[field] = Dictionary.load(os.path.join(cfg.data, field, 'dict.txt'))
            if field in configs.maskable_fields:
                data_dictionary_dict[field].add_symbol('<mask>')  # to align with the dictionary used in pretraining

            logger.info(f'| [input] {field} dictionary: {len(data_dictionary_dict[field])} types')

        # load inst pair dictionary -- instruction pair annotation
        inst_pair_dict = Dictionary.load(os.path.join(cfg.data, 'dep_cmp_emb', 'dict.txt'))
        print('| [inst_pair] dictionary: {} types'.format(len(inst_pair_dict)))

        return cls(cfg, data_dictionary_dict, inst_pair_dict)

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

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            src_dataset = PrependTokenDataset(src_dataset, self.source_dictionary[field].bos())

            if field in configs.byte_fields + configs.mem_fields:
                src_tokens[field] = BytevalueDataset(src_dataset, self.source_dictionary[field])
            else:
                src_tokens[field] = RightPadDataset(
                    TruncateDataset(src_dataset, self.cfg.max_positions),
                    pad_idx=self.source_dictionary[field].pad()
                )

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

        inst_pair_path = os.path.join(self.cfg.data, 'dep_cmp_emb', split)
        inst_pair_dataset = data_utils.load_indexed_dataset(
            inst_pair_path,
            self.inst_pair_dictionary,
            combine=combine,
        )
        if inst_pair_dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, inst_pair_path))
        inst_pair_dataset = PrependTokenDataset(inst_pair_dataset, self.inst_pair_dictionary.bos())
        inst_pair_dataset = RightPadDataset(
            TruncateDataset(inst_pair_dataset, self.cfg.max_positions),
            pad_idx=self.inst_pair_dictionary.pad()
        )
        dataset.update(inst_pair=inst_pair_dataset)

        label_path = os.path.join(self.cfg.data, 'label', f'{split}.label')
        if not os.path.exists(label_path):
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, label_path))

        dataset.update(
            target=RawLabelDataset([
                int(x.strip()) for x in open(label_path).readlines()
            ])
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

        model.register_embedding_pair_local_head(
            self.cfg.classification_head_name,
            num_classes=self.cfg.num_classes,
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
    def inst_pair_dictionary(self):
        return self._inst_pair_dictionary
