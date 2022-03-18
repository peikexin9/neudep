# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders
from command import configs
from itertools import product


class XdepHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/main/examples/roberta
    """

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

        hexval = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e', 'f']
        self.real_bytes = set(f'{i}{j}' for i, j in product(hexval, repeat=2))

    @property
    def device(self):
        return self._float_tensor.device

    def __process_token_dict(self, tokens: dict):
        token_fields = tokens.keys()
        assert len(token_fields) == len(configs.fields) + 1  # dep_cmp_emb

        for field in configs.fields + ['dep_cmp_emb']:
            if tokens[field].dim() == 1:
                tokens[field] = tokens[field].unsqueeze(0)
            if tokens[field].size(-1) > self.model.max_positions():
                raise ValueError(
                    "tokens exceeds maximum length: {} > {}".format(
                        tokens[field].size(-1), self.model.max_positions()
                    )
                )
            tokens[field] = tokens[field].to(device=self.device).long()

        return tokens

    def __process_byte_tokens(self, sentence: str):
        split = sentence.strip().split()
        parsed = []
        for token in split:
            if token in self.real_bytes:
                parsed.append(int(token, 16) / 256)
            else:
                parsed.append(float(1))
        return torch.Tensor(parsed)

    def encode(self, sentence: dict) -> torch.LongTensor:
        """
        encode a code piece to its embedding index.
        This is executed independently with extract_features or predict
        """
        sentence_fields = sentence.keys()
        assert len(sentence_fields) == len(configs.fields) + 1

        token_dict = dict()

        for field in configs.fields:
            if field in configs.byte_fields:
                token_dict[field] = self.__process_byte_tokens(sentence[field])
            else:
                token_dict[field] = self.task.source_dictionary[field].encode_line(
                    sentence[field], append_eos=False, add_if_not_exist=False
                )

        token_dict['dep_cmp_emb'] = self.task.inst_pair_dictionary.encode_line(
            sentence['dep_cmp_emb'], append_eos=False, add_if_not_exist=False
        )

        return token_dict

    def extract_features(self, tokens: dict) -> torch.Tensor:
        tokens = self.__process_token_dict(tokens)

        features = self.model(
            tokens,
            features_only=True,
        )[0]

        return features  # just the last layer's features

    def predict(self, head: str, tokens: dict):
        func_embedding = self.extract_features(tokens)

        inst_pair = tokens['dep_cmp_emb']
        inst1_idx = inst_pair.eq(self.task.inst_pair_dictionary.index('1'))
        inst2_idx = inst_pair.eq(self.task.inst_pair_dictionary.index('2'))

        # pool the embedding
        inst1_embedding_mean = torch.mean(func_embedding * inst1_idx.unsqueeze(-1), dim=1)
        inst2_embedding_mean = torch.mean(func_embedding * inst2_idx.unsqueeze(-1), dim=1)

        # concatenate u, v, |u-v|, u*v
        concat_in = torch.cat((torch.mean(func_embedding, dim=1),
                               inst1_embedding_mean,
                               inst2_embedding_mean,
                               torch.abs(inst1_embedding_mean - inst2_embedding_mean),
                               inst1_embedding_mean * inst2_embedding_mean),
                              dim=-1)
        logits = self.model.classification_heads[head](concat_in)
        return logits
