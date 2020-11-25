from typing import Dict

import torch
from torch import nn

from configs import EncoderConfig
from utils.common import FROM_TOKEN, TO_TOKEN, PATH_TYPES
import numpy as np

class PathEncoder(nn.Module):
    def __init__(
        self,
        config: EncoderConfig,
        out_size: int,
        n_subtokens: int,
        subtoken_pad_id: int,
        n_types: int,
        type_pad_id: int,
    ):
        super().__init__()
        self.type_pad_id = type_pad_id
        self.num_directions = 2 if config.use_bi_rnn else 1
        self.n_subtokens = n_subtokens

        self.subtoken_embedding = nn.Linear(n_subtokens, config.embedding_size)
        self.subtoken_embedding1 = nn.Embedding(n_subtokens, config.embedding_size, padding_idx=subtoken_pad_id)
        self.type_embedding = nn.Embedding(n_types, config.embedding_size, padding_idx=type_pad_id)

        self.dropout_rnn = nn.Dropout(config.rnn_dropout)
        self.path_lstm = nn.LSTM(
            config.embedding_size,
            config.rnn_size,
            num_layers=config.rnn_num_layers,
            bidirectional=config.use_bi_rnn,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
        )

        concat_size = config.embedding_size * 2 + config.rnn_size * self.num_directions
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.linear = nn.Linear(concat_size, out_size, bias=False)
        self.norm = nn.LayerNorm(out_size)
        
    def forward(self, contexts: Dict[str, torch.Tensor], already_one_hot: bool=False) -> torch.Tensor:
        # [max name parts; total paths]
        from_token = contexts[FROM_TOKEN]
        to_token = contexts[TO_TOKEN]

        # [total paths; embedding size]
        use_embedding_layer = False
        if not use_embedding_layer:
            if not already_one_hot:
                from_token1 = from_token.permute(1, 0).half()
                print(from_token1.shape,'===============')
                from_token1 = torch.zeros(from_token1.size(0), from_token1.size(1), self.n_subtokens).scatter_(2, from_token1.unsqueeze(2), 1.).squeeze().half()
            else:
                from_token1 = from_token
            encoded_from_tokens = self.subtoken_embedding(from_token1).sum(1)
            
            if not already_one_hot:
                # N x 5
                to_token1 = to_token.permute(1, 0)
                # N x 5 x |V|
                to_token1 = torch.zeros(to_token1.size(0), to_token1.size(1), self.n_subtokens).scatter_(2, to_token1.unsqueeze(2), 1.).squeeze()
            else:
                to_token1 = to_token
            # N x 5 x 64 -> sum(1) -> # N x 64
            encoded_to_tokens = self.subtoken_embedding(to_token1).sum(1)
        else:
            encoded_from_tokens = self.subtoken_embedding1(from_token).sum(0)
            encoded_to_tokens = self.subtoken_embedding1(to_token).sum(0)
        
        print(encoded_from_tokens.shape)
        print(encoded_to_tokens.shape)
        qwe
        # [max path length; total paths]
        path_types = contexts[PATH_TYPES]
        # [max path length; total paths; embedding size]
        path_types_embed = self.type_embedding(path_types)

        # create packed sequence (don't forget to set enforce sorted True for ONNX support)
        with torch.no_grad():
            path_lengths = (path_types != self.type_pad_id).sum(0)
        packed_path_types = nn.utils.rnn.pack_padded_sequence(path_types_embed, path_lengths, enforce_sorted=False)

        # [num layers * num directions; total paths; rnn size]
        _, (h_t, _) = self.path_lstm(packed_path_types)
        # [total_paths; rnn size * num directions]
        encoded_paths = h_t[-self.num_directions :].transpose(0, 1).reshape(h_t.shape[1], -1)
        encoded_paths = self.dropout_rnn(encoded_paths)

        # [total_paths; 2 * embedding size + rnn size (*2)]
        concat = torch.cat([encoded_from_tokens, encoded_paths, encoded_to_tokens], dim=-1)

        # [total_paths; output size]
        concat = self.embedding_dropout(concat)
        output = self.linear(concat)
        output = self.norm(output)
        output = torch.tanh(output)

        return output

