import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import cfg


# from github.com/allenai/allennlp
class Highway(torch.nn.Module):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.
    This module will apply a fixed number of highway layers to its input, returning the final
    result.
    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size, ...,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``Callable[[torch.Tensor], torch.Tensor]``, optional (default=``torch.nn.functional.relu``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim,
                 num_layers=1,
                 activation=torch.nn.functional.relu):
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, x):
        for layer in self._layers:
            projected_input = layer(x)
            linear_part = x
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            x = gate * linear_part + (1 - gate) * nonlinear_part
        return x


class YoonKimModel(nn.Module):
    name = 'YoonKimModel'

    def __init__(self,
                 n_filters,
                 cnn_kernel_size,
                 hidden_dim_out,
                 dropout=0.5,
                 embedding_dim=None,
                 pool_kernel_size=cfg.max_word_len,
                 alphabet_len=None,
                 max_text_len=None,
                 max_word_len=None,
                 num_classes=2):
        """
        Paper: https://arxiv.org/abs/1508.06615

        The model in principal similar to one in the paper, but have differences

        No highway layer
        Only one kernel size (not three different kernels like in the paper)
        Default pooling is MaxOverTime pooling
        """
        super().__init__()
        self.alphabet_len = alphabet_len or len(cfg.alphabet)
        self.embedding_dim = embedding_dim or self.alphabet_len
        self.dropout_prob = dropout
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_dim_out = hidden_dim_out
        self.max_text_len = max_text_len or cfg.max_text_len
        self.max_word_len = max_word_len or cfg.max_word_len

        # TODO: hardcode
        self.embedding = nn.Embedding(self.alphabet_len, embedding_dim, padding_idx=0)
        self.char_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, n_filters, kernel_size=cnn_kernel_size, stride=1),
            nn.ReLU()
        )
        self.highway = Highway(n_filters)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.word_rnn = nn.GRU(n_filters, hidden_dim_out, batch_first=True)
        self.projector = nn.Linear(hidden_dim_out, num_classes)

        # Initializations
        torch.nn.init.kaiming_normal_(self.char_cnn[0].weight)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, word_len)
        """
        (batch_size, seq_len, word_len) = x.size()

        x = self.embedding(x)
        x = x.reshape([batch_size * seq_len, word_len, -1])
        x = x.permute([0, 2, 1])  # (batch_size * seq_len, hidden, word_len)

        x = self.char_cnn(x)
        x = torch.max(x, 2).values  # maxpool over word characters
        x = x.reshape([batch_size, seq_len, -1])  # (batch_size, seq_len, hidden)

        x = self.highway(x)
        x = self.dropout(x)
        x, _ = self.word_rnn(x)
        x = torch.max(x, 1).values  # maxpool over words

        x = self.dropout(x)
        x = self.projector(x)
        return x


class RNNClassifier(nn.Module):
    name = 'RNNClassifier'

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5, cell_type='GRU', num_classes=2, elmo=None):
        super().__init__()
        if elmo is not None:
            raise RuntimeError('ELMo is not supported')

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout

        if cell_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True,)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError('Wrong cell_type', cell_type)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.projector = nn.Linear(hidden_dim, num_classes)

        # Initializers
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):
        """
        :param x: Tensor, (batch_size, seq_len, features)
        """

        x, _ = self.rnn(x)
        x = torch.max(x, 1)  # namedtuple (values, indices)
        x = self.dropout(x.values)
        x = self.projector(x)
        return x


class ALaCarteClassifier(nn.Module):
    name = 'ALaCarteClassifier'

    def __init__(self, hidden_dim, emb_matrix,
                 num_layers=1, dropout=0.5, cell_type='GRU', num_classes=2,
                 induction_matrix=None, trainable_induction=False, window_half_size=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout

        self.embedding = None

        # make embedding matrix
        self.embedding_size = emb_matrix.shape[1]
        self.pad_idx = emb_matrix.shape[0]

        emb_matrix_pt = torch.FloatTensor(emb_matrix)
        emb_matrix_pt = torch.cat([emb_matrix_pt, torch.zeros([1, self.embedding_size])])

        self.unk_index = -1  # TODO: hardcode
        self.embedding = nn.Embedding.from_pretrained(emb_matrix_pt, padding_idx=self.pad_idx, freeze=True)
        self.unk_vec = torch.randn(self.embedding_size).cuda()

        self.induction_matrix = induction_matrix  # None case
        if isinstance(induction_matrix, str) and induction_matrix == 'identity':
            self.induction_matrix = torch.Tensor(np.identity(self.embedding.embedding_dim))
        elif isinstance(induction_matrix, np.ndarray):
            self.induction_matrix = torch.Tensor(induction_matrix)
        self.induction_matrix = torch.nn.Parameter(self.induction_matrix)

        if not trainable_induction:
            self.induction_matrix.requires_grad = False

        self.window_half_size = window_half_size

        if cell_type == 'GRU':
            self.rnn = nn.GRU(self.embedding_size, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True,)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(self.embedding_size, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError('Wrong cell_type', cell_type)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.projector = nn.Linear(hidden_dim, num_classes)

        # Initializers
        torch.nn.init.xavier_normal_(self.projector.weight)

    def to(self, device):
        self.unk_vec.to(device)
        super().to(device)

    def induce_embeddings(self, x, unk_indices):
        batch_size, seq_len, _ = x.shape
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                if unk_indices[batch_idx, seq_idx]:
                    left_word = max(0, seq_idx - self.window_half_size)
                    right_word = min(seq_len, seq_idx + self.window_half_size + 1)
                    context_indices = [v for v in list(range(left_word, right_word)) if not unk_indices[batch_idx, seq_idx]]  # noqa E501
                    context_vector = self.unk_vec
                    if len(context_indices):
                        context_vector = torch.mean(x[batch_idx, context_indices]).cuda()

                    x[batch_idx, seq_idx] = self.induction_matrix @ context_vector
        return x

    def forward(self, x):
        """
        :param x: Tensor, (batch_size, seq_len, features)
        """
        unk_indices = x == self.unk_index
        x.masked_fill_(unk_indices, self.pad_idx)
        x = self.embedding(x)

        if self.induction_matrix is not None:
            x = self.induce_embeddings(x, unk_indices)
        else:
            x[unk_indices, :] = self.unk_vec
        self._embedded_batch = x

        x, _ = self.rnn(x)
        x = torch.max(x, 1).values
        x = self.dropout(x)
        x = self.projector(x)
        return x
