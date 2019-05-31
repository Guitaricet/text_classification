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
        assert cnn_kernel_size % 2  # for 'same' padding

        super().__init__()
        self.embedding_dim = embedding_dim or len(cfg.alphabet)
        self.alphabet_len = alphabet_len or len(cfg.alphabet)
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
        self.words_rnn = nn.GRU(n_filters, hidden_dim_out, batch_first=True)
        self.projector = nn.Linear(hidden_dim_out, num_classes)

        # Initializations
        torch.nn.init.kaiming_normal_(self.char_cnn[0].weight)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):

        batch_size = x.size(1)
        # TODO: vectorize this and get rid of words_tensor
        words_tensor = torch.zeros(self.max_text_len, batch_size, self.conv_dim)
        if cfg.cuda:
            words_tensor = words_tensor.cuda()

        for i in range(self.max_text_len):
            word = x[i * self.max_word_len: (i + 1) * self.max_word_len, :]
            word = self.embedding(word)
            word = word.permute(1, 2, 0)
            word = self.char_cnn(word)
            word = self.highway(word)
            word = word.view(word.size(0), -1)
            words_tensor[i, :] = word

        # words_tensor = torch.nn.utils.rnn.pack_padded_sequence(...)
        x, _ = self.words_rnn(words_tensor)
        x = self.dropout(x)
        x = self.projector(x[-1])
        return x


class RNNClassifier(nn.Module):
    name = 'RNNClassifier'

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5, cell_type='GRU', num_classes=2, elmo=None):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        if elmo is not None:
            self.elmo = elmo

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
        if self.elmo is not None:
            raise RuntimeError('ELMo is broken by now')
            x = self.elmo(x)['elmo_representations'][0]

        x, _ = self.rnn(x)
        _, x = torch.max(x, 1)
        x = self.dropout(x)
        x = self.projector(x)
        return x
