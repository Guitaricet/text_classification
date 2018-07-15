import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

import cfg


# https://github.com/akurniawan/pytorch-transformer
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = Variable(torch.FloatTensor([key_dim]))
        if cfg.cuda:
            self._key_dim = self._key_dim.cuda()
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=0)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=0)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=0)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim)
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            # we need to enforce converting mask to Variable, since
            # in pytorch we can't do operation between Tensor and
            # Variable
            mask = Variable(torch.ones(diag_mat.size()) * (-2**32 + 1), requires_grad=False)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat-1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(
            attention.split(split_size=restore_chunk_size, dim=0), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
#         attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention


class AttentionedYoonKimModel(nn.Module):
    name = 'AttentionedYoonKimModel'

    def __init__(self,
                 n_filters,
                 cnn_kernel_size,
                 hidden_dim_out,
                 heads=1,
                 dropout=0.5,
                 embedding_dim=len(cfg.alphabet),
                 pool_kernel_size=cfg.max_word_len,
                 alphabet_len=None):
        """
        CharCNN-WordRNN model with multi-head attention

        Default pooling is MaxOverTime pooling
        """
        assert cnn_kernel_size % 2  # for 'same' padding

        super(AttentionedYoonKimModel, self).__init__()
        self.alphabet_len = alphabet_len or len(cfg.alphabet)
        self.dropout_prob = dropout
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_dim_out = hidden_dim_out
        self.heads = heads

        self.embedding = nn.Linear(self.alphabet_len, embedding_dim)
        self.chars_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, n_filters, kernel_size=cnn_kernel_size, stride=1, padding=int(cnn_kernel_size - 1) // 2),  # 'same' padding
            nn.ReLU(),
            # nn.BatchNorm1d(n_filters),
            nn.MaxPool1d(kernel_size=pool_kernel_size)
        )

        _conv_stride = 1  # by default
        _pool_stride = pool_kernel_size  # by default
        # I am not sure this formula is always correct:
        self.conv_dim = n_filters * max(1, int(((cfg.max_word_len - cnn_kernel_size) / _conv_stride - pool_kernel_size) / _pool_stride + 1))

        self.words_rnn = nn.GRU(self.conv_dim, hidden_dim_out, dropout=self.dropout_prob)
        self.attention = MultiHeadAttention(hidden_dim_out, hidden_dim_out, hidden_dim_out, dropout_p=self.dropout_prob, h=self.heads)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.projector = nn.Linear(hidden_dim_out, 2)

        # Initializers
        torch.nn.init.kaiming_normal_(self.chars_cnn[0].weight)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):
        batch_size = x.size(1)
        # TODO: hadrcode! (for CUDA)
        words_tensor = torch.zeros(cfg.max_text_len, batch_size, self.conv_dim).cuda()

        for i in range(cfg.max_text_len):
            word = x[i * cfg.max_word_len : (i + 1) * cfg.max_word_len, :]
            word = self.embedding(word)
            word = word.permute(1, 2, 0)
            word = self.chars_cnn(word)
            word = word.view(word.size(0), -1)
            words_tensor[i, :] = word

        x, _ = self.words_rnn(words_tensor)
        x = self.attention(x, x)
        x = self.dropout(x)
        x = self.projector(x[-1])
        return x


class YoonKimModel(nn.Module):
    name = 'YoonKimModel'

    def __init__(self, n_filters, cnn_kernel_size, hidden_dim_out,
                 dropout=0.5, embedding_dim=None, pool_kernel_size=cfg.max_word_len, alphabet_len=None):
        """
        Paper: https://arxiv.org/abs/1508.06615

        Модель принципиально работает так же, но есть некоторые сильные упрощения:

        нету highway-слоя
        тут используется фильтры только одного размера (а не трёх, как в оригинальной статье)
        Default pooling is MaxOverTime pooling
        """
        assert cnn_kernel_size % 2  # for 'same' padding

        super(YoonKimModel, self).__init__()
        self.embedding_dim = embedding_dim or len(cfg.alphabet)
        self.alphabet_len = alphabet_len or len(cfg.alphabet)
        self.dropout_prob = dropout
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.hidden_dim_out = hidden_dim_out

        self.embedding = nn.Linear(self.alphabet_len, embedding_dim)
        self.chars_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, n_filters, kernel_size=cnn_kernel_size, stride=1, padding=int(cnn_kernel_size - 1) // 2),  # 'same' padding
            nn.ReLU(),
            # nn.BatchNorm1d(n_filters),
            nn.MaxPool1d(kernel_size=pool_kernel_size)
        )

        _conv_stride = 1  # by default
        _pool_stride = pool_kernel_size  # by default
        # I am not sure this formula is always correct:
        self.conv_dim = n_filters * max(1, int(((cfg.max_word_len - cnn_kernel_size) / _conv_stride - pool_kernel_size) / _pool_stride + 1))
        self.dropout = nn.Dropout(self.dropout_prob)
        self.words_rnn = nn.GRU(self.conv_dim, hidden_dim_out)
        self.projector = nn.Linear(hidden_dim_out, 2)

        # Initializers
        torch.nn.init.kaiming_normal_(self.chars_cnn[0].weight)
        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):
        batch_size = x.size(1)
        # TODO: hadrcode! (for CUDA)
        words_tensor = torch.zeros(cfg.max_text_len, batch_size, self.conv_dim).cuda()

        for i in range(cfg.max_text_len):
            word = x[i * cfg.max_word_len: (i + 1) * cfg.max_word_len, :]
            word = self.embedding(word)
            word = word.permute(1, 2, 0)
            word = self.chars_cnn(word)
            word = word.view(word.size(0), -1)
            words_tensor[i, :] = word

        x, _ = self.words_rnn(words_tensor)
        x = self.dropout(x)
        x = self.projector(x[-1])
        return x


class RNNBinaryClassifier(nn.Module):
    name = 'RNNBinaryClassifier'

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.5, type_='GRU'):
        super(RNNBinaryClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout

        if type_ == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers)
        elif type_ == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers)
        # elif type_ == 'QRNN':
        #     self.rnn = QRNN(embedding_dim, hidden_dim, num_layers=num_layers)
        # elif type_ == 'SRU':
        #     self.rnn = sru.SRU(embedding_dim, hidden_dim, num_layers=num_layers)
        else:
            raise ValueError('Wrong type_', type_)
        # self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.projector = nn.Linear(hidden_dim, 2)

        # Initializers
        torch.nn.init.xavier_normal_(self.projector.weight)

    def forward(self, x):
        x, _ = self.rnn(x)
        # x = self.layernorm(x)
        x = self.dropout(x[-1])
        x = self.projector(x)
        return x


class CharCNN(nn.Module):
    name = 'CharCNN'

    # Note: use max-over-time pooling via torch.max?
    def __init__(self, n_filters, cnn_kernel_size, maxlen, alphabet_len, dropout=0.5):
        """
        :param dropout: dropout probability (1 - keep prob)
        """
        super(CharCNN, self).__init__()
        self.dropout_prob = dropout
        self.n_filters = n_filters
        self.cnn_kernel_size = cnn_kernel_size  # 15
        self.cnn_stride = 2
        self.pool_kernel_size = 64  # MAXLEN  # 64
        self.pool_stride = 32  # self.pool_kernel_size  # 32

        self.embedding = nn.Linear(alphabet_len, alphabet_len)
        self.conv = nn.Sequential(
            nn.Conv1d(alphabet_len, self.n_filters, kernel_size=self.cnn_kernel_size, stride=self.cnn_stride),
            nn.ReLU(),
            # nn.BatchNorm1d(self.n_filters),
            nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        )
        torch.nn.init.xavier_normal_(self.conv[0].weight)

        conv_dim = self.n_filters * (int(
            ((maxlen - self.cnn_kernel_size) / self.cnn_stride - self.pool_kernel_size) / self.pool_stride) + 1)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.projector = nn.Linear(conv_dim, 2)

        torch.nn.init.xavier_normal_(self.embedding.weight)
        torch.nn.init.xavier_normal_(self.projector.weight)
    def forward(self, x):
        """
        :param x: Tensor of shape (seq_len, batch_size, signal_dim)
        """
        x = self.embedding(x)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.projector(x)
        return x


str2model = {
    AttentionedYoonKimModel.name.lower(): AttentionedYoonKimModel,
    YoonKimModel.name.lower(): YoonKimModel,
    RNNBinaryClassifier.name.lower(): RNNBinaryClassifier,
    CharCNN.name.lower(): CharCNN
}
