import random
import torch
from torch import nn
from torch.nn import functional as F

def softmax(x, temperature=10): # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional):
        super().__init__()

        self.input_dim = input_dim          # source vocab size
        self.emb_dim = emb_dim              # vector length for each token
        self.hid_dim = hid_dim              # hidden dim of the RNN
        self.n_layers = n_layers            # num LSTM layers
        self.dropout = dropout              # prob of zeroing out units
        self.bidirectional = bidirectional  # bool, if the RNN will be biderectional

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        #src = [src sent len, batch size]

        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))

        #embedded = [src sent len, batch size, emb dim]

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        outputs, (hidden, cell) = self.rnn(embedded)
        #print("outputs.shape\n\t", outputs.shape)

        #outputs = [src sent len, batch size, hid dim * n directions]

        #hidden = [n layers * n directions, batch size, hid dim]  | for n directions = 1
        #       = [n layers, batch size, hid dim]                 | for n directions = 1
        #       = [n layers, batch size, hid dim * n directions]
        #cell = [n layers * n directions, batch size, hid dim]    | for n directions = 1
        #     = [n layers, batch size, hid dim]                   | for n directions = 1
        #     = [n layers, batch size, hid dim * n directions]


        #outputs are always from the top hidden layer
        if self.bidirectional:
            #print("hidden.shape before adjustment\n\t", hidden.shape)
            hidden = hidden.reshape(self.n_layers, 2, -1, self.hid_dim)
            hidden = hidden.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)
            #print("hidden.shape after adjustment\n\t", hidden.shape)

            #print("cell.shape before adjustment\n\t", cell.shape)
            cell = cell.reshape(self.n_layers, 2, -1, self.hid_dim)
            cell = cell.transpose(1, 2).reshape(self.n_layers, -1, 2 * self.hid_dim)
            #print("cell.shape after adjustment\n\t", cell.shape)


        # in both cases biderectional=True/False we get the followong shapes
        #   hidden = [n layers, batch size, hid dim * n directions]
        #   cell = [n layers, batch size, hid dim * n directions]
        return outputs, hidden, cell


# use your temperature
def softmax(x, temperature):
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)


# you can paste code of attention from modules.py
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, softmax_temp=1):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)

        # defaults to 1 (the usual softmax w/o temperature)
        self.softmax_temp = softmax_temp

    def forward(self, hidden, encoder_outputs):

        # encoder_outputs = [src sent len, batch size, enc_hid_dim]

        # only take the last layer of the decoder's hidden units
        # hidden = [n_layers, batch size, dec_hid_dim]
        last_hidden = hidden[-1, :, :].unsqueeze(0)
        # last_hidden = [1, batch size, dec_hid_dim]

        # repeat hidden and concatenate it with encoder_outputs
        hiddens = last_hidden.repeat(encoder_outputs.shape[0], 1, 1)
        # hiddens = [src sent len, batch size, dec_hid_dim]

        #print("encoder_outputs.shape:", encoder_outputs.shape)
        #print("hiddens.shape:", hiddens.shape)

        concat_h_s = torch.cat([hiddens, encoder_outputs], dim=2)
        #print("concat_h_s.shape.shape:", concat_h_s.shape)
        #print("(self.enc_hid_dim + self.dec_hid_dim).shape:", self.enc_hid_dim + self.dec_hid_dim)
        # concat_h_s = [src sent len, batch size, enc_hid_dim + dec_hid_dim]

        # calculate energy: E
        E = torch.tanh(self.attn(concat_h_s))
        # E = [src sent len, batch size, enc_hid_dim] (see self.attn)

        # get attention (not normalized to probabilities yet)
        a = self.v(E)
        # a = [src sent len, batch size, 1]

        # use softmax function which is defined, can change temperature
        return softmax(a, temperature=self.softmax_temp)


# you can paste code of decoder from modules.py
class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim,
                 n_layers, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim        # vocab size in the target language (hence output)
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)


        # use GRU: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.rnn = nn.GRU(input_size=emb_dim + enc_hid_dim, # see blue box on the image
                          hidden_size=dec_hid_dim,
                          num_layers=n_layers, dropout=dropout)

        # linear layer to get next word: f(y_t, w_t, s_t)
        # see purple box on the image
        self.out = nn.Linear(in_features=emb_dim + enc_hid_dim + dec_hid_dim,
                             out_features=output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, dec_hid_dim]
        # print(encoder_outputs.shape)

        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, dec_hid_dim]

        # input: [batch size] -> [1, batch size]
        input = input.unsqueeze(0) # because only one word, seq_len=1

        # Compute an embedding from the input data and apply dropout to it
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]

        # get weighted sum of (encoder_outputs = [src sent len, batch size, enc_hid_dim])
        a = self.attention(hidden, encoder_outputs)     # a = [src sent len, batch size, 1]
        weighted = torch.bmm(a.permute(1, 2, 0),        # [batch size, 1, src sent len]
                      encoder_outputs.transpose(0, 1)   # [batch size, src sent len, enc_hid_dim]
                      ).transpose(0, 1) # w = [batch size, 1, enc_hid_dim] -> [1, batch size, enc_hid_dim]

        # print("embedded.shape:", embedded.shape)
        # print("w.shape:", w.shape)

        # concatenate weighted sum and embedded, break through the GRU
        # embedded  = [1, batch size, emb dim]
        # weighted  = [1, batch size, enc_hid_dim]
        output, hidden = self.rnn(torch.cat([embedded, weighted], dim=2), hidden)
        # output    = [1, batch size, dec_hid_dim]
        # hidden    = [n layers, batch size, dec_hid_dim]
        #   so if we want to use hidden in prediction, we need to get the last layer:
        #   hidden[-1,:,:] = [batch size, dec_hid_dim]


        # need the dimentions to agree for torch.cat() to work
        #
        # torch.cat([embedded, weighted, output], dim=2).squeeze(0)
        #   = [batch size, emb_dim + enc_hid_dim + dec_hid_dim] =
        # torch.cat([embedded, weighted, hidden[-1,:,:].unsqueeze(0)], dim=2).squeeze(0)
        #   =
        # torch.cat([embedded.squeeze(0), weighted.squeeze(0), hidden[-1,:,:]], dim=1)

        # get predictions
        #
        # my initial version:
        # prediction = self.out(torch.cat([embedded, weighted, output], dim=2).squeeze(0))
        #
        # original paper version:
        prediction = self.out(torch.cat([embedded, # [1, batch size, emb dim]
                                         weighted, # [1, batch size, enc_hid_dim]
                                         # hidden[-1,:,:].unsqueeze(0) = [1, batch size, dec_hid_dim]
                                         hidden[-1,:,:].unsqueeze(0)], dim=2).squeeze(0))
        #prediction = [batch size, output dim]

        # will be used as arguments again:
        #   (part of input (top1 or teacher forced correct), hidden)
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim * (1 + encoder.bidirectional) == decoder.dec_hid_dim, \
                "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):

        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimention instead of zero
        trg_len = trg.shape[0]      # trg = [trg sent len, batch size]
        batch_size = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden, cell = self.encoder(src)
        #print('Shapes for encoder_outputs, hidden, cell')
        #print(encoder_outputs.shape, hidden.shape, cell.shape, '\n')

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden, and encoder_outputs
            # receive output tensor (predictions) and new hidden
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(-1)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
