import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class ETM(nn.Module):
    def __init__(self, num_topics, num_times, vocab_size1, vocab_size2, t_hidden_size, rho_size, emsize,
                 theta_act, gamma, embeddings1=None, embeddings2=None, train_embeddings1=True,
                 train_embeddings2=True, rho_fixed1=False, rho_fixed2=False, enc_drop=0.5, e2e=False,
                 lstm_hidden_dim=None, nlayers=None, num_classes=None, lstm_dropout=0.0, predcoef=1):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        self.num_times = num_times

        self.e2e = e2e
        self.gamma = gamma
        self.predcoef = predcoef

        self.theta_act = self.get_activation(theta_act)

        self.train_embeddings1 = train_embeddings1
        self.rho_fixed1 = rho_fixed1

        self.train_embeddings2 = train_embeddings2
        self.rho_fixed2 = rho_fixed2

        ## define the word embedding matrix \rho
        if self.train_embeddings1:
            self.rho1 = nn.Linear(rho_size, vocab_size1, bias=True)  # L x V
        else:
            if not rho_fixed1:
                num_embeddings, emsize = embeddings1.size()
                self.rho1 = nn.Embedding(num_embeddings, emsize)
                # embeddings1 is of shape (num_embeddings, embedding_dim)
                self.rho1.weight.data.copy_(embeddings1)
            else:
                self.rho1 = embeddings1

        if self.train_embeddings2:
            self.rho2 = nn.Linear(rho_size, vocab_size2, bias=True)  # L x V
        else:
            if not rho_fixed2:
                num_embeddings, emsize = embeddings2.size()
                self.rho2 = nn.Embedding(num_embeddings, emsize)
                # embeddings1 is of shape (num_embeddings, embedding_dim)
                self.rho2.weight.data.copy_(embeddings2)
            else:
                self.rho2 = embeddings2

        ## define the matrix containing the topic embeddings
        self.alphas1 = nn.Linear(rho_size, num_topics, bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))
        self.alphas2 = nn.Linear(rho_size, num_topics, bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size1+vocab_size2, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size),
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        if e2e:
            self.lstm_hidden_dim = lstm_hidden_dim
            self.num_classes = num_classes
            self.predcoef = predcoef
            self.pred_nlayer = nlayers

            self.lstm = nn.LSTM(num_topics, lstm_hidden_dim, nlayers, batch_first=True, dropout=lstm_dropout)
            self.linear = nn.Linear(lstm_hidden_dim, self.num_classes, bias=True)
            # self.output = nn.Sigmoid()
            self.output = nn.Softmax(dim=-1)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

        # theta ~ mu + std N(0,1)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def calc_beta(self, rho, alphas):
        try:
            logit = alphas(rho.weight)
        except:
            logit = alphas(rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  ## softmax over vocab dimension
        return beta

    def get_beta(self):
        beta1 = self.calc_beta(self.rho1, self.alphas1)
        beta2 = self.calc_beta(self.rho2, self.alphas2)
        return beta1, beta2

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta, mu_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def calc_pred_loss(self, y, output, mask, weights, gamma):
        class_weights = torch.FloatTensor(weights).to(device)
        FL = FocalLoss(weight=class_weights, gamma=gamma)
        y_true = y[mask]
        focal_loss = FL.forward(output[mask], torch.max(y_true, 1)[1])
        return focal_loss

    def forward(self, bows, normalized_bows, y_true=None, mask=None, weights=None, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta, x = self.get_theta(normalized_bows)
        else:
            kld_theta = None
        bows1 = bows[:, :self.vocab_size1]
        bows2 = bows[:, -self.vocab_size2:]
        ## get \beta
        beta1, beta2 = self.get_beta()

        ## get prediction loss
        preds1 = self.decode(theta, beta1)
        preds2 = self.decode(theta, beta2)
        recon_loss = -(preds1 * bows1).sum(1) - (preds2 * bows2).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()

        if self.e2e:
            bsize = int(theta.shape[0] / self.num_times)
            x = x.reshape(self.num_times, bsize, self.num_topics)
            hidden = self.pred_init_hidden(bsize)
            x, _ = self.lstm(torch.transpose(x, 0, 1), hidden)
            x = self.linear(x)
            y_pred = self.output(x)
            pred_loss = self.predcoef * self.calc_pred_loss(y_true, y_pred, mask, weights, self.gamma)
            return recon_loss * self.predcoef, kld_theta, pred_loss
        else:
            return recon_loss * self.predcoef, kld_theta

    def pred_init_hidden(self, bsize):
        """Initializes the first hidden state of the RNN used as inference network for y_pred.
        """
        weight = next(self.parameters())
        nlayers = self.pred_nlayer
        nhid = self.lstm_hidden_dim

        return (weight.new_zeros(nlayers, bsize, nhid), weight.new_zeros(nlayers, bsize, nhid))

