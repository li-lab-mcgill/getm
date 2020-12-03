import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ETM(nn.Module):
    def __init__(self, num_topics, num_times, vocab_size1, vocab_size2, t_hidden_size, rho_size, emsize,
                 theta_act, delta, nlayer, embeddings1=None, embeddings2=None, train_embeddings1=True,
                 train_embeddings2=True, enc_drop=0.5, add_freq=False, base_freq1=None, base_freq2=None):
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
        # self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
        self.theta_act = self.get_activation(theta_act)
        self.num_times = num_times
        self.delta = delta
        self.nlayer = nlayer

        self.train_embeddings1 = train_embeddings1
        self.train_embeddings2 = train_embeddings2
        self.add_freq = add_freq

        self.base_freq1 = base_freq1
        self.base_freq2 = base_freq2

        ## define the word embedding matrix \rho
        if self.train_embeddings1:
            self.rho1 = nn.Linear(rho_size, vocab_size1, bias=True)
        else:
            self.rho1 = embeddings1.clone().float().to(device)
        if self.train_embeddings2:
            self.rho2 = nn.Linear(rho_size, vocab_size2, bias=True)  # L x V
        else:
            # num_embeddings, emsize = embeddings.size()
            # rho = nn.Embedding(num_embeddings, emsize)
            self.rho2 = embeddings2.clone().float().to(device)  # V x L
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size))

        # define variational distribution for \theta_{1:D} via amortizartion
        # self.q_theta = nn.Sequential(
        #     nn.Linear(vocab_size, t_hidden_size),
        #     self.theta_act,
        # )
        # self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        # self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.q_theta_map = nn.Linear(self.vocab_size1+vocab_size2, self.t_hidden_size)
        self.q_theta = nn.LSTM(self.t_hidden_size, self.t_hidden_size, self.nlayer, batch_first=True, dropout=enc_drop)
        self.mu_q_theta = nn.Linear(self.t_hidden_size+self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(self.t_hidden_size+self.num_topics, self.num_topics, bias=True)

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

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1).mean()
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1).mean()
        return kl

    def get_alpha(self):  ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])

            p_mu_t = alphas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha

    def get_theta(self, normalized_bows): ## structured amortized inference
        batch_size = normalized_bows.size()[0]
        inp = self.q_theta_map(normalized_bows)
        hidden = self.init_hidden(batch_size)
        output, _ = self.q_theta(inp, hidden)


        zs = torch.zeros(self.num_times, batch_size, self.num_topics).to(device)
        kl_theta = []

        inp_0 = torch.cat([output[:, 0, :], torch.zeros(batch_size, self.num_topics).to(device)], dim=1)
        mu_0 = self.mu_q_theta(inp_0)
        logsigma_0 = self.logsigma_q_theta(inp_0)
        zs[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(batch_size, self.num_topics).to(device)
        logsigma_p_0 = torch.zeros(batch_size, self.num_topics).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_theta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[:, t, :], zs[t - 1]], dim=1)
            mu_t = self.mu_q_theta(inp_t)
            logsigma_t = self.logsigma_q_theta(inp_t)
            zs[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = zs[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(batch_size, self.num_topics).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)

            kl_theta.append(kl_t)
        kl_theta = torch.stack(kl_theta).sum()
        thetas = F.softmax(zs, dim=-1)
        return thetas, kl_theta

    def init_hidden(self, bsize):
        """Initializes the first hidden state of the RNN used as inference network for theta.
        """
        weight = next(self.parameters())
        nlayers = self.nlayer
        nhid = self.t_hidden_size

        return (weight.new_zeros(nlayers, bsize, nhid), weight.new_zeros(nlayers, bsize, nhid))

    def calc_beta(self, alpha, rho, train_embeddings):
        """Returns the topic matrix \beta of shape  T x K x V
        """
        if train_embeddings:
            logit = rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta

    def get_beta(self, alpha):
        beta1 = self.calc_beta(alpha, self.rho1, self.train_embeddings1)
        beta2 = self.calc_beta(alpha, self.rho2, self.train_embeddings2)
        return beta1, beta2

    def get_likelihood(self, theta, beta, bows, base_freq):
        kld_z = []
        log_likelihood = []
        freq = torch.stack([base_freq for _ in range(bows.size(0))])
        for t in range(self.num_times):

            pi_t = torch.bmm(beta[t].unsqueeze(2), theta[t].T.unsqueeze(1)).T  # D*V*K
            pi_t = F.softmax(pi_t, dim=-1)
            z_t = F.gumbel_softmax(pi_t, dim=-1, hard=True)
            kld_z_t = -torch.sum(torch.bmm(z_t, torch.log(theta[t]).unsqueeze(2)), dim=-1).mean() - \
                torch.sum(torch.sum(pi_t * z_t, dim=-1) * torch.sum(torch.log(pi_t) * z_t, dim=-1), dim=-1).mean()

            beta_z_t = beta[t]*(z_t.mean(0)).T
            theta_z_t = theta[t]*z_t.mean(1)
            if self.add_freq:
                res_t = torch.mm(theta_z_t, beta_z_t)-freq
            else:
                res_t = torch.mm(theta_z_t, beta_z_t)
            preds_t = torch.log(res_t)
            log_likelihood_t = (preds_t * bows[:, t, :]).sum(1).mean()
            kld_z.append(kld_z_t)
            log_likelihood.append(log_likelihood_t)
        kld_z = torch.stack(kld_z).sum()
        log_likelihood = torch.stack(log_likelihood).sum()
        return kld_z, log_likelihood

    def decode(self, theta, beta1, beta2, bows):
        bows1 = bows[:, :, :self.vocab_size1]
        bows2 = bows[:, :, -self.vocab_size2:]

        kld_z1, log_likelihood1 = self.get_likelihood(theta, beta1, bows1, self.base_freq1)
        kld_z2, log_likelihood2 = self.get_likelihood(theta, beta2, bows2, self.base_freq2)

        return kld_z1, log_likelihood1, kld_z2, log_likelihood2

    def forward(self, bows, normalized_bows, theta=None):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None
        alpha, kld_alpha = self.get_alpha()
        ## get \beta
        beta1, beta2 = self.get_beta(alpha)

        ## get prediction loss
        kld_z1, log_likelihood1, kld_z2, log_likelihood2 = self.decode(theta, beta1, beta2, bows)
        recon_loss = -log_likelihood1-log_likelihood2


        return recon_loss, kld_theta, kld_z1, kld_z2, kld_alpha
