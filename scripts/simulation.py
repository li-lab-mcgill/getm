import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_bounds(mat, upper, lower):
    lower = lower
    upper = upper
    y_upper = (torch.ones(mat.size())*upper).float().to(device)
    y_lower = (torch.ones(mat.size())*lower).float().to(device)
    mat = torch.where(mat >= lower, mat, y_lower)
    mat = torch.where(mat <= upper, mat, y_upper)
    return mat

class ETM(nn.Module):
    def __init__(self, num_topics, num_times, vocab_size, eta_hidden_size, t_hidden_size, rho_size, emsize,
                 theta_act, delta, nlayer, set_alpha=False, alpha_embedding=None, set_eta=False,
                 train_rho=True, rho_embeddings=None, enc_drop=0.1, eta_drop=0.1, upper=100, lower=-100):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        self.t_hidden_size = t_hidden_size
        self.eta_hidden_size = eta_hidden_size
        self.rho_size = rho_size

        self.enc_drop = enc_drop
        self.eta_drop = eta_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)
        # self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
        self.theta_act = self.get_activation(theta_act)
        self.num_times = num_times


        self.delta = delta
        self.nlayer = nlayer


        self.set_alpha = set_alpha
        self.set_eta = set_eta
        self.train_rho = train_rho



        self.upper = upper
        self.lower = lower

        ## define the word embedding matrix \rho
        if not self.set_alpha:
            self.rho = nn.Linear(rho_size, vocab_size, bias=True)
        else:
            # num_embeddings, emsize = rho_embeddings.size()
            # self.rho = nn.Embedding(num_embeddings, emsize)
            # # embeddings1 is of shape (num_embeddings, embedding_dim)
            # self.rho.weight.data.copy_(rho_embeddings)
            self.rho = rho_embeddings.clone().float().to(device)


            # self.rho1 = embeddings1.clone().float().to(device)
        if self.set_alpha:
            self.alpha = alpha_embedding.clone().float().to(device)
        else:
            ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
            self.mu_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size))
            self.logsigma_q_alpha = nn.Parameter(torch.randn(num_topics, num_times, rho_size))



        self.q_theta = nn.Sequential(
                    nn.Linear(self.vocab_size+self.num_topics, self.t_hidden_size),
                    self.theta_act,
                    nn.Linear(self.t_hidden_size, self.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(self.vocab_size, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.nlayer, batch_first=True, dropout=self.eta_drop)
        self.mu_q_eta = nn.Linear(self.eta_hidden_size+self.num_topics, self.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size+self.num_topics, self.num_topics, bias=True)


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
        # if self.training:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)
        # else:
        #     return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            q_logsigma = set_bounds(q_logsigma, self.upper, self.lower)
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1).mean()
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1).mean()
        return kl

    def get_alpha(self):  ## mean field
        if not self.set_alpha:
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
            kl_alpha = torch.stack(kl_alpha).mean()
            return alphas, kl_alpha
        else:
            return self.alpha

    def get_eta(self, bow_t, num_times): ## structured amortized inference

        batch_size = bow_t.size()[0]
        inp = self.q_eta_map(bow_t)
        hidden = self.init_hidden(batch_size)
        output, _ = self.q_eta(inp, hidden)


        etas = torch.zeros(num_times, batch_size, self.num_topics).to(device)
        kl_eta = []

        inp_0 = torch.cat([output[:, 0, :], torch.zeros(batch_size, self.num_topics).to(device)], dim=1)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(batch_size, self.num_topics).to(device)
        logsigma_p_0 = torch.zeros(batch_size, self.num_topics).to(device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, num_times):
            inp_t = torch.cat([output[:, t, :], etas[t - 1]], dim=1)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(batch_size, self.num_topics).to(device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)

            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()

        return etas, kl_eta



    def get_theta(self, eta_age, bows): ## amortized inference
        """Returns the topic proportions.
        """
        bows = torch.stack([bows for _ in range(self.num_times)])
        inp = torch.cat([bows, eta_age], dim=-1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_age, torch.zeros(self.num_topics).to(device))
        return theta, kl_theta, z

    def init_hidden(self, bsize):
        """Initializes the first hidden state of the RNN used as inference network for theta.
        """
        weight = next(self.parameters())
        nlayers = self.nlayer
        nhid = self.eta_hidden_size

        return (weight.new_zeros(nlayers, bsize, nhid), weight.new_zeros(nlayers, bsize, nhid))

    def get_beta(self, alpha):
        """Returns the topic matrix \beta of shape  T x K x V
        """
        if not self.set_alpha:
            logit = self.rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            # logit = torch.mm(tmp, self.rho.weight.permute(1, 0))
            logit = torch.mm(tmp, self.rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta



    def calc_constraints(self, beta1, beta2):
        con_loss = []
        for t in range(self.num_times):
            c11 = torch.mm(torch.mm(beta1[t], self.adj_mat11), beta1[t].T)
            c12 = torch.mm(torch.mm(beta1[t], self.adj_mat12), beta2[t].T)
            c22 = torch.mm(torch.mm(beta2[t], self.adj_mat22), beta2[t].T)
            constraints = self.lambda11*torch.trace(c11) + self.lambda12*torch.trace(c12) + self.lambda22*torch.trace(c22)
            con_loss.append(-constraints)
        con_loss = torch.stack(con_loss).sum()

        return con_loss

    def get_likelihood(self, theta, beta, bow_t):
        nll = []
        for t in range(self.num_times):
            if len(theta[t]) > 0:
                loglik = torch.mm(theta[t], beta[t])
                loglik = torch.log(loglik+1e-6)
                nll.append(10*(-loglik * bow_t[:, t, :]).sum(1).mean())
        nll = torch.stack(nll).sum()
        return nll




    def decode(self, theta, beta, bow_t):
        nll = self.get_likelihood(theta, beta, bow_t)
        return nll

    def forward(self, normalized_bows, bow_t, theta=None, eta=None):
        ## get \theta
        if not self.set_alpha and self.set_eta:
            alpha, kld_alpha = self.get_alpha()
        ## get \beta
            beta = self.get_beta(alpha)
            eta1 = eta
            theta = theta
            nll = self.decode(theta, beta, bow_t)
            return nll + kld_alpha

        elif self.set_alpha and not self.set_eta:
            alpha = self.get_alpha()
            beta = self.get_beta(alpha)
            eta1, kld_eta1 = self.get_eta(bow_t, self.num_times)
            theta, kld_theta, _ = self.get_theta(eta1, normalized_bows)
            nll = self.decode(theta, beta, bow_t)
            return nll + kld_theta + kld_eta1

        ## get prediction loss
        # kld_z1, log_likelihood1, kld_z2, log_likelihood2 = self.decode(theta, beta1, beta2, bows, age)
        # recon_loss = -log_likelihood1-log_likelihood2



