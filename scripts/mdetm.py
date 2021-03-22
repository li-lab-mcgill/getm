import torch
import torch.nn.functional as F
import numpy as np
import math

from torch import nn

from IPython import embed


def set_bounds(mat, upper, lower):
    y_upper = (torch.ones(mat.size())*upper).float().to(mat.device)
    y_lower = (torch.ones(mat.size())*lower).float().to(mat.device)
    mat = torch.where(mat >= lower, mat, y_lower)
    mat = torch.where(mat <= upper, mat, y_upper)
    return mat

class MDETM(nn.Module):
    def __init__(self, device, num_topics, num_times, code_types, vocab_size, eta_hidden_size, t_hidden_size,
                 rho_size, emsize, theta_act, delta, nlayer,
                 # adj_mat12, adj_mat11, adj_mat22, lambda12, lambda11, lambda22, 
                 constraint, 
                 embeddings, train_embeddings, enc_drop=0.1, eta_drop=0.1, upper=100, lower=-100):
        super(MDETM, self).__init__()
        self.device = device

        ## define hyperparameters
        self.num_topics = num_topics
        self.code_types = code_types
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size

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
        self.recon_coef = 10

        self.constraint = constraint

        self.train_embeddings = train_embeddings  # [3]
        self.eta_hidden_size = eta_hidden_size

        self.upper = upper
        self.lower = lower

        ## define the word embedding matrix \rho
        self.rho = {}
        for i, c in enumerate(self.code_types):
            if self.train_embeddings[i]:
                self.rho[c] = nn.Linear(rho_size, vocab_size[i], bias=True)  # L x V
            else:
                num_embeddings, emsize = embeddings[i].size()
                self.rho[c] = nn.Embedding(num_embeddings, emsize)
                # self.rho[c] = nn.Embedding.from_pretrained(embedding[i])
                # embeddings[i] is of shape (num_embeddings, embedding_dim)
                self.rho[i].weight.data.copy_(embeddings[i])
        self.rho = nn.ModuleDict(self.rho)

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
        self.q_theta = nn.Sequential(
                    nn.Linear(sum(self.vocab_size)+self.num_topics, self.t_hidden_size),
                    self.theta_act,
                    nn.Linear(self.t_hidden_size, self.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(self.t_hidden_size, self.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(sum(self.vocab_size), self.eta_hidden_size)
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
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu)**2) / (sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1).mean()
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1).mean()
        return kl

    def get_alpha(self):  ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)

        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :])

            p_mu_t = alphas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))

            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).mean()
        return alphas, kl_alpha

    def get_eta(self, bow_t, num_times, visualize=False): ## structured amortized inference
        batch_size = bow_t.size()[0]
        # FIXME:  sparse tensor
        if bow_t.is_sparse:
            inp = self.q_eta_map(bow_t.to_dense())
        else:
            inp = self.q_eta_map(bow_t)
        hidden = self.init_hidden(batch_size)
        output, _ = self.q_eta(inp, hidden)


        etas = torch.zeros(num_times, batch_size, self.num_topics).to(self.device)
        kl_eta = []

        inp_0 = torch.cat([output[:, 0, :], torch.zeros(batch_size, self.num_topics).to(self.device)], dim=1)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        logsigma_0 = set_bounds(logsigma_0, self.upper, self.lower)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(batch_size, self.num_topics).to(self.device)
        logsigma_p_0 = torch.zeros(batch_size, self.num_topics).to(self.device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, num_times):
            inp_t = torch.cat([output[:, t, :], etas[t - 1]], dim=1)
            mu_t = self.mu_q_eta(inp_t)
            if not visualize: 
                logsigma_t = self.logsigma_q_eta(inp_t)
                logsigma_t = set_bounds(logsigma_t, self.upper, self.lower)
            else:
                logsigma_t = torch.zeros(mu_t.shape).to(self.device)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t - 1]
            logsigma_p_t = torch.log(self.delta * torch.ones(batch_size, self.num_topics).to(self.device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)

            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()

        return etas, kl_eta


    def get_theta(self, eta_age, bows, visualize=False): ## amortized inference
        """Returns the topic proportions.
        """
        # if len(bows.size()) == 2:
            # bows = torch.stack([bows for _ in range(self.num_times)])
        # else:
        bows = torch.transpose(bows, 0, 1)
        # FIXME:  sparse tensor
        if bows.is_sparse:
            inp = torch.cat([bows.to_dense(), eta_age], dim=-1)
        else:
            inp = torch.cat([bows, eta_age], dim=-1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        if not visualize:
            logsigma_theta = self.logsigma_q_theta(q_theta)
            logsigma_theta = set_bounds(logsigma_theta, self.upper, self.lower)
        else:
            logsigma_theta = torch.zeros(mu_theta.shape).to(self.device)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_age, torch.zeros(self.num_topics).to(self.device))
        return theta, kl_theta, z

    def init_hidden(self, bsize):
        """Initializes the first hidden state of the RNN used as inference network for theta.
        """
        weight = next(self.parameters())
        nlayers = self.nlayer
        nhid = self.eta_hidden_size

        return (weight.new_zeros(nlayers, bsize, nhid), weight.new_zeros(nlayers, bsize, nhid))

    def calc_beta(self, alpha, rho, train_embeddings):
        """Returns the topic matrix \beta of shape  T x K x V
        """
        if train_embeddings:
            logit = rho(alpha.view(alpha.size(0) * alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0) * alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, rho.weight.permute(1, 0))
            # logit = torch.mm(tmp, rho.permute(1, 0))
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta

    def get_beta(self, alpha):
        beta = {}
        for i, c in enumerate(self.code_types):
            beta[c] = self.calc_beta(alpha, self.rho[c], self.train_embeddings[i])
        return beta

    def calc_constraints(self, beta):
        con_loss = [torch.zeros(1).to(self.device)]
        for t in range(self.num_times):
            constraints = []
            for (c1, c2), (adj_mat, lambda_c) in self.constraint.items():
                ccc = torch.mm(torch.mm(beta[c1], self.adj_mat), beta[c2].T)
                constraints.append(lambda_c * torch.trace(ccc))
            con_loss.append(-sum(constraints))
        con_loss = sum(con_loss)

        return con_loss

    def get_likelihood(self, theta, beta, bow_t):
        nll = []
        for t in range(self.num_times):
            if len(theta[t]) > 0:
                loglik = torch.mm(theta[t], beta[t])
                loglik = torch.log(loglik+1e-6)
                if bow_t.is_sparse:
                    nll.append(self.recon_coef*(-loglik * bow_t[:, t, :].to_dense()).sum(1).mean())
                else:
                    nll.append(self.recon_coef*(-loglik * bow_t[:, t, :]).sum(1).mean())
        nll = torch.stack(nll).sum()
        return nll

    # def decode(self, theta, beta1, beta2, bows, age):
    #     bows1 = bows[:, :self.vocab_size1]
    #     bows2 = bows[:, -self.vocab_size2:]
    #
    #     kld_z1, log_likelihood1 = self.get_likelihood(theta, beta1, bows1, age, self.base_freq1)
    #     kld_z2, log_likelihood2 = self.get_likelihood(theta, beta2, bows2, age, self.base_freq2)
    #
    #     return kld_z1, log_likelihood1, kld_z2, log_likelihood2

    # def get_coherence(self, beta1, beta2):
    #     cll1 = []
    #     cll2 = []
    #     for t in range(self.num_times):
    #         loglik1 = torch.mm(beta1[t], self.base_cat1).sum()
    #         cll1.append(-loglik1)
    #         loglik2 = torch.mm(beta2[t], self.base_cat2).sum()
    #         cll2.append(-loglik2)
    #     return torch.stack(cll1).sum(), torch.stack(cll2).sum()


    def decode(self, theta, beta, bow_t):
        nll = torch.zeros(len(self.code_types)).to(self.device)
        partition = np.cumsum([0]+self.vocab_size)
        for i, c in enumerate(self.code_types):
            bows_i = bow_t.to_dense()[:, :, partition[i]:partition[i+1]]
            nll[i] = self.get_likelihood(theta, beta[c], bows_i)
        return nll

    def forward(self, bows, bow_t):
        ## get \theta

        alpha, kld_alpha = self.get_alpha()
        ## get \beta
        beta = self.get_beta(alpha)
        eta1, kld_eta1 = self.get_eta(bow_t, self.num_times)
        theta, kld_theta, _, = self.get_theta(eta1, bows)

        ## get prediction loss
        # kld_z1, log_likelihood1, kld_z2, log_likelihood2 = self.decode(theta, beta1, beta2, bows, age)
        # recon_loss = -log_likelihood1-log_likelihood2
        nll = self.decode(theta, beta, bow_t)
        recon_loss = nll.sum()
        cond_loss = self.calc_constraints(beta)

        # return recon_loss, kld_theta, kld_z1, kld_z2, kld_alpha, kld_eta1
        return recon_loss, kld_theta, kld_alpha, kld_eta1, cond_loss


