# /usr/bin/python

from __future__ import print_function

import argparse
import torch
import numpy as np
import os
import math
import pickle
from torch.utils.data import DataLoader
from torch import nn, optim
from simulation import ETM
from utils import nearest_neighbors, get_topic_coherence, get_topic_diversity
from dataset import SimulationDataset

parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('--data_path', type=str, default='input_data', help='directory containing data')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--emb_path', type=str, default='input_data', help='directory containing embeddings')

# parser.add_argument('--save_path', type=str, default='./results', help='path to save results')

parser.add_argument('--save_path', type=str, default='results', help='path to save results')

parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')

### model-related arguments
parser.add_argument('--vocab_size', type=int, default=400, help='number of unique drugs')

parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--rho_size', type=int, default=256, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=256, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=128, help='dimension of hidden space of q(theta)')
parser.add_argument('--eta_hidden_size', type=int, default=128, help='dimension of hidden space of q(eta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')
parser.add_argument('--nlayer', type=int, default=3, help='number of layers for theta')
parser.add_argument('--num_times', type=int, default=3, help='number of age periods for eta')
parser.add_argument('--num_visits', type=int, default=3, help='number of visits for eta')
parser.add_argument('--upper', type=int, default=100, help='upper boundary for Gaussian variance')
parser.add_argument('--lower', type=int, default=-100, help='lower boundary for Gaussian variance')



parser.add_argument('--train_rho', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--set_alpha', type=int, default=0, help='whether to fix alpha or train it')
parser.add_argument('--set_eta', type=int, default=0, help='whether to fix eta and theta or train it')
parser.add_argument('--add_freq', type=int, default=0, help='whether to consider baseline frequency or not')

parser.add_argument('--embedding1', type=str, default="vertex_embeddings.npy", help='file contained fixed rho for type1')
parser.add_argument('--embedding2', type=str, default="vertex_embeddings.npy", help='file contained fixed rho for type2')
### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train...150 for 20ng 100 for others')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')

parser.add_argument('--enc_drop', type=float, default=0.0, help='dropout rate on encoder')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate on rnn for eta')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')

parser.add_argument('--nonmono', type=int, default=5, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')
parser.add_argument('--gpu_device', type=str, default="cuda", help='gpu device name')
### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=1, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=100, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')

args = parser.parse_args()

device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)




embed_file = os.path.join(args.data_path, "rho_simulated.npy")
rho_embeddings = np.load(embed_file)
rho_embeddings = torch.from_numpy(rho_embeddings).float().to(device)

alpha_embeddings = None
if args.set_alpha:
    embed_file = os.path.join(args.data_path, "alpha_simulated.npy")
    alpha_embeddings = np.load(embed_file)
    alpha_embeddings = torch.from_numpy(alpha_embeddings).float().to(device)




## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path,
                        'etm_UKPD_K_{}_Htheta_{}_Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
                            args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
                            args.lr, args.batch_size, args.rho_size, args.train_rho))

## define model and optimizer
model = ETM(args.num_topics, args.num_times, args.vocab_size, args.eta_hidden_size,
            args.t_hidden_size, args.rho_size, args.emb_size, args.theta_act,
            args.delta, args.nlayer, args.set_alpha, alpha_embeddings, args.set_eta, args.train_rho,
            rho_embeddings, args.enc_drop, args.eta_dropout, args.upper, args.lower).to(device)

print('model: {}'.format(model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    acc_real_loss = 0
    cnt = 0

    train_filename = os.path.join(args.data_path, "bow_train.npy")
    train_t_filename = os.path.join(args.data_path, "bow_t_train.npy")
    theta_filename = os.path.join(args.data_path, "theta_train.npy")
    eta_filename = os.path.join(args.data_path, "eta_train.npy")
    if not args.set_eta:
        TrainDataset = SimulationDataset(train_filename, train_t_filename)
    else:
        TrainDataset = SimulationDataset(train_filename, train_t_filename, theta_filename, eta_filename)
    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    for idx, (sample_batch, index) in enumerate(TrainDataloader):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch = sample_batch['Data'].float().to(device)
        data_batch_t = sample_batch['Data_t'].float().to(device)

        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        # recon_loss, kld_theta, kld_z1, kld_z2, kld_alpha, kld_eta1 = \
        #     model(data_batch, normalized_data_batch,rnn_inp_age_train, rnn_inp_visits_train, age, visits)
        if args.set_eta:
            theta = sample_batch["Theta"].float().to(device)
            theta = torch.transpose(theta, 0, 1)
            eta = sample_batch["Eta"].float().to(device)
            eta = torch.transpose(eta, 0, 1)
            total_loss = model(normalized_data_batch, data_batch_t, theta, eta)
        else:
            total_loss = model(normalized_data_batch, data_batch_t)

        # total_loss = recon_loss + kld_theta + kld_z1 + kld_z2 + kld_alpha + kld_eta1

        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_real_loss += torch.sum(total_loss).item()

        # acc_kl_eta2_loss += torch.sum(kld_eta2).item()

        cnt += 1



        if idx % args.log_interval == 0 and idx > 0:
            cur_real_loss = round(acc_real_loss / cnt, 2)

            # cur_kl_eta2 = round(acc_kl_eta2_loss / cnt, 2)

            # cur_real_loss = round(cur_loss + cur_kl_theta + cur_kl_z1 + cur_kl_z2 + cur_kl_alpha + cur_kl_eta1, 2)
            # print('Epoch: {} .. batch: {} .. LR: {} .. KL_theta: {} .. KL_z1: {} .. KL_z2: {} .. '
            #           'Rec_loss: {} .. KL_alpha: {} .. KL_eta_age: {} .. NELBO: {}'.
            #         format(epoch, idx, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_kl_z1, cur_kl_z2, cur_loss,
            #                cur_kl_alpha, cur_kl_eta1, cur_real_loss))

            print('Epoch: {} .. batch: {} .. LR: {} .. NELBO: {}'.
                    format(epoch, idx, optimizer.param_groups[0]['lr'], cur_real_loss))


    cur_real_loss = round(acc_real_loss / cnt, 2)

    # cur_kl_eta2 = round(acc_kl_eta2_loss / cnt, 2)
    #
    # cur_real_loss = round(cur_loss + cur_kl_theta + cur_kl_z1 + cur_kl_z2 + cur_kl_alpha + cur_kl_eta1, 2)
    # print('*' * 100)
    # print('Epoch----->{} .. LR: {} .. KL_theta: {} .. KL_z1: {} .. KL_z2: {} .. Rec_loss: {} .. KL_alpha {} ..'
    #       'KL_eta_age: {} .. NELBO: {}'.
    #       format(epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_kl_z1, cur_kl_z2,
    #                            cur_loss, cur_kl_alpha, cur_kl_eta1, cur_real_loss))
    # print('*' * 100)


    print('*' * 100)
    print('Epoch----->{} .. LR: {} ..  NELBO: {}'.
          format(epoch, optimizer.param_groups[0]['lr'], cur_real_loss))
    print('*' * 100)




def get_rnn_input(data_batch, times, num_times, vocab_size):

    rnn_input = torch.zeros(num_times, vocab_size).to(device)


    for t in range(num_times):
        rnn_input[t] = (data_batch[times == t].sum(0))/len(times[times == t])


    return rnn_input

def evaluate(m, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        test_filename = os.path.join(args.data_path, "bow_test.npy")
        test_t_filename = os.path.join(args.data_path, "bow_t_test.npy")
        theta_filename = os.path.join(args.data_path, "theta_test.npy")
        eta_filename = os.path.join(args.data_path, "eta_test.npy")

        if not args.set_eta:
            TestDataset = SimulationDataset(test_filename, test_t_filename)
        else:
            TestDataset = SimulationDataset(test_filename, test_t_filename, theta_filename, eta_filename)
        TestDataloader = DataLoader(TestDataset, batch_size=args.eval_batch_size,
                                    shuffle=True, num_workers=args.num_workers)
        if not args.set_alpha:
            alpha, _ = m.get_alpha()
        else:
            alpha = m.get_alpha()
        beta = m.get_beta(alpha)

        # eta2, _ = m.get_eta(rnn_inp_visits_test, args.num_visits)
        acc_loss = 0
        cnt = 0
        for idx, (sample_batch, index) in enumerate(TestDataloader):
            ### do dc and tc here
            ## get theta from first half of docs
            data_batch = sample_batch["Data"].float().to(device)
            data_batch_t = sample_batch["Data_t"].float().to(device)
            if not args.set_eta:
                eta1, _ = m.get_eta(data_batch_t, args.num_times)

            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            if not args.set_eta:
                theta, _, _ = m.get_theta(eta1, normalized_data_batch)
            else:
                theta = sample_batch['Theta'].float().to(device)
                theta = torch.transpose(theta, 0, 1)

            # print("sums_2: {}".format(sums_2.squeeze()))
            # _, log_likelihood1, _, log_likelihood2 = m.decode(theta, beta1, beta2, data_batch, age)
            # recon_loss = -log_likelihood1 - log_likelihood2
            nll = m.decode(theta, beta, data_batch_t)
            recon_loss = nll / 10

            loss = recon_loss
            # loss = loss.mean().item()
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        # ppl_dc = round(math.exp(cur_loss), 1)
        print('*' * 100)
        print('Test Negative Log Likelihood: {}'.format(cur_loss))
        print('*' * 100)

        TQ = TC = TD = 0

        if tc or td:
            beta1 = beta1.data.cpu().numpy()
            beta2 = beta2.data.cpu().numpy()
        if tc:
            print('Computing topic coherence...')
            TC_all, cnt_all = get_topic_coherence(beta, train_tokens, vocab)

            TC_all = torch.tensor(TC_all)
            cnt_all = torch.tensor(cnt_all)
            TC_all = TC_all / cnt_all
            TC_all[TC_all < 0] = 0

            TC = TC_all.mean().item()
            print('Topic Coherence is: ', TC)
            print('\n')

        if td:
            print('Computing topic diversity...')
            TD_all = get_topic_diversity(beta, 25)
            TD = np.mean(TD_all)
            print('Topic Diversity is: {}'.format(TD))

            print('Get topic quality...')
            TQ = TD * TC
            print('Topic Quality is: {}'.format(TQ))
            print('#' * 100)

        return cur_loss, TQ, TC, TD


if args.mode == 'train':
    ## train model on data
    best_epoch = 0
    best_val_ppl = 1e9
    all_val_ppls = []
    # print('\n')
    # print('Visualizing model quality before training...')
    # visualize(model)
    # print('\n')

    for epoch in range(1, args.epochs):
        train(epoch)
        val_ppl, tq, tc, td = evaluate(model)
        if val_ppl < best_val_ppl or not os.path.exists(ckpt):
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (
                    len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        # if epoch % args.visualize_every == 0:
        #     visualize(model)
        all_val_ppls.append(val_ppl)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    val_ppl = evaluate(model)
else:
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)

model.eval()



if not args.set_alpha:
    alpha, _ = model.get_alpha()
    beta = model.get_beta(alpha)
    beta = beta.detach().cpu().numpy()
    saved_file = os.path.join(args.save_path, "beta.npy")
    np.save(saved_file, beta)
    alpha = alpha.detach().cpu().numpy()
    saved_alpha = os.path.join(args.save_path, "alpha.npy")
    np.save(saved_alpha, alpha)

else:
    alpha = model.get_alpha()
    beta = model.get_beta(alpha)

#
#
#
    filename = os.path.join(args.data_path, "bow.npy")
    filename_t = os.path.join(args.data_path, "bow_t.npy")

# filename = os.path.join("802_444_mix_data", "bow.npy")
# filename_t = os.path.join("802_444_mix_data", "bow_t.npy")

    Dataset = SimulationDataset(filename, filename_t)
    MyDataloader = DataLoader(Dataset, batch_size=1000,
                             shuffle=False, num_workers=args.num_workers)
#
# # eta2, _ = model.get_eta(rnn_inp_visits, args.num_visits)
#
    index_list = []
    for idx, (sample_batch, index) in enumerate(MyDataloader):
        index_list.append(index.cpu().numpy())
        data_batch = sample_batch["Data"].float().to(device)
        data_batch_t = sample_batch["Data_t"].float().to(device)
        eta1, _ = model.get_eta(data_batch_t, args.num_times)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        theta, _, _ = model.get_theta(eta1, normalized_data_batch)
        theta = theta.detach().cpu().numpy()
        saved_folder = os.path.join(args.save_path, "theta")
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
        saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
        np.save(saved_theta, theta)
        saved_index = os.path.join(saved_folder, "index.pkl")
        with open(saved_index, "wb") as f:
            pickle.dump(index_list, f)
#
#
#
