#/usr/bin/python

from __future__ import print_function

import argparse
import torch
import numpy as np
import os 
import math
from torch.utils.data import DataLoader
from torch import optim
from scripts.etm import ETM
from scripts.utils import get_topic_coherence, get_topic_diversity
from scripts.dataset import PatientDrugDataset
import pickle


parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('--data_path', type=str, default='input_data', help='directory containing data')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--emb_path', type=str, default='input_data', help='directory containing embeddings')
parser.add_argument('--ratio_file', type=str, default='binary_ratio.npy', help='weights for focal loss')


# parser.add_argument('--save_path', type=str, default='./results', help='path to save results')

parser.add_argument('--save_path', type=str, default='results', help='path to save results')

parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for training')

### model-related arguments
parser.add_argument('--vocab_size', type=int, default=787, help='number of unique drugs')
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--num_classes', type=int, default=8, help='number of classes')
parser.add_argument('--pred_nlayer', type=int, default=3, help='number of layers for prediction')
parser.add_argument('--gamma', type=float, default=2, help='hyperparameter fo rfocal loss')
parser.add_argument('--rho_size', type=int, default=256, help='dimension of rho')
parser.add_argument('--emb_size', type=int, default=256, help='dimension of embeddings')
parser.add_argument('--t_hidden_size', type=int, default=128, help='dimension of hidden space of q(theta)')
parser.add_argument('--lstm_hidden_size', type=int, default=5, help='dimension of hidden space of q(eta)')
parser.add_argument('--theta_act', type=str, default='relu', help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--predcoef', type=int, default=10, help='coefficient for prediction loss')
parser.add_argument('--num_times', type=int, default=7, help='number of age periods for eta')
parser.add_argument('--train_embeddings', type=int, default=1, help='whether to include pretrained embedding')
parser.add_argument('--rho_fixed', type=int, default=0, help='whether to fix rho or train it')
parser.add_argument('--e2e', type=int, default=0, help='whether to do end to end training')

parser.add_argument('--embedding', type=str, default="vertex_embeddings.npy", help='file contained prettrained rho')
parser.add_argument('--label_name', type=str, default="y_age", help='file contained age label')
parser.add_argument('--mask_name', type=str, default="mask_age", help='file contained mask for age label')
parser.add_argument('--X_name', type=str, default="bow", help='file contained input for age label')
parser.add_argument('--Xt_name', type=str, default="bow_t", help='file contained time-varying input for age label')
### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')

parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train...150 for 20ng 100 for others')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')


parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout rate on rnn for prediction')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')

parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=1, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')
parser.add_argument('--gpu_device', type=str, default="cuda", help='gpu device name')

args = parser.parse_args()

device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)



embeddings = None
if not args.train_embeddings:
    embed_file = os.path.join(args.data_path, args.embedding)
    embeddings = np.load(embed_file)
    embeddings = torch.from_numpy(embeddings).float().to(device)

if args.e2e:
    loss_weights = np.load(os.path.join(args.data_path, args.ratio_file))

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path, 
        'etm_UKPD_K_{}_Htheta_{} Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
        args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
            args.lr, args.batch_size, args.rho_size, args.train_embeddings))

## define model and optimizer
model = ETM(args.num_topics, args.num_times, args.vocab_size, args.t_hidden_size, args.rho_size, args.emb_size,
                args.theta_act, args.gamma, embeddings, args.train_embeddings, args.rho_fixed,
                 args.enc_drop, args.e2e, args.lstm_hidden_size,
                args.pred_nlayer, args.num_classes, args.lstm_dropout, args.predcoef).to(device)

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
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0

    train_t_filename = os.path.join(args.data_path, f"{args.X_name}_train.npy")
    if not args.e2e:
        y_filename = None
        mask_filename = None
    else:
        y_filename = os.path.join(args.data_path, f"{args.label_name}_train.npy")
        mask_filename = os.path.join(args.data_path, f"{args.mask_name}_train.npy")
        acc_pred_loss = 0
    TrainDataset = PatientDrugDataset(train_t_filename, y_filename, mask_filename)
    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    for idx, (sample_batch, index) in enumerate(TrainDataloader):

        optimizer.zero_grad()
        model.zero_grad()
        data_batch = sample_batch['Data'].float().to(device)
        # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0)*data_batch_t.size(1), data_batch_t.size(2))
        if args.e2e:
            y = sample_batch["Y"].long().to(device)
            mask = sample_batch["Mask"].to(device)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        if not args.e2e:
            recon_loss, kld_theta = model(data_batch, normalized_data_batch)

        # NELBO = -(loglikelihood - KL[q||p]) = -loglikelihood + KL[q||p]
            total_loss = recon_loss + kld_theta
        else:
            recon_loss, kld_theta, pred_loss = model(data_batch, normalized_data_batch, y, mask, loss_weights)
            total_loss = recon_loss + kld_theta + pred_loss
        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()
        if args.e2e:
            acc_pred_loss += torch.sum(pred_loss).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            if not args.e2e:
                cur_real_loss = round(cur_loss + cur_kl_theta, 2)

                print('Epoch: {} .. batch: {}/320 .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
                    epoch, idx, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
            else:
                cur_pred_loss = round(acc_pred_loss / cnt, 2)
                cur_real_loss = round(cur_loss + cur_kl_theta + cur_pred_loss, 2)
                print('Epoch: {} .. batch: {}/320 .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. Pred_loss: {} .. NELBO: {}'.format(
                    epoch, idx, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_pred_loss, cur_real_loss))

    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    if not args.e2e:
        cur_real_loss = round(cur_loss + cur_kl_theta, 2)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
        print('*'*100)
    else:
        cur_pred_loss = round(acc_pred_loss / cnt, 2)
        cur_real_loss = round(cur_loss + cur_kl_theta + cur_pred_loss, 2)
        print('*'*100)
        print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. Pred_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_pred_loss, cur_real_loss))
        print('*'*100)

# def visualize(m, show_emb=True):
#     if not os.path.exists('./results'):
#         os.makedirs('./results')
#
#     m.eval()
#
#     if args.dataset == "20ng":
#         queries = ['andrew', 'computer', 'sports', 'religion', 'man', 'love',
#                     'intelligence', 'money', 'politics', 'health', 'people', 'family']
#     else:
#         queries = ['border', 'vaccines', 'coronaviruses', 'masks']
#
#     ## visualize topics using monte carlo
#     with torch.no_grad():
#         print('#'*100)
#         print('Visualize topics...')
#         topics_words = []
#         gammas = m.get_beta()
#         for k in range(args.num_topics):
#             gamma = gammas[k]
#             top_words = list(gamma.cpu().numpy().argsort()[-args.num_words+1:][::-1])
#             topic_words = [vocab[a] for a in top_words]
#             topics_words.append(' '.join(topic_words))
#             print('Topic {}: {}'.format(k, topic_words))
#
#         if show_emb:
#             ## visualize word embeddings by using V to get nearest neighbors
#             print('#'*100)
#             print('Visualize word embeddings by using output embedding matrix')
#             try:
#                 embeddings = m.rho.weight  # Vocab_size x E
#             except:
#                 embeddings = m.rho         # Vocab_size x E
#             neighbors = []
#             for word in queries:
#                 print('word: {} .. neighbors: {}'.format(
#                     word, nearest_neighbors(word, embeddings, vocab)))
#             print('#'*100)

def evaluate(m, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        test_t_filename = os.path.join(args.data_path, f"{args.X_name}_test.npy")

        if not args.e2e:
            y_filename = None
            mask_filename = None
        else:
            y_filename = os.path.join(args.data_path, f"{args.label_name}_test.npy")
            mask_filename = os.path.join(args.data_path, f"{args.mask_name}_test.npy")
        TestDataset = PatientDrugDataset(test_t_filename, y_filename, mask_filename)
        TestDataloader = DataLoader(TestDataset, batch_size=args.eval_batch_size,
                                    shuffle=True, num_workers=args.num_workers)
        beta = m.get_beta()
        acc_loss = 0
        cnt = 0
        for idx, (sample_batch, index) in enumerate(TestDataloader):
            ### do dc and tc here
            ## get theta from first half of docs
            data_batch= sample_batch['Data'].float().to(device)
            # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0) * data_batch_t.size(1),
            #                                                          data_batch_t.size(2))
            if args.e2e:
                y = sample_batch["Y"].long().to(device)
                mask = sample_batch["Mask"].to(device)
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _, x= m.get_theta(normalized_data_batch)
            if args.e2e:
                bsize = int(theta.shape[0] / args.num_times)
                x = x.reshape(args.num_times, bsize, args.num_topics)
                hidden = m.pred_init_hidden(bsize)
                x, _ = m.lstm(torch.transpose(x, 0, 1), hidden)
                x = m.linear(x)
                y_pred = m.output(x)
                pred_loss = m.calc_pred_loss(y, y_pred, mask, loss_weights, args.gamma)

            res = torch.mm(theta, beta)
            preds = torch.log(res)
            recon_loss = -(preds * data_batch).sum(1)
            
            if not args.e2e:
                loss = recon_loss.mean().item()
            else:
                loss = recon_loss.mean().item() + pred_loss
            acc_loss += loss
            cnt += 1
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print('*'*100)
        print('Test Doc Completion PPL: {}'.format(ppl_dc))
        print('*'*100)

        TQ = TC = TD = 0
        
        if tc or td:
            beta = beta.data.cpu().numpy()

        if tc:
            print('Computing topic coherence...')
            TC_all, cnt_all = get_topic_coherence(beta, train_tokens, vocab)

            TC_all = torch.tensor(TC_all)
            cnt_all = torch.tensor(cnt_all)
            TC_all = TC_all / cnt_all
            TC_all[TC_all<0] = 0

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
            print('#'*100)

        return ppl_dc, TQ, TC, TD

if args.mode == 'train':
    ## train model on data 
    best_epoch = 0
    best_val_ppl = 1e100
    all_val_ppls = []
    # print('\n')
    # print('Visualizing model quality before training...')
    # visualize(model)
    # print('\n')
    file = open(f'{args.save_path}/perplexity_{args.num_topics}.txt', 'w')
    for epoch in range(1, args.epochs):
        train(epoch)
        val_ppl, tq, tc, td = evaluate(model)
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor
        # if epoch % args.visualize_every == 0:
        #     visualize(model)
        s1 = f"Perplexity at epoch {epoch}: " + str(best_val_ppl)
        file.write(s1 + '\n')
        all_val_ppls.append(val_ppl)
    file.close()
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    val_ppl = evaluate(model)
else:   
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    
model.eval()

# saved_nelbo = os.path.join(args.save_path,"nelbo.npy")
# np.save(saved_nelbo, np.array(loss_epochs))
## get document completion perplexities and topic quality
# test_ppl, tq, tc, td = evaluate(model, 'test', tc=True, td=True)
#
# f=open(ckpt+'_tq.txt','w')
# s1="Topic Quality: "+str(tq)
# s2="Topic Coherence: "+str(tc)
# s3="Topic Diversity: "+str(td)
# f.write(s1+'\n'+s2+'\n'+s3+'\n')
# f.close()
#
# f=open(ckpt+'_tq.txt','r')
# [print(i,end='') for i in f.readlines()]
# f.close()

## get most used topics
# indices = torch.tensor(range(args.num_docs_train))
# indices = torch.split(indices, args.batch_size)
# thetaAvg = torch.zeros(1, args.num_topics).to(device)
# thetaWeightedAvg = torch.zeros(1, args.num_topics).to(device)
# cnt = 0
# for idx, ind in enumerate(indices):
#     data_batch = data.get_batch(train_tokens, train_counts, ind, args.vocab_size, device)
#     sums = data_batch.sum(1).unsqueeze(1)
#     cnt += sums.sum(0).squeeze().cpu().numpy()
#     if args.bow_norm:
#         normalized_data_batch = data_batch / sums
#     else:
#         normalized_data_batch = data_batch
#     theta, _ = model.get_theta(normalized_data_batch)
#     thetaAvg += theta.sum(0).unsqueeze(0) / args.num_docs_train
#     weighed_theta = sums * theta
#     thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
#     if idx % 100 == 0 and idx > 0:
#         print('batch: {}/{}'.format(idx, len(indices)))
# thetaWeightedAvg = thetaWeightedAvg.squeeze().detach().cpu().numpy() / cnt
# print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))

## show topics
beta = model.get_beta().detach().cpu().numpy()
saved_file = os.path.join(args.save_path, "beta.npy")
np.save(saved_file, beta)

saved_rho = os.path.join(args.save_path, "rho.npy")

try:
    rho = model.rho.weight.detach().cpu().numpy()
except:
    rho = model.rho.detach().cpu().numpy()
np.save(saved_rho, rho)


saved_alpha = os.path.join(args.save_path, "alpha.npy")
alpha = model.alphas.weight.detach().cpu().numpy()
np.save(saved_alpha, alpha)
# topic_indices = list(np.random.choice(args.num_topics, 10)) # 10 random topics
# print('\n')
# for k in range(args.num_topics):#topic_indices:
#     gamma = beta[k]
#     top_words = list(gamma.detach().cpu().numpy().argsort()[-args.num_words+1:][::-1])
#     topic_words = [vocab[a] for a in top_words]
#     print('Topic {}: {}'.format(k, topic_words))


filename_t = os.path.join(args.data_path, f"{args.X_name}_train.npy")

# filename = os.path.join("802_444_mix_data", "bow.npy")
# filename_t = os.path.join("802_444_mix_data", "bow_t.npy")

Dataset = PatientDrugDataset(filename_t)
MyDataloader = DataLoader(Dataset, batch_size=1000,
                             shuffle=False, num_workers=args.num_workers)
#
# # eta2, _ = model.get_eta(rnn_inp_visits, args.num_visits)
#
index_list = []
for idx, (sample_batch, index) in enumerate(MyDataloader):
    index_list.append(index.cpu().numpy())
    data_batch= sample_batch['Data'].float().to(device)
    # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0) * data_batch_t.size(1),
    #                                                          data_batch_t.size(2))
    theta, _, mu_theta= model.get_theta(data_batch)

    theta = theta.detach().cpu().numpy()
    mu_theta = mu_theta.detach().cpu().numpy()

    saved_folder = os.path.join(args.save_path, "theta_train")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    saved_mu = os.path.join(saved_folder, f"mu_theta{idx}.npy")


    np.save(saved_theta, theta)
    np.save(saved_mu, mu_theta)

saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)





filename_t = os.path.join(args.data_path, f"{args.X_name}_test.npy")
if not args.e2e:
    y_filename = None
    mask_filename = None
else:
    y_filename = os.path.join(args.data_path, f"{args.label_name}_test.npy")
    mask_filename = os.path.join(args.data_path, f"{args.mask_name}_test.npy")
Dataset = PatientDrugDataset(filename_t, y_filename, mask_filename)
MyDataloader = DataLoader(Dataset, batch_size=1000,
                          shuffle=False, num_workers=args.num_workers)
#
# # eta2, _ = model.get_eta(rnn_inp_visits, args.num_visits)
#
index_list = []
for idx, (sample_batch, index) in enumerate(MyDataloader):
    index_list.append(index.cpu().numpy())
    data_batch = sample_batch['Data'].float().to(device)
    # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0) * data_batch_t.size(1),
    #                                                          data_batch_t.size(2))
    if args.e2e:
        y = sample_batch["Y"].long().to(device)
        mask = sample_batch["Mask"].to(device)
    theta, _, mu_theta = model.get_theta(data_batch)
    if args.e2e:
        bsize = int(theta.shape[0] / args.num_times)
        x = x.reshape(args.num_times, bsize, args.num_topics)
        hidden = model.pred_init_hidden(bsize)
        x, _ = model.lstm(torch.transpose(x, 0, 1), hidden)
        x = model.linear(x)
        y_pred = model.output(x)


    theta = theta.detach().cpu().numpy()
    mu_theta = mu_theta.detach().cpu().numpy()

    saved_folder = os.path.join(args.save_path, "theta_test")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    saved_mu = os.path.join(saved_folder, f"mu_theta{idx}.npy")

    if args.e2e:
        saved_test = os.path.join(args.save_path, f"y_prob{idx}.npy")
        np.save(saved_test, y_pred.detach().cpu().numpy())
    np.save(saved_theta, theta)
    np.save(saved_mu, mu_theta)

saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)

