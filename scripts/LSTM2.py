import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pickle
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import LSTMDataset

parser = argparse.ArgumentParser(description='LSTM Multilabel Classification')

parser.add_argument('--data_path_input', type=str, default='LSTM_data', help='directory containing input data')
parser.add_argument('--data_path_label', type=str, default='LSTM_data', help='directory containing label data')
parser.add_argument('--save_path', type=str, default='results_LSTM', help='directory saving data')
parser.add_argument('--X_name', type=str, default="theta", help="variable name of input")
parser.add_argument('--Y_name', type=str, default="np", help="variable name of label")
parser.add_argument('--alpha', type=float, default=100, help='Coefficient for loss mean')
parser.add_argument('--beta', type=float, default=1, help='Coefficient for cross entropy')
parser.add_argument('--nlayers', type=int, default=2, help='Number of hidden layers')
parser.add_argument('--num_times', type=int, default=2, help='Number of time intervals')
parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
parser.add_argument('--input_dim', type=int, default=64, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=64, help='Number of hidden dimension')
parser.add_argument('--mean_loss', type=int, default=1, help='Whether to use target at each time point')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--nonmono', type=int, default=5, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')
parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--gpu_device', type=str, default="cuda", help='gpu device name')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=100, help='input batch size for testing')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
parser.add_argument('--log_interval', type=int, default=1, help='when to log training')
parser.add_argument('--thresh', type=int, default=0, help='whether to specify threshhold value')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')



args = parser.parse_args()

device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

thr = None
if args.thresh:
    thr = np.load(os.path.join(args.data_path, "thresh.npy"))
    thr = torch.from_numpy(thr).float()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
ckpt = os.path.join(args.save_path, "best_model")

class LSTMClassifier(nn.Module):
    def __init__(self, alpha, hidden_dim, input_dim, nlayers, num_times, num_classes, beta, mean_loss=True, drop_out=0.0):
        super(LSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.nlayers = nlayers
        self.hidden_dim = hidden_dim

        self.alpha = alpha
        self.beta = beta
        self.num_times = num_times
        self.mean_loss = mean_loss

        self.lstm = nn.LSTM(input_dim, hidden_dim, nlayers, batch_first=True, dropout=drop_out)
        self.linear = nn.Linear(hidden_dim, self.num_classes, bias=True)
        self.output = nn.Sigmoid()


    def init_hidden(self, bsize):
        """Initializes the first hidden state of the RNN used as inference network for theta.
        """
        weight = next(self.parameters())
        nlayers = self.nlayers
        nhid = self.hidden_dim

        return (weight.new_zeros(nlayers, bsize, nhid), weight.new_zeros(nlayers, bsize, nhid))

    def binary_cross_entropy(self, y_true, y_pred, alpha, beta):
        loss = -alpha*y_true*torch.log(y_pred)-beta*(1-y_true)*torch.log(1-y_pred)
        return torch.sum(loss)

    def calc_mean_loss(self, y, output, mask):
        # weights = [9]
        # class_weights = torch.FloatTensor(weights).to(device)
        # loss = nn.BCELoss(weight=class_weights)
        y_true = y[mask]
        y_pred = output[mask]
        # return 10*loss(y_pred, y_true)
        return self.binary_cross_entropy(y_true, y_pred, self.alpha, self.beta)

    def calc_last_loss(self, y, output):
        loss = nn.BCELoss()
        l = loss(output[-1], y[-1])
        return 10*l

    def get_lstm_output(self, x, bsize):
        hidden = self.init_hidden(bsize)
        x, _ = self.lstm(x, hidden)


    def forward(self, x, y, batch_size):
        hidden = self.init_hidden(batch_size)
        x, _ = self.lstm(x, hidden)
        x = self.linear(x[:, :-1, :])
        output = self.output(x)

        return output


def train(model, epoch, optimizer):
    model.train()
    acc_loss = 0
    cnt = 0

    theta_filename = os.path.join(args.data_path_input, f"{args.X_name}_train.npy")
    label_filename = os.path.join(args.data_path_label, f"label_train_{args.Y_name}.npy")
    mask_filename = os.path.join(args.data_path_label, f"mask_train_{args.Y_name}.npy")

    TrainDataset = LSTMDataset(theta_filename, label_filename, mask_filename)
    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    for idx, (sample_batch, index) in enumerate(TrainDataloader):
        optimizer.zero_grad()
        model.zero_grad()
        x = sample_batch['Data'].float().to(device)
        y = sample_batch['Target'].float().to(device)
        mask = sample_batch['Mask'].to(device)
        bsize = x.size(0)
        output = model(x, y, bsize)
        if args.mean_loss:
            loss = model.calc_mean_loss(y, output, mask)
        else:
            loss = model.calc_last_loss(y, output)
        loss.backward()
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 4)
            print('Epoch: {} .. batch: {} .. Loss: {} .. '.format(epoch, idx, cur_loss))
    cur_loss = round(acc_loss / cnt, 4)
    print('*' * 100)
    print('Epoch----->{} .. Loss: {} .. '.format(epoch, cur_loss))
    print('*' * 100)


def determine_label(output, thr=None):
    if not thr:
        thr = torch.tensor([0.5 for i in range(args.num_classes)]).float().to(device)
    y_pred = output > thr
    y_pred = y_pred.float().to(device)
    return y_pred

def calc_acc(y, y_pred):
    total_cnt = y.size(0) * y.size(1)
    num_true = (torch.sum(y == y_pred)).float()
    return num_true / total_cnt

def validate(model):
    model.eval()
    with torch.no_grad():
        theta_filename = os.path.join(args.data_path_input, f"{args.X_name}_test.npy")
        label_filename = os.path.join(args.data_path_label, f"label_test_{args.Y_name}.npy")
        mask_filename = os.path.join(args.data_path_label, f"mask_test_{args.Y_name}.npy")

        ValDataset = LSTMDataset(theta_filename, label_filename, mask_filename)
        ValDataloader = DataLoader(ValDataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.num_workers)
        accuracy = 0
        cnt = 0

        for idx, (sample_batch, index) in enumerate(ValDataloader):
            x = sample_batch['Data'].float().to(device)
            y = sample_batch['Target'].float().to(device)
            mask = sample_batch['Mask'].to(device)
            bsize = x.size(0)
            output = model(x, y, bsize)
            y_true = y[mask]
            y_pred = determine_label(output, thr)
            y_pred = y_pred[mask]
            acc = calc_acc(y_true, y_pred)
            accuracy += acc
            cnt += 1
        cur_acc = accuracy / cnt
        print('*' * 100)
        print('Validation accuracy: {}'.format(cur_acc))
        print('*' * 100)
        return cur_acc







def main():
    model = LSTMClassifier(args.alpha, args.hidden_dim, args.input_dim, args.nlayers, args.num_times, args.num_classes,
                           args.beta, args.mean_loss, args.dropout).to(device)
    print('model: {}'.format(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    best_epoch = 0
    best_val_acc = 0
    all_val_accs = []
    for epoch in range(1, args.epochs):
        train(model, epoch, optimizer)
        # val_acc = validate(model)
        with open(ckpt, 'wb') as f:
            torch.save(model, f)
        # if val_acc < best_val_acc or not os.path.exists(ckpt):
        #     with open(ckpt, 'wb') as f:
        #         torch.save(model, f)
        #     best_epoch = epoch
        #     best_val_acc = val_acc
        # else:
        #     ## check whether to anneal lr
        #     lr = optimizer.param_groups[0]['lr']
        #     if args.anneal_lr and (
        #             len(all_val_accs) > args.nonmono and val_acc > min(all_val_accs[:-args.nonmono]) and lr > 1e-5):
        #         optimizer.param_groups[0]['lr'] /= args.lr_factor
        # all_val_accs.append(val_acc)
        # print(best_epoch)
    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()

    theta_filename = os.path.join(args.data_path_input, f"{args.X_name}_test.npy")
    label_filename = os.path.join(args.data_path_label, f"label_test_{args.Y_name}.npy")
    mask_filename = os.path.join(args.data_path_label, f"mask_test_{args.Y_name}.npy")

    TestDataset = LSTMDataset(theta_filename, label_filename, mask_filename)
    TestDataloader = DataLoader(TestDataset, batch_size=args.test_batch_size,
                               shuffle=False, num_workers=args.num_workers)

    index_list = []
    for idx, (sample_batch, index) in enumerate(TestDataloader):
        index_list.append(index.cpu().numpy())
        x = sample_batch['Data'].float().to(device)
        y = sample_batch['Target'].float().to(device)
        mask = sample_batch['Mask'].to(device)
        bsize = x.size(0)
        output = model(x, y, bsize)
        y_true = y[mask]
        y_pred = determine_label(output, thr)
        y_pred = y_pred[mask]
        acc = calc_acc(y_true, y_pred)
        print('*' * 100)
        print('Testing accuracy: {}'.format(round(acc.item(), 4)))
        print('*' * 100)
        saved_test = os.path.join(args.save_path, f"y_prob{idx}.npy")
        np.save(saved_test, output.detach().cpu().numpy())
    index_file = os.path.join(args.save_path, "index.pkl")
    with open(index_file, "wb") as f:
        pickle.dump(index_list, f)


if __name__ == "__main__":
    main()









