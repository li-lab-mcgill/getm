from sklearn import linear_model
import argparse
import numpy as np

args = argparse.ArgumentParser(description='Logistic model')
args.add_argument('--max_iter', type=int, default=1000)
args.add_argument('--input_file', type=str, help="directory to load X")
args.add_argument('--label_file', type=str, help="directory to load Y")
args.add_argument('--save_file', type=str, help="directory to save predicted Y")

args = args.parse_args()
lr_clf = linear_model.LogisticRegression(max_iter=args.max_iter)
x_train = np.load(f"{args.input_file}_train.npy")
x_test = np.load(f"{args.input_file}_test.npy")
y_train = np.load(f"{args.label_file}_train.npy")
y_test = np.load(f"{args.label_file}_test.npy")

lr_clf.fit(x_train, y_train)
omega = lr_clf._coef

y_proba = lr_clf.predict_proba(x_test)[:, 1]
y_pred = lr_clf.predict(x_test)

np.save(f"{args.save_file}_pred.npy", y_pred)
np.save(f"{args.save_file}_proba.npy", y_proba)
np.save(f"{args.save_file}_omega.npy", omega)
