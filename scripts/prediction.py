from sklearn import linear_model
import argparse
import numpy as np

args = argparse.ArgumentParser(description='Logistic model')
args.add_Argument('--max_iter', type=int, default=1000)
args.add_Argument('--input_file', type=int, help="directory to load X")
args.add_Argument('--label_file', type=int, help="directory to load Y")
args.add_Argument('--save_file', type=int, help="directory to save predicted Y")


lr_clf = linear_model.LogisticRegression(max_iter=args.max_iter)
x_train = np.load(f"{args.input_file}_train.npy")
x_test = np.load(f"{args.input_file}_test.npy")
y_train = np.load(f"{args.label_file}_train.npy")
y_test = np.load(f"{args.label_file}_test.npy")

lr_clf.fit(x_train, y_train)

y_proba = lr_clf.predict_proba(x_test)[:, 1]
y_pred = lr_clf.predict(x_test)

np.save(f"{save_file}_pred.npy", y_pred)
np.save(f"{save_file}_proba.npy", y_proba)