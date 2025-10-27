import os
import xlrd
import torch
from torch import nn
import numpy as np
import random
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

class FSFT6mA(nn.Module):
    def __init__(self):
        super(FSFT6mA, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=80, kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=4)
        self.conv4 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=80, out_channels=80, kernel_size=4)

        self.act = nn.LeakyReLU(negative_slope=0.001, inplace=True)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.2)
        self.drop5 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(80 * 28, 100)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 1)

        # -----------------------------
        # 权重初始化 (He-normal)
        # -----------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # (batch, seq_len=41, channels=4) → (batch, channels=4, seq_len=41)
        x = x.permute(0, 2, 1)
        x = self.drop1(self.act(self.conv1(x)))
        x = self.drop2(self.act(self.conv2(x)))
        x = self.drop3(self.act(self.conv3(x)))
        x = self.drop4(self.act(self.conv4(x)))
        x = self.drop5(self.act(self.conv5(x)))

        x = x.view(x.size(0), -1)  # flatten

        x = self.drop_fc(self.act(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        return x

def read_seq_label(filename):
    workbook = xlrd.open_workbook(filename=filename)

    booksheet = workbook.sheet_by_index(0)
    nrows = booksheet.nrows

    seq = []
    label = []
    for i in range(nrows):
        seq.append(booksheet.row_values(i)[0])
        label.append(booksheet.row_values(i)[1])

    return seq, (torch.from_numpy(np.array(label))).to(torch.float32)

def seq_to01_to0123(filename):

    seq, label = read_seq_label(filename)

    nrows = len(seq)
    seq_len = len(seq[0])

    seq_01 = np.zeros((nrows, seq_len, 4), dtype='int')
    seq_0123 = np.zeros((nrows, seq_len), dtype='int')

    for i in range(nrows):
        one_seq = seq[i]
        one_seq = one_seq.replace('A', '0')
        one_seq = one_seq.replace('C', '1')
        one_seq = one_seq.replace('G', '2')
        one_seq = one_seq.replace('T', '3')
        seq_start = 0
        for j in range(seq_len):
            seq_0123[i, j] = int(one_seq[j - seq_start])
            if j < seq_start:
                seq_01[i, j, :] = 0.25
            else:
                try:
                    seq_01[i, j, int(one_seq[j - seq_start])] = 1
                except:
                    seq_01[i, j, :] = 0.25
    return seq, seq_01, label

def load_data(datapath, dataname):
    fullpath = os.path.join(datapath, dataname)
    fullpath_train = fullpath + '_train.xlsx'
    fullpath_test = fullpath + '_test.xlsx'

    seq_, seq01_, label_ = seq_to01_to0123(fullpath_train)
    r = random.random
    random.seed(2)
    num_total = len(label_)
    a = np.linspace(0, num_total - 1, num_total).astype(int)
    random.shuffle(a, random=r)
    num_train = int(num_total * 0.9)
    num_val = num_total - num_train
    train_index = a[:num_train]
    valid_index = a[num_train:num_total]

    seq_train = np.array(seq_)[train_index]
    seq_val = np.array(seq_)[valid_index]

    x_train = seq01_[train_index, :, :]
    x_val = seq01_[valid_index, :, :]

    y_train = label_[train_index]
    y_val = label_[valid_index]

    seq_test, x_test, y_test = seq_to01_to0123(fullpath_test)

    return seq_train, seq_val, seq_test, x_train, y_train, x_val, y_val, x_test, y_test

def score2label(data,th):
    out =[]
    for indexResults in range(len(data)):
        if float(data[indexResults]) > th or float(
                data[indexResults]) == th:
            out.append(1)
        else:
            out.append(0)
    return out

def acc_f1_mcc_auc_pre_rec(preds, labels, probs):
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds)

    acc = accuracy_score(preds, labels)
    sen = 1.0 * TP / (TP + FN)
    spe = 1.0 * TN / (FP + TN)
    pre = 1.0 * TP / (FP + TP)

    return {
        "auc": auc,
        "ap": ap,
        "mcc": mcc,
        "f1": f1,
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "pre": pre,

    }

def evaluate(model, data):
    model.eval()
    pred_all = np.empty(0)
    y_all = np.empty(0)
    th = 0.5

    for x_batch, y_batch in data:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        with torch.no_grad():
            output_batch = model(x_batch)
            loss = F.binary_cross_entropy(output_batch[:,0], y_batch)

            y_all = np.concatenate((y_all, y_batch.detach().cpu().numpy()), axis=0)
            pred_all = np.concatenate((pred_all, output_batch[:,0].detach().cpu().numpy()), axis=0)

    predict_class = score2label(pred_all, th)
    results = acc_f1_mcc_auc_pre_rec(predict_class, y_all, pred_all)
    return results

def predict(model, data):
    model.eval()
    pred_all = np.empty(0)
    y_all = np.empty(0)
    with torch.no_grad():
        for x_batch, y_batch in data:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                output_batch = model(x_batch)
                loss = F.binary_cross_entropy(output_batch[:,0], y_batch)

                y_all = np.concatenate((y_all, y_batch.detach().cpu().numpy()), axis=0)
                pred_all = np.concatenate((pred_all, output_batch[:,0].detach().cpu().numpy()), axis=0)

        predict_class = score2label(pred_all, 0.5)
        result = acc_f1_mcc_auc_pre_rec(predict_class, y_all, pred_all)

        output_eval_file = dataname + ".txt"
        with open(output_eval_file, "a") as writer:
            for key in sorted(result.keys()):
                result_rt = key + " " + str(result[key])[:6]
                writer.write(result_rt + "\n")

        df = DataFrame({'predict': pred_all.flatten(), 'predicty': np.array(predict_class).flatten(),
                        'y_text': y_all.flatten()},
                       index=range(len(y_all.flatten().tolist())))
        df.to_csv(dataname + "_predict_results.csv", index=False)

def train(model, train_iter, valid_iter,epochs,lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_auc = 0

    for epoch in range(epochs):
        epoch_loss_train = 0.0
        model.train()
        for x_batch, y_batch in train_iter:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output_batch = model(x_batch)
            loss = F.binary_cross_entropy(output_batch[:,0], y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()
        # print(epoch_loss_train)

        results = evaluate(model, valid_iter)# 验证

        val_auc = results["auc"]
        print(epoch)
        print("val_acc", format(results["acc"], '.4f'))
        print("val_auc", format(results["auc"], '.4f'))
        print("val_ap", format(results["ap"], '.4f'))
        print("val_mcc", format(results["mcc"], '.4f'))
        print("val_f1", format(results["f1"], '.4f'))
        print("val_sen", format(results["sen"], '.4f'))
        print("val_spe", format(results["spe"], '.4f'))
        print("val_pre", format(results["pre"], '.4f'))

        if val_auc > best_auc:
            torch.save(model.state_dict(), dataname)
            best_auc = val_auc
            print(f"best_auc: {best_auc}")


    return 0


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datapath = './Data/'

    dataname = 'A.thaliana'
    # dataname = 'D.melanogaster'

    [seq_train, seq_val, seq_test, X_train, y_train, X_valid, y_valid, X_test, y_test] = load_data(datapath, dataname)

    X_train = (torch.from_numpy(X_train)).to(torch.float32)
    X_valid = (torch.from_numpy(X_valid)).to(torch.float32)
    X_test = (torch.from_numpy(X_test)).to(torch.float32)

    train_dataset = Data.TensorDataset(X_train, y_train)
    valid_dataset = Data.TensorDataset(X_valid, y_valid)
    test_dataset = Data.TensorDataset(X_test, y_test)

    train_iter = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_iter = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = FSFT6mA()
    model.to(device)

    train(model, train_iter, valid_iter, epochs=1, lr=0.001)
    # 加载模型保存结果
    model.load_state_dict(torch.load(dataname))
    predict(model, test_iter)
