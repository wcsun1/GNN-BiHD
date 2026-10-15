import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from args import args
from models import GNNBiHD
from utils import auprc, auroc, load_data, prediction, setup_seed


class Train:
    def __init__(self, args, repeat, acc_list, rec_list, prec_list, f1_list, auc_list, aupr_list, ftr_list, tpr_list, r_list, p_list):
        self.args = args
        self.repeat = repeat
        self.load_data()

        self.acc_list = acc_list
        self.rec_list = rec_list
        self.prec_list = prec_list
        self.f1_list = f1_list
        self.auc_list = auc_list
        self.aupr_list = aupr_list
        self.ftr_list = ftr_list
        self.tpr_list = tpr_list
        self.r_list = r_list
        self.p_list = p_list

        self.model = GNNBiHD(
            in_dim=self.data["features"].size(1),
            dim=args.emb_dim,
            heads=args.num_heads,
            blocks=args.K,
            tau=args.tau,
        ).to(args.device)
        self.criterion = torch.nn.BCELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def load_data(self):
        setup_seed(args.seed, torch.cuda.is_available())
        self.data = load_data(args.dataset, self.repeat, args.device, seed=args.seed)
        print('Data load init finish')
        print('lr={:.4g} weight_decay={:.4g} train_edges={} test_edges={}'.format(
            args.lr, args.weight_decay, int(self.data["train_y"].numel()), int(self.data["test_y"].numel())
        ))

    def _forward(self):
        return self.model.encode(self.data["features"], self.data["adj"], self.data["edge_index"])

    def _loss(self, embedding, local_pre, local_out, global_pre, global_out, source, target, labels):
        probability = self.model.score_pairs(embedding, source, target).clamp(1e-7, 1.0 - 1e-7)
        classification = self.criterion(probability, labels)
        local_contrast = self.model.contrastive(local_out, local_pre)
        global_contrast = self.model.contrastive(global_out, global_pre)
        loss = classification + self.args.alpha * local_contrast + self.args.beta * global_contrast
        return probability, loss, classification, local_contrast, global_contrast

    def train_student(self, epoch):
        started = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        embedding, local_pre, local_out, global_pre, global_out = self._forward()
        probability, loss, classification, local_contrast, global_contrast = self._loss(
            embedding, local_pre, local_out, global_pre, global_out,
            self.data["train_src"], self.data["train_dst"], self.data["train_y"],
        )
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            train_auc, _, _ = auroc(probability, self.data["train_y"])
            prediction_label = (probability >= 0.5).float()
            train_acc = (prediction_label == self.data["train_y"]).float().mean()
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss.item()),
              'ce: {:.4f}'.format(classification.item()),
              'con_lg: {:.4f}'.format(local_contrast.item()),
              'con_gl: {:.4f}'.format(global_contrast.item()),
              'acc_train: {:.4f}'.format(train_acc.item()),
              'auc_train: {:.4f}'.format(train_auc.item()),
              'time: {:.4f}s'.format(time.time() - started))

        self.model.eval()
        with torch.no_grad():
            embedding, _, _, _, _ = self._forward()
            test_probability = self.model.score_pairs(
                embedding, self.data["test_src"], self.data["test_dst"]
            ).clamp(1e-7, 1.0 - 1e-7)
            test_auc, _, _ = auroc(test_probability, self.data["test_y"])
        print('ts Test set results: auc= {:.4f}'.format(test_auc.item()))

    def test(self, ts='test'):
        self.model.eval()
        with torch.no_grad():
            embedding, _, _, _, _ = self._forward()
            probability = self.model.score_pairs(
                embedding, self.data["test_src"], self.data["test_dst"]
            ).clamp(1e-7, 1.0 - 1e-7)
            loss_test = self.criterion(probability, self.data["test_y"])
            auc_test, fpr, tpr = auroc(probability, self.data["test_y"])
            aupr, precision_curve, recall_curve = auprc(probability, self.data["test_y"])

        predlabel = (probability.detach().cpu().numpy() >= 0.5).astype(np.int32)
        labels = self.data["test_y"].int().detach().cpu().numpy()
        acc, precision, recall, f1 = prediction(predlabel, labels)
        save_dir = self.args.save_dir
        pd.DataFrame(embedding.cpu().detach().numpy()).to_csv('{}/emb_{}.csv'.format(save_dir, self.repeat), index=0, header=0)
        pd.DataFrame(self.data["test_src"].cpu().numpy()).to_csv('{}/index1_{}.csv'.format(save_dir, self.repeat), index=0, header=0)
        pd.DataFrame(self.data["test_dst"].cpu().numpy()).to_csv('{}/index2_{}.csv'.format(save_dir, self.repeat), index=0, header=0)
        pd.DataFrame(labels).to_csv('{}/labels_{}.csv'.format(save_dir, self.repeat), index=0, header=0)
        pd.DataFrame(predlabel).to_csv('{}/pred_labels_{}.csv'.format(save_dir, self.repeat), index=0, header=0)
        pd.DataFrame(probability.detach().cpu().numpy()).to_csv('{}/pred_prob_{}.csv'.format(save_dir, self.repeat), index=0, header=0)

        print("{ts} Test set results:".format(ts=ts),
              "loss= {:.4f}".format(loss_test.item()),
              "auc= {:.4f}".format(auc_test.item()))
        self.acc_list.append(round(float(acc), 4))
        self.rec_list.append(round(float(recall), 4))
        self.prec_list.append(round(float(precision), 4))
        self.f1_list.append(round(float(f1), 4))
        self.auc_list.append(round(float(auc_test), 4))
        self.aupr_list.append(round(float(aupr), 4))
        self.ftr_list.append(fpr)
        self.tpr_list.append(tpr)
        self.r_list.append(recall_curve)
        self.p_list.append(precision_curve)
