import torch
import numpy as np

from evaluation.metrics import regression_metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Evaluator:

    def __init__(self, model, device, task="regression"):

        self.model = model
        self.device = device
        self.task = task


    def predict(self, loader):

        self.model.eval()

        preds = []
        targets = []

        with torch.no_grad():

            for fc, sc, y in loader:

                fc = fc.to(self.device)
                sc = sc.to(self.device)
                y = y.to(self.device)

                out, _, _, _, _ = self.model(fc, sc)

                preds.append(out.detach().cpu())
                targets.append(y.detach().cpu())

        preds = torch.cat(preds)
        targets = torch.cat(targets)

        return preds.numpy(), targets.numpy()


    def evaluate_regression(self, loader):

        preds, targets = self.predict(loader)

        rmse, mae = regression_metrics(targets, preds)

        return {
            "RMSE": rmse,
            "MAE": mae
        }


    def evaluate_classification(self, loader):

        preds, targets = self.predict(loader)

        probs = preds.squeeze()

        pred_label = (probs > 0.5).astype(int)

        acc = accuracy_score(targets, pred_label)

        f1 = f1_score(targets, pred_label)

        try:
            auc = roc_auc_score(targets, probs)
        except:
            auc = 0.0

        return {
            "Accuracy": acc,
            "F1": f1,
            "AUC": auc
        }


    def evaluate(self, loader):

        if self.task == "regression":

            return self.evaluate_regression(loader)

        else:

            return self.evaluate_classification(loader)