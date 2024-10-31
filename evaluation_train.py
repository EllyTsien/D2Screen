import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import paddle as pdl
import paddle.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class evaluation_train:
    def __init__(self, model, train_loader, valid_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        label_true = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
        label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
        
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in data_loader:
            label_predict_batch = self.model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            label_predict_batch = F.softmax(label_predict_batch)
            
            label_true = pdl.concat((label_true, label_true_batch.detach()), axis=0)
            label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
        
        y_pred = label_predict[:, 1].cpu().numpy()
        y_true = label_true.cpu().numpy()
        
        ap = round(average_precision_score(y_true, y_pred), 4)
        auc = round(roc_auc_score(y_true, y_pred), 4)
        
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        accuracy = round(accuracy_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        recall = round(recall_score(y_true, y_pred), 4)
        f1 = round(f1_score(y_true, y_pred), 4)
        confusion_mat = confusion_matrix(y_true, y_pred)
        
        metric = {
            'ap': ap, 
            'auc': auc, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'confusion_mat': confusion_mat
        }
        
        return metric

    def plot(self, train, valid, metric):
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, color="blue", label=f'Training {metric}')
        plt.plot(epochs, valid, color="orange", label=f'Validation {metric}')
        plt.title(f'Training and validation {metric}')
        plt.xlabel('Epochs')
        plt.ylabel(f'{metric}')
        plt.legend()
        plt.show()

    def plot_metrics(self, metric_train_list, metric_valid_list):
        metric_train = pd.DataFrame(metric_train_list)
        metric_valid = pd.DataFrame(metric_valid_list)

        self.plot(metric_train['accuracy'], metric_valid['accuracy'], metric='accuracy')
        self.plot(metric_train['ap'], metric_valid['ap'], metric='ap')
        self.plot(metric_train['auc'], metric_valid['auc'], metric='auc')
        self.plot(metric_train['f1'], metric_valid['f1'], metric='f1')
        self.plot(metric_train['precision'], metric_valid['precision'], metric='precision')
        self.plot(metric_train['recall'], metric_valid['recall'], metric='recall')

# 使用示例
# model = ...  # 你的模型
# train_loader = ...  # 训练数据加载器
# valid_loader = ...  # 验证数据加载器

# evaluator = evaluation_train(model, train_loader, valid_loader)
# metrics_train = evaluator.evaluate(train_loader)
# metrics_valid = evaluator.evaluate(valid_loader)
# evaluator.plot_metrics([metrics_train], [metrics_valid])
