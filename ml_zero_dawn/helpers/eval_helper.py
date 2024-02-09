import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class EvalHelper:
    def __init__(self, hardware_helper, model, eval_iters, val_batcher, criterion):
        self.hw = hardware_helper
        self.model = model
        self.eval_iters = eval_iters
        self.val_batcher = val_batcher
        self.criterion = criterion
    
    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        losses = torch.zeros(self.eval_iters)
        all_outputs = []
        all_targets = []
        for k in range(self.eval_iters):
            X, Y = self.val_batcher.get_batch()
            with self.hw.context:
                outputs = self.model(X)
                Y = Y.view(Y.size(0), 1)
                loss = self.criterion(outputs, Y)
            losses[k] = loss.item()
            
            sigmoid_outputs = torch.sigmoid(outputs)
            all_outputs.extend(sigmoid_outputs.view(-1).cpu().detach().numpy().round().astype(int))
            all_targets.extend(Y.view(-1).cpu().numpy().round().astype(int))

        # Print the shapes and data types of all_targets and all_outputs
        # print(f"Unique values in all_targets: {np.unique(all_targets)}")
        # print(f"Unique values in all_outputs: {np.unique(all_outputs)}")
        # print(f"all_targets shape: {np.shape(all_targets)}, dtype: {np.array(all_targets).dtype}")
        # print(f"all_outputs shape: {np.shape(all_outputs)}, dtype: {np.array(all_outputs).dtype}")
            
        loss_mean = losses.mean()
        
        accuracy = accuracy_score(all_targets, all_outputs)
        precision = precision_score(all_targets, all_outputs)
        recall = recall_score(all_targets, all_outputs)
        auc_roc = roc_auc_score(all_targets, all_outputs)
        
        self.model.train()
        return loss_mean, accuracy, precision, recall, auc_roc