from torch import nn

class Model:
    def __init__(self, input_dim):
        # 5000 iterations Validation Loss: 0.05425126105546951, Accuracy: 0.9815, Precision: 0.74, Recall: 0.6065573770491803, AUC-ROC: 0.7999264451001447
        self.m = nn.Sequential(
            nn.Linear(input_dim, 150), 
            nn.GELU(),
            nn.Linear(150, 25),
            nn.GELU(),
            nn.Linear(25, 25),
            nn.GELU(),
            nn.Linear(25, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
    def get_model(self):
        return self.m, self.criterion
    