import torch
from torch import nn

# Define the same model structure
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Load the trained model
model.load_state_dict(torch.load('ml-zero-dawn/ml_zero_dawn/0_binary_classification/model.pth'))

# Use the model to make predictions on new data
def predict(new_data):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(new_data, dtype=torch.float32)
        outputs = model(inputs)
        return outputs