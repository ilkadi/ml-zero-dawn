import torch
from pandas import pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Load the preprocessed data
X_train = pd.read_csv('ml-zero-dawn/ml_zero_dawn/0_binary_classification/X_train.csv')
y_train = pd.read_csv('ml-zero-dawn/ml_zero_dawn/0_binary_classification/y_train.csv')

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)

# Create a DataLoader for the training data
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

# Define the model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(100):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'ml-zero-dawn/ml_zero_dawn/0_binary_classification/model.pth')