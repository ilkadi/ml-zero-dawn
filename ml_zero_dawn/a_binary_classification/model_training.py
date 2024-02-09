import torch
import datetime
import os
import pandas as pd
from torch import optim
from ml_zero_dawn.helpers.config_helper import ConfigHelper
from ml_zero_dawn.helpers.hardware_helper import HardwareHelper
from ml_zero_dawn.helpers.data_helper import DataHelper 
from ml_zero_dawn.helpers.lr_helper import LRHelper, LRViewer 
from ml_zero_dawn.helpers.eval_helper import EvalHelper 
from ml_zero_dawn.a_binary_classification.model import Model 

# Setup configuration and hardware
config_helper = ConfigHelper()
config = config_helper.update_with_config('ml_zero_dawn/a_binary_classification/config.yaml')

hw = HardwareHelper(config['data_type'], config['device_type'], config['device'])

# todo move to helper
# Load the preprocessed data
# Train
X_train_path = f"{config['preprocessed_path']}/X_train.pkl"
y_train_path = f"{config['preprocessed_path']}/y_train.pkl"
if not os.path.exists(X_train_path) or not os.path.exists(y_train_path):
    raise ValueError("Please generate the preprocessed data first with 'data_processing.py' script.")
X_train = pd.read_pickle(X_train_path)
y_train = pd.read_pickle(y_train_path)

X_train = torch.tensor(X_train.values, dtype=hw.dtype)
y_train = torch.tensor(y_train.values, dtype=hw.dtype)
train_batcher = DataHelper(hw, X_train, y_train, batch_size=config['batch_size'])

# todo move to helper
# Test
X_test_path = f"{config['preprocessed_path']}/X_test.pkl"
y_test_path = f"{config['preprocessed_path']}/y_test.pkl"
if not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
    raise ValueError("Please generate the preprocessed data first with 'data_processing.py' script.")
X_test = pd.read_pickle(X_test_path)
y_test = pd.read_pickle(y_test_path)

X_test = torch.tensor(X_test.values, dtype=hw.dtype)
y_test = torch.tensor(y_test.values, dtype=hw.dtype)
test_batcher = DataHelper(hw, X_test, y_test, batch_size=config['batch_size'])

if config['input_dim'] != X_train.shape[1]:
    raise ValueError("The input dimension of the model does not match the input data.")
model, criterion = Model(config['input_dim']).get_model()
model = model.to(hw.device)

epochs = config['epochs']
optimizer = optim.Adam(model.parameters(), lr=config['default_lr'])
lr_helper = LRHelper(config['default_lr'], config['warmup_iters'], config['decay_lr_iters'], config['min_lr'])
eval_helper = EvalHelper(hw, model, config['eval_iters'], test_batcher, criterion)
#LRViewer().view_lr_helper(lr_helper, num_epochs=epochs)

# Train the model
for epoch in range(epochs):
    lr_helper.set_cousine_annealing_lr(optimizer, epoch)
    X, Y = train_batcher.get_batch()
    
    with hw.context:
        outputs = model(X)
        Y = Y.view(Y.size(0), 1) 
        loss = criterion(outputs, Y)
    
    hw.scaler.scale(loss).backward()   
    hw.scaler.step(optimizer)
    hw.scaler.update()
    
    optimizer.zero_grad(set_to_none=True)
        
    if epoch % 200 == 0: 
        print(f"Epoch {epoch}, Train Loss: {loss.item()}")
    if epoch % config['eval_interval'] == 0:
        loss_mean, accuracy, precision, recall, auc_roc = eval_helper.estimate_loss()
        print(f"-- Epoch {epoch}, Validation Loss: {loss_mean}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC-ROC: {auc_roc} --")

loss_mean, accuracy, precision, recall, auc_roc = eval_helper.estimate_loss()
print("---------------------------------")
print("Training finished.")
print(f"Final Validation Loss: {loss_mean}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, AUC-ROC: {auc_roc}")

# Save the trained model
if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])
now = datetime.datetime.now()
model_name = now.strftime("%Y%m%d_%H%M%S" + ".pth")
torch.save(model.state_dict(), f"{config['model_path']}/{model_name}")