import pandas as pd
import numpy as np
import joblib
import pickle
from scipy.stats import binom
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_zero_dawn.helpers.config_helper import ConfigHelper
from ml_zero_dawn.helpers.visualisation_helper import VisualisationHelper

# Load the configuration
config_helper = ConfigHelper()
config = config_helper.update_with_config('ml_zero_dawn/a_binary_classification/config.yaml')

# Load the dataset
df = pd.read_csv(f"{config['dataset_path']}/{config['dataset_name']}")
df = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
df_before = df
df = df.dropna()
print(f"Size of the dataset after dropping missing values: {df.shape}")
print(f"Number of rows dropped: {df_before.shape[0] - df.shape[0]}")

df = pd.get_dummies(df, columns=['Type'])

columns_to_scale = ['Air temperature [K]', 
                    'Process temperature [K]', 
                    'Rotational speed [rpm]', 
                    'Torque [Nm]', 
                    'Tool wear [min]']
if config.get('print_stats', False):
    VisualisationHelper.print_stats(df, columns_to_scale)
    
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split the data into training and testing sets
X = df.drop(['Target'], axis=1)
y = df['Target']

# Analyze the dataset to recommend batch-size of good heterogenous probability
p_ones = np.mean(y)
p_zeros = 1 - p_ones
threshold = 0.05  # threshold for the probability of a homogeneous batch

for batch_size in range(1, len(y)):
    p_homogeneous = binom.pmf(0, batch_size, p_ones) + binom.pmf(batch_size, batch_size, p_ones)
    if p_homogeneous < threshold:
        break
print(f"Recommended min batch size: {batch_size} with a threshold of {threshold} for homogeneous batch probability.")
# min batch size 87 for this dataset

# Save the preprocessed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_pickle(f"{config['preprocessed_path']}/X_train.pkl")
X_test.to_pickle(f"{config['preprocessed_path']}/X_test.pkl")
y_train.to_pickle(f"{config['preprocessed_path']}/y_train.pkl")
y_test.to_pickle(f"{config['preprocessed_path']}/y_test.pkl")

joblib.dump(scaler, f"{config['preprocessed_path']}/scaler.pkl")
train_columns = X.columns.tolist()
with open(f"{config['preprocessed_path']}/train_columns.pkl", 'wb') as f:
    pickle.dump(train_columns, f)