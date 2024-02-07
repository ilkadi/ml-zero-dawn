import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ml_zero_dawn.helpers.config_helper import ConfigHelper

# Load the configuration
config_helper = ConfigHelper()
config = config_helper.update_with_config('ml_zero_dawn/0_binary_classification/config.yaml')

# Load the dataset
df = pd.read_csv(f"{config['dataset_path']}/{config['dataset_name']}")
df = df.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
df_before = df
df = df.dropna()
print(f"Size of the dataset after dropping missing values: {df.shape}")
print(f"Number of rows dropped: {df_before.shape[0] - df.shape[0]}")

le = LabelEncoder()
scaler = StandardScaler()
df['Type'] = le.fit_transform(df['Type'])

columns_to_scale = ['Air temperature [K]', 
                    'Process temperature [K]', 
                    'Rotational speed [rpm]', 
                    'Torque [Nm]', 
                    'Tool wear [min]']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Split the data into training and testing sets
X = df.drop(['Target', 'Type'], axis=1)
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data
X_train.to_pickle(f"{config['preprocessed_path']}/X_train.pkl")
X_test.to_pickle(f"{config['preprocessed_path']}/X_test.pkl")
y_train.to_pickle(f"{config['preprocessed_path']}/y_train.pkl")
y_test.to_pickle(f"{config['preprocessed_path']}/y_test.pkl")