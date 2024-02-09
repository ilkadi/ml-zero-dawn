import torch
import joblib
import pickle
import argparse
from halo import Halo
import pandas as pd
from ml_zero_dawn.a_binary_classification.model import Model 
from ml_zero_dawn.helpers.config_helper import ConfigHelper
from ml_zero_dawn.helpers.hardware_helper import HardwareHelper

class ModelInference:
    def __init__(self, model_name):
        config_helper = ConfigHelper()
        self.config = config_helper.update_with_config('ml_zero_dawn/a_binary_classification/config.yaml')
        self.hw = HardwareHelper(self.config['data_type'], self.config['device_type'], self.config['device'])

        self.scaler = joblib.load(f"{self.config['preprocessed_path']}/scaler.pkl")
        self.model = Model(self.config['input_dim']).m
        self.model = self.model.to(self.hw.device)
        self.model.load_state_dict(torch.load(f"{self.config['model_path']}/{model_name}"))
        
        with open(f"{self.config['preprocessed_path']}/train_columns.pkl", 'rb') as f:
            self.train_columns = pickle.load(f)

    # todo rework common method dependencies across inference and training
    def prepare_single_input(self, input_data):
        df = pd.DataFrame([input_data])
        df = pd.get_dummies(df, columns=['Type'])

        # Ensure the one-hot encoded dataframe has the same columns as the training data
        # Add missing columns and fill with 0s
        for col in self.train_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder the columns to match the training data
        columns_to_scale = ['Air temperature [K]', 
                    'Process temperature [K]', 
                    'Rotational speed [rpm]', 
                    'Torque [Nm]', 
                    'Tool wear [min]']
        df = df[self.train_columns]
        df[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        return df

    def run_model(self):
        print("Starting the interactive console..")
        spinner = Halo(text='model thinks', spinner='dots')
        while True:
            type = input("type >: ")
            air_t = input("airT >: ")
            pro_t = input("proT >: ")
            rpm = input("rpm >: ")
            nm = input("Nm >: ")
            wear = input("Wear >: ")
            
            input_data = {'Type': type, 'Air temperature [K]': float(air_t), 'Process temperature [K]': float(pro_t), 'Rotational speed [rpm]': float(rpm), 'Torque [Nm]': float(nm), 'Tool wear [min]': float(wear)}
            prepared_input = self.prepare_single_input(input_data)

            spinner.start()
            with torch.no_grad():
                with self.hw.context:
                    x = torch.tensor(prepared_input.values, dtype=self.hw.dtype).to(self.hw.device)
                    y = self.model(x)
            spinner.stop()
            pb = torch.sigmoid(y).item()
            print(f"Prediction: {round(pb)} [{pb:.2f}]")
            print('---------------')
            cont = input("Do you want to continue? (yes/no): ")
            if cont.lower() != 'yes':
                break
    
def main(model_name):
    runner = ModelInference(model_name)
    runner.run_model()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Start the model run. 
    Use --model flag to load a specific model.""")
    parser.add_argument('--model', type=str, required=True, help='Name of the model stored at <config[model_path]> dir.')
    args = parser.parse_args()

    main(args.model)