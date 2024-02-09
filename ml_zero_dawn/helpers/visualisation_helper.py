import pandas as pd
import matplotlib.pyplot as plt

class VisualisationHelper:
    @staticmethod
    def print_stats(df: pd.DataFrame, columns: list):
        for column in columns:
            if column in df.columns:
                print(f"\nStats for {column}:")
                print(f"Min: {df[column].min()}")
                print(f"Max: {df[column].max()}")
                plt.figure(figsize=(10, 5))
                plt.hist(df[column], bins=30)
                plt.title(f"Distribution of {column}")
                plt.show()