import numpy as np
import pandas as pd

def extract_best_metrics(csv_file_path):
    """Extract best metrics from csv.

    Args:
        csv_file_path (str): Path to the csv file.

    Returns:
        str: A string representation of best metrics.
    """
    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_file_path)
    
    # Compute the fitness value based on the given weights
    fitness_values = 0.1 * data['       metrics/mAP50(B)'] + 0.9 * data['    metrics/mAP50-95(B)']

    # Get the index of the best fitness value
    best_index = np.argmax(fitness_values)

    # Get the row with the best metrics
    best_metrics = data.loc[best_index]

    result_str = f"best epoch: {best_metrics['                  epoch']}, mAP_0.5: {best_metrics['       metrics/mAP50(B)']}, mAP_0.5:0.95: {best_metrics['    metrics/mAP50-95(B)']}"
    
    return result_str


def main():
    # Model versions
    models = ['n', 's', 'm', 'l', 'x']

    for model in models:
        # Form the CSV path for each model
        csv_path = f'/home/t1/repos/pothole-detection/YOLOv8/train_comparison_{model}/results.csv'
        
        best_metrics = extract_best_metrics(csv_path)
        print(f"model: YOLOv8{model}, {best_metrics}")


if __name__ == "__main__":
    main()

