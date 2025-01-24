import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import configure_logging, get_project_dir

# Configure logging
configure_logging()

# Paths and constants
DATA_DIR = get_project_dir('data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train_data.csv')
INFERENCE_PATH = os.path.join(DATA_DIR, 'inference_data.csv')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_split_data():
    """Download and split the Iris dataset."""
    logging.info("Downloading the Iris dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    iris_data = pd.read_csv(url, header=None, names=columns)

    logging.info("Splitting the dataset into train and inference sets...")
    train_data, inference_data = train_test_split(iris_data, test_size=0.2, random_state=42)

    train_data.to_csv(TRAIN_PATH, index=False)
    inference_data.to_csv(INFERENCE_PATH, index=False)

    logging.info(f"Training data saved to {TRAIN_PATH}")
    logging.info(f"Inference data saved to {INFERENCE_PATH}")

if __name__ == "__main__":
    download_and_split_data()
