import sys
import os
import logging
import pandas as pd
import torch

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities and the SimpleNN model from the training module
from utils import configure_logging, get_project_dir
from training.train import SimpleNN  # Adjusted import path for SimpleNN

# Configure logging
configure_logging()

# Paths and constants
DATA_DIR = get_project_dir('data')
MODEL_DIR = get_project_dir('models')
RESULTS_DIR = get_project_dir('results')
INFERENCE_FILE = os.path.join(DATA_DIR, 'inference_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_model.pth')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'inference_results.csv')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_inference():
    """Run inference using the trained model."""
    logging.info("Loading the trained model...")
    # Define the model architecture (ensure it matches the one used in train.py)
    model = SimpleNN(input_size=4, output_size=3)  # Adjust input/output sizes if needed
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    logging.info("Loading inference data...")
    inference_data = pd.read_csv(INFERENCE_FILE)
    X_inference = inference_data.iloc[:, :-1].values  # Exclude the target column

    # Convert inference data to PyTorch tensor
    X_inference_tensor = torch.tensor(X_inference, dtype=torch.float32)

    logging.info("Running inference...")
    with torch.no_grad():
        predictions = model(X_inference_tensor)
        predicted_classes = torch.argmax(predictions, axis=1).numpy()

    logging.info("Saving inference results...")
    inference_data['predicted_class'] = predicted_classes
    inference_data.to_csv(RESULTS_PATH, index=False)
    logging.info(f"Inference results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_inference()
