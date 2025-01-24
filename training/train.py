import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import sys
from sklearn.metrics import accuracy_score
import time

# Add the project root directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from utils import configure_logging, get_project_dir

# Configure logging
configure_logging()

# Define directories and paths
DATA_DIR = get_project_dir('data')
MODEL_DIR = get_project_dir('models')
RESULTS_DIR = get_project_dir('results')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_data.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'iris_model.pth')
SCORES_PATH = os.path.join(RESULTS_DIR, 'training_scores.txt')

# Ensure the directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model():
    """Train a simple neural network and calculate metrics."""
    logging.info("Loading training data...")
    data = pd.read_csv(TRAIN_FILE)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].factorize()[0]  # Convert class labels to numerical indices

    # Get input and output sizes
    input_size = X.shape[1]
    output_size = len(set(y))

    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize the model, loss function, and optimizer
    model = SimpleNN(input_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logging.info("Training the Neural Network...")
    num_epochs = 10
    best_loss = float('inf')
    patience = 3
    trigger_times = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logging.info("Early stopping triggered")
                break

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

    # Evaluate the model
    logging.info("Evaluating the model on the test dataset...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, axis=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    logging.info(f"Test Accuracy: {accuracy:.4f}")

    # Save the model and accuracy
    logging.info("Saving the trained model...")
    torch.save(model.state_dict(), MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    logging.info("Saving training scores...")
    with open(SCORES_PATH, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
    logging.info(f"Training scores saved to {SCORES_PATH}")

if __name__ == "__main__":
    train_model()