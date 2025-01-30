import unittest
import pandas as pd
import os
import torch
from utils import get_project_dir
from training.train import train_model, SimpleNN
from inference.inference import run_inference
from data_process.data_retreival import download_and_split_data

class TestDataProcessing(unittest.TestCase):
    def test_data_processing(self):
        """Test if data processing creates train and inference files."""
        data_dir = get_project_dir('data')
        train_path = os.path.join(data_dir, 'train_data.csv')
        inference_path = os.path.join(data_dir, 'inference_data.csv')

        download_and_split_data()
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(inference_path))

        # Verify file contents
        train_data = pd.read_csv(train_path)
        inference_data = pd.read_csv(inference_path)
        self.assertFalse(train_data.empty)
        self.assertFalse(inference_data.empty)

class TestTraining(unittest.TestCase):
    def test_model_training(self):
        """Test if the model is trained and saved."""
        model_dir = get_project_dir('models')
        model_path = os.path.join(model_dir, 'iris_model.pth')

        train_model()
        self.assertTrue(os.path.exists(model_path))

        # Verify model is not corrupted
        model = SimpleNN(input_size=4, output_size=3)
        try:
            model.load_state_dict(torch.load(model_path))
        except Exception as e:
            self.fail(f"Failed to load model: {e}")

class TestInference(unittest.TestCase):
    def test_inference(self):
        """Test if inference creates the results file."""
        results_dir = get_project_dir('results')
        results_path = os.path.join(results_dir, 'inference_results.csv')

        run_inference()
        self.assertTrue(os.path.exists(results_path))

        # Verify results file has predictions
        results = pd.read_csv(results_path)
        self.assertIn('predicted_class', results.columns)

if __name__ == '__main__':
    unittest.main()