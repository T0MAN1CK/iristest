# Iris Dataset Project

This project trains a machine learning model on the Iris dataset and runs batch inference in a modularized and containerized setup.

## Project Structure
- `data_process/`: Data processing scripts.
- `training/`: Model training scripts.
- `inference/`: Inference scripts.
- `unittest/`: Unit tests.
- `utils.py`: Helper functions.
- `settings.json`: Configuration file.
- `Dockerfile`: Dockerfiles for training and inference.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
    python data_process/data_generation.py
    python training/train.py
    python inference/run.py
a
