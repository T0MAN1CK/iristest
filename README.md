# Iris Dataset Project

This project trains a machine learning model on the Iris dataset and runs batch inference in a modularized and containerized setup.

## Project Structure

- `data_process/`: Data processing scripts.
- `training/`: Model training scripts.
- `inference/`: Inference scripts.
- `unittest/`: Unit tests.
- `utils.py`: Helper functions.
- `settings.json`: Configuration file.
- `docker-compose.yml`: Docker Compose file for managing services.
- `Dockerfile`: Dockerfiles for data preparation, training, and inference.

## Setup Instructions

### Prerequisites

1. **Docker**: Ensure Docker is installed on your system. You can download it from [here](https://www.docker.com/get-started).
2. **Docker Compose**: Ensure Docker Compose is installed. It usually comes bundled with Docker.

### Running the Project

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:T0MAN1CK/iristest.git
   cd iristest
2. **Start the Workflow**:
   ```bash
    docker-compose up --build
3. **Inspecting logs**:
   ```bash
    docker logs iristest-<name of container>
4. **Generated Files**: 
    The following files and folders will be created in the project directory:
    data/: Contains training and inference data (train_data.csv, inference_data.csv).
    models/: Contains the trained model (iris_model.pth).
    results/: Contains training scores (training_scores.txt) and inference results (inference_results.csv).
    

    