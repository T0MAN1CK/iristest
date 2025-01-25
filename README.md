# Iris Dataset Project

This project trains a machine learning model on the Iris dataset and runs batch inference in a modularized and containerized setup.

## Project Structure

- `data_process/`: Contains scripts for data preparation.
  - `data_retreival.py`: Downloads the Iris dataset, splits it into training and inference sets, and saves them to the `data/` directory.
- `training/`: Contains scripts for training the model.
  - `train.py`: Loads training data, trains the neural network model, evaluates it, and saves the model and training scores.
- `inference/`: Contains scripts for running inference.
  - `inference.py`: Loads the trained model, performs inference on new data, and saves the results to the `results/` directory.
- `unittest/`: Contains unit tests for verifying the functionality of the project's scripts and components.
- `utils.py`: Contains helper functions for logging and project path management.
- `settings.json`: Configuration file for managing project settings (e.g., paths, hyperparameters).
- `Dockerfile`: Dockerfiles for building containers for each module (data preparation, training, and inference).

## Setup Instructions

### Prerequisites

1. **Docker**: Ensure Docker is installed on your system. You can download it from [here](https://www.docker.com/get-started).

### Running the Project

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:T0MAN1CK/iristest.git
   cd iristest
   ```

2. **Build and Run Containers**:
   Run the following commands to execute each step in the workflow:
   - **Data Preparation**:
     ```bash
     docker build -t iristest-data_preparation -f data_process/Dockerfile .
     docker run --rm -v "$(pwd)/data:/app/data" --name iristest-data_preparation iristest-data_preparation
     ```
   - **Training**:
     ```bash
     docker build -t iristest-training -f training/Dockerfile .
     docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/results:/app/results" --name iristest-training iristest-training
     ```
   - **Inference**:
     ```bash
     docker build -t iristest-inference -f inference/Dockerfile .
     docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" -v "$(pwd)/results:/app/results" --name iristest-inference iristest-inference
     ```

3. **Inspect Logs**:
   To check the logs of a specific container after it runs:
   ```bash
   docker logs <container_name>
   ```

4. **Generated Files**:
   The following files and folders will be created in the project directory:
   - `data/`: Contains training and inference data (`train_data.csv`, `inference_data.csv`).
   - `models/`: Contains the trained model (`iris_model.pth`).
   - `results/`: Contains training scores (`training_scores.txt`) and inference results (`inference_results.csv`).

## Additional Details

### File and Directory Descriptions:
- **`data_process/`**: Handles the preparation of the Iris dataset, ensuring it's ready for training and inference.
- **`training/`**: Contains the logic for building, training, and saving the neural network model.
- **`inference/`**: Loads the trained model, runs predictions on the prepared inference dataset, and outputs the results.
- **`unittest/`**: Includes test cases to validate the correctness of individual scripts and modules.
- **`utils.py`**: Provides utility functions for logging and managing project paths consistently across scripts.
- **`settings.json`**: Stores configuration options, such as paths and hyperparameters.

### Notes:
- The `data/` folder is automatically created during data preparation. If the folder already exists, you may delete it before running the workflow.
- Ensure all paths and volume bindings are correct when running Docker commands to avoid file-saving issues.
