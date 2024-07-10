# Engineering-and-Evaluating_AI_CA
=======

# Chained Multi-outputs for Multi-label Classification

## Overview

This project implements a chained multi-output approach for multi-label classification of emails. The dataset consists of various interaction types that need to be classified into multiple dependent variables (Type 2, Type 3, Type 4). This project aims to demonstrate the effectiveness of chained multi-output models in sequential classification tasks.

## Directory Structure

```
CA_Code_with_Chained_Multi_Outputs/
├── .idea/                          # IDE configuration files
├── __pycache__/                    # Compiled bytecode files
├── data/                           # Data files used for training and evaluation
├── model/                          # Model definitions and implementations
├── modelling/                      # Data modelling and processing scripts
├── Config.py                       # Configuration file
├── embeddings.py                   # Embedding generation script
├── gradient_boosting.py            # Gradient boosting model script
├── main.py                         # Main script for running the pipeline
├── preprocess.py                   # Data preprocessing script
└── README.md                       # Project documentation
```

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/CA_Code_with_Chained_Multi_Outputs.git
    cd CA_Code_with_Chained_Multi_Outputs
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Ensure the data files are in place**: 
   Place the `AppGallery.csv` and `Purchasing.csv` files in the `data/` directory.

## Usage

1. **Run the main script**:
    ```sh
    python main.py
    ```

2. **Check the output**:
   The script will output the classification results and accuracy metrics.

## Project Details

### Chained Multi-outputs Approach

The chained multi-outputs approach involves sequential classification, where the prediction of one type influences the subsequent classifications.

1. **Stage 1**: A model (e.g., Random Forest) classifies Type 2.
2. **Stage 2**: The same model classifies combined Type 2 and Type 3.
3. **Stage 3**: The model classifies combined Type 2, Type 3, and Type 4.

### Data Elements

- **Input Data**: 
  - `AppGallery.csv`
  - `Purchasing.csv`
- **Types and Classes**:
  - **Type 2**: Suggestion, Problem/Fault, Others
  - **Type 3**: Various payment and usage-related issues
  - **Type 4**: Various app-related issues and requests

### Preprocessing

Data preprocessing includes de-duplication and noise removal to ensure clean and consistent input data.

### Model Training and Evaluation

The models are trained in a chained sequence, with each subsequent model using the predictions from the previous stage.

## Results

### AppGallery & Games

- **Accuracy for Type 2**: 0.76
- **Accuracy for Type 3**: 1.0
- **Accuracy for Type 4**: 1.0

### In-App Purchase

- **Accuracy for Type 2**: 0.82
- **Accuracy for Type 3**: 1.0
- **Accuracy for Type 4**: 1.0

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for review.

## License

This project is licensed under the MIT License.

