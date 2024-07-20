# Engineering and Evaluating AI - CA Project

## Overview

This project focuses on the engineering and evaluation of an AI model that classifies various types of support tickets. The model utilizes text embeddings and classification algorithms to predict the type of issue described in a support ticket.


## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/x23233982/Engineering-and-Evaluating_AI_CA.git
    cd Engineering-and-Evaluating_AI_CA
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main script**:
    ```bash
    python main.py
    ```

2. **Generate Visualizations**:
    - The visualization will be generated and saved in the `images/` directory.

## Model Performance

### Overall Accuracy for Type 2: 73.81%
**Classification Report for Type 2:**

| Class             | Precision | Recall | F1-Score | Support |
|-------------------|-----------|--------|----------|---------|
| Others            | 0.60      | 0.43   | 0.50     | 7       |
| Problem/Fault     | 0.64      | 0.64   | 0.64     | 11      |
| Suggestion        | 0.81      | 0.88   | 0.84     | 24      |
| **Accuracy**      |           |        | 0.74     | 42      |
| **Macro Avg**     | 0.68      | 0.65   | 0.66     | 42      |
| **Weighted Avg**  | 0.73      | 0.74   | 0.73     | 42      |

### Overall Accuracy for Type 3: 69.05%
**Classification Report for Type 3:**

| Class                              | Precision | Recall | F1-Score | Support |
|------------------------------------|-----------|--------|----------|---------|
| AppGallery-Install/Upgrade         | 1.00      | 0.50   | 0.67     | 2       |
| AppGallery-Use                     | 1.00      | 0.33   | 0.50     | 3       |
| Coupon/Gifts/Points Issues         | 0.43      | 0.75   | 0.55     | 4       |
| General                            | 1.00      | 1.00   | 1.00     | 2       |
| Invoice                            | 1.00      | 0.50   | 0.67     | 2       |
| Missing                            | 0.57      | 0.57   | 0.57     | 7       |
| Payment                            | 0.75      | 0.88   | 0.81     | 17      |
| Payment issue                      | 0.00      | 0.00   | 0.00     | 0       |
| Third Party APPs                   | 0.00      | 0.00   | 0.00     | 2       |
| VIP / Offers / Promotions          | 1.00      | 0.67   | 0.80     | 3       |
| **Accuracy**                       |           |        | 0.69     | 42      |
| **Macro Avg**                      | 0.68      | 0.52   | 0.56     | 42      |
| **Weighted Avg**                   | 0.73      | 0.69   | 0.68     | 42      |

### Overall Accuracy for Type 4: 61.90%
**Classification Report for Type 4:**

| Class                              | Precision | Recall | F1-Score | Support |
|------------------------------------|-----------|--------|----------|---------|
| Can't install Apps                 | 1.00      | 0.50   | 0.67     | 2       |
| Can't use or acquire               | 0.75      | 1.00   | 0.86     | 3       |
| Cooperated campaign issue          | 0.00      | 0.00   | 0.00     | 1       |
| Invoice related request            | 1.00      | 0.50   | 0.67     | 2       |
| Missing                            | 0.45      | 0.71   | 0.56     | 7       |
| Offers / Vouchers / Promotions     | 1.00      | 0.67   | 0.80     | 3       |
| Others                             | 0.00      | 0.00   | 0.00     | 1       |
| Personal data                      | 1.00      | 1.00   | 1.00     | 1       |
| Query deduction details            | 0.00      | 0.00   | 0.00     | 2       |
| Refund                             | 0.00      | 0.00   | 0.00     | 2       |
| Security issue / malware           | 0.00      | 0.00   | 0.00     | 1       |
| Subscription cancellation          | 0.65      | 0.87   | 0.74     | 15      |
| UI Abnormal in Huawei AppGallery   | 0.00      | 0.00   | 0.00     | 2       |
| **Accuracy**                       |           |        | 0.62     | 42      |
| **Macro Avg**                      | 0.45      | 0.40   | 0.41     | 42      |
| **Weighted Avg**                   | 0.55      | 0.62   | 0.56     | 42      |

### Group Accuracies
- **Average Accuracy for AppGallery & Games group**: 40.00%
- **Average Accuracy for In-App Purchase group**: 86.27%
- **Overall Average Accuracy for all groups**: 65.08%

## Visualization

The sunburst chart visualizing the classification results can be found in the `images/` directory.

![image](https://github.com/user-attachments/assets/cdfa7ff1-9114-4bb0-b28c-3bcfee4bfef0)


## Project Structure
Engineering-and-Evaluating_AI_CA/
├── config/
│ ├── Config.py
├── features/
│ ├── embeddings.py
│ ├── input_data.py
│ ├── preprocess.py
├── model/
│ ├── base.py
│ ├── catboost.py
│ ├── evaluate.py
│ ├── gradient_boosting.py
│ ├── randomforest.py
│ ├── utils.py
│ ├── init.py
├── modelling/
│ ├── data_model.py
│ ├── modelling.py
├── tests/
│ ├── test.ipynb
│ ├── test_base.py
├── visualize/
│ ├── visualization.py
├── .gitignore
├── LICENSE
├── main.py
├── requirements.txt
├── true_and_predicted_results.csv

## Contributing

Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
