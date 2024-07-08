import pandas as pd
from sklearn.model_selection import train_test_split
from hierarchical_model.preprocess_hierarchical import load_and_preprocess_data
from hierarchical_model.model_hierarchical import train_and_predict_type2, train_and_predict_type3, train_and_predict_type4

# Load and preprocess the data
file_path = '../data/AppGallery.csv'
df = load_and_preprocess_data(file_path)

# Splitting the dataset
X = df[['Feature1', 'Feature2']]
y_type2 = df['Type2']

X_train, X_test, y_train, y_test = train_test_split(X, y_type2, test_size=0.2, random_state=42)

# Train and predict Type2
y_pred_type2, model_type2 = train_and_predict_type2(X_train, y_train, X_test)
accuracy_type2 = accuracy_score(y_test, y_pred_type2)
print(f'Type2 Accuracy: {accuracy_type2 * 100:.2f}%')

# Filter data for Type3 based on Type2 predictions
filtered_data = df[df['Type2'] == y_test.iloc[0]]  # Filtering based on a single class for simplicity
X_filtered = filtered_data[['Feature1', 'Feature2']]
y_filtered_type3 = filtered_data['Type3']

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filtered, y_filtered_type3, test_size=0.2, random_state=42)

# Train and predict Type3
y_pred_type3, model_type3 = train_and_predict_type3(X_train_f, y_train_f, X_test_f)
accuracy_type3 = accuracy_score(y_test_f, y_pred_type3)
print(f'Type3 Accuracy: {accuracy_type3 * 100:.2f}%')

# Filter data for Type4 based on Type3 predictions
filtered_data_type4 = df[df['Type3'] == y_pred_type3[0]]  # Filtering based on a single class for simplicity
X_filtered_type4 = filtered_data_type4[['Feature1', 'Feature2']]
y_filtered_type4 = filtered_data_type4['Type4']

X_train_f4, X_test_f4, y_train_f4, y_test_f4 = train_test_split(X_filtered_type4, y_filtered_type4, test_size=0.2, random_state=42)

# Train and predict Type4
y_pred_type4, model_type4 = train_and_predict_type4(X_train_f4, y_train_f4, X_test_f4)
accuracy_type4 = accuracy_score(y_test_f4, y_pred_type4)
print(f'Type4 Accuracy: {accuracy_type4 * 100:.2f}%')
