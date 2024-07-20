from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import os

# Custom imports for preprocessing and input data loading
from features.preprocess import de_duplication, noise_remover
from features.input_data import get_input_data
from visualize.visualization import generate_sunburst_chart  
from model.randomforest import perform_modelling_with_randomforest
from model.evaluate import verify_predictions, calculate_group_accuracy, calculate_row_accuracy

seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    df['Ticket Summary'] = df['Ticket Summary'].fillna('Missing Ticket Summary')
    df['Interaction content'] = df['Interaction content'].fillna('Missing Interaction Content')
    return df

def get_embeddings(df: pd.DataFrame):
    df['text'] = df['Ticket Summary'] + ' ' + df['Interaction content']
    tfidf = TfidfVectorizer(stop_words='english', min_df=1, max_df=0.9)
    X = tfidf.fit_transform(df['text'])
    return X, df

def reencode_labels(labels):
    labels = labels.fillna('Missing')
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    return new_labels, le

def acgfx(df):
    df = preprocess_data(df)
    df['y2_encoded'], le_y2 = reencode_labels(df['y2'])
    df['y3_encoded'], le_y3 = reencode_labels(df['y3'])
    df['y4_encoded'], le_y4 = reencode_labels(df['y4'])

    X, _ = get_embeddings(df)
    Y = df[['y2_encoded', 'y3_encoded', 'y4_encoded']].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
    model = perform_modelling_with_randomforest(X_train, Y_train)

    return verify_predictions(model, X_test, Y_test, le_y2, le_y3, le_y4)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    results_df = acgfx(df)

    csv_file_path = 'true_and_predicted_results.csv'
    output_image_path = os.path.join('images', 'sunburst_chart.png')
    os.makedirs('images', exist_ok=True) 
    generate_sunburst_chart(csv_file_path, output_image_path)
