from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from features.preprocess import *
from features.embeddings import *
from model.catboost import CatBoost
from modelling.modelling import *
from modelling.data_model import *
from features.input_data import get_input_data
from model.gradient_boosting import GradientBoosting
import random
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # Load the input data
    df = get_input_data()
    return df


def preprocess_data(df):
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    return df


def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)


def perform_modelling_with_gradientboost(data: Data, df: pd.DataFrame):
    model = GradientBoosting("GradientBoosting", data.get_embeddings(), data.get_type())
    model.train(data)
    y_pred_type2 = model.predict(data.X_test)
    verify_predictions_with_gradientboost(model, data, y_pred_type2)


def verify_predictions_with_gradientboost(model, data, y_pred_type2):
    y_test = data.y_test
    df_test = pd.DataFrame({'y2': y_test, 'y2_pred': y_pred_type2})
    
    # Ensure both y2 and y2_pred are of the same data type
    df_test['y2'] = df_test['y2'].astype(str)
    df_test['y2_pred'] = df_test['y2_pred'].astype(str)

    # Stage 1: Verify Type 2 predictions
    accuracy_type2 = accuracy_score(df_test['y2'], df_test['y2_pred'])
    print(f"Accuracy for Type 2: {accuracy_type2}")
    print("Classification Report for Type 2:")
    print(classification_report(df_test['y2'], df_test['y2_pred'], zero_division=0))

    # Initialize placeholders for the next stages
    df_test_type3 = None
    df_test_type4 = None

    # Stage 2: Verify Type 3 predictions for correctly predicted Type 2 instances
    correct_type2_indices = df_test['y2'] == df_test['y2_pred']
    if np.any(correct_type2_indices):
        X_test_type3 = data.X_test[correct_type2_indices]
        y_test_type3 = data.y_test[correct_type2_indices] if len(data.y_test) == len(data.X_test) else data.y[correct_type2_indices]

        if len(y_test_type3) > 0:
            y_pred_type3 = model.predict(X_test_type3)
            df_test_type3 = pd.DataFrame({'y3': y_test_type3, 'y3_pred': y_pred_type3})

            # Ensure both y3 and y3_pred are of the same data type
            df_test_type3['y3'] = df_test_type3['y3'].astype(str)
            df_test_type3['y3_pred'] = df_test_type3['y3_pred'].astype(str)

            accuracy_type3 = accuracy_score(df_test_type3['y3'], df_test_type3['y3_pred'])
            print(f"Accuracy for Type 3: {accuracy_type3}")
            print("Classification Report for Type 3:")
            print(classification_report(df_test_type3['y3'], df_test_type3['y3_pred'], zero_division=0))
        else:
            print("No correct Type 2 predictions, skipping Type 3 evaluation")
    else:
        print("No correct Type 2 predictions, skipping Type 3 evaluation")

    # Stage 3: Verify Type 4 predictions for correctly predicted Type 2 and Type 3 instances
    if df_test_type3 is not None:
        correct_type3_indices = df_test_type3['y3'] == df_test_type3['y3_pred']
        if np.any(correct_type3_indices):
            X_test_type4 = X_test_type3[correct_type3_indices]
            y_test_type4 = y_test_type3[correct_type3_indices] if len(y_test_type3) == len(X_test_type3) else data.y[correct_type3_indices]

            if len(y_test_type4) > 0:
                y_pred_type4 = model.predict(X_test_type4)
                df_test_type4 = pd.DataFrame({'y4': y_test_type4, 'y4_pred': y_pred_type4})

                # Ensure both y4 and y4_pred are of the same data type
                df_test_type4['y4'] = df_test_type4['y4'].astype(str)
                df_test_type4['y4_pred'] = df_test_type4['y4_pred'].astype(str)

                accuracy_type4 = accuracy_score(df_test_type4['y4'], df_test_type4['y4_pred'])
                print(f"Accuracy for Type 4: {accuracy_type4}")
                print("Classification Report for Type 4:")
                print(classification_report(df_test_type4['y4'], df_test_type4['y4_pred'], zero_division=0))
            else:
                print("No correct Type 3 predictions, skipping Type 4 evaluation")
        else:
            print("No correct Type 3 predictions, skipping Type 4 evaluation")


def perform_modelling_with_catboost(data: Data, df: pd.DataFrame):
    model = CatBoost("CatBoost", data.get_embeddings(), data.get_type())
    model.train(data)
    y_pred_type2 = model.predict(data.X_test)
    verify_predictions_with_catboost(model, data, y_pred_type2)


def verify_predictions_with_catboost(model, data, y_pred_type2):
    y_test = data.y_test
    flattened_y_pred_type2 = y_pred_type2.reshape(-1)
    df_test = pd.DataFrame({'y2': y_test, 'y2_pred': flattened_y_pred_type2})

    # Ensure both y2 and y2_pred are of the same data type
    df_test['y2'] = df_test['y2'].astype(str)
    df_test['y2_pred'] = df_test['y2_pred'].astype(str)

    # Stage 1: Verify Type 2 predictions
    accuracy_type2 = accuracy_score(df_test['y2'], df_test['y2_pred'])
    print(f"Accuracy for Type 2: {accuracy_type2}")
    print("Classification Report for Type 2:")
    print(classification_report(df_test['y2'], df_test['y2_pred'], zero_division=0))

    # Initialize placeholders for the next stages
    df_test_type3 = None
    df_test_type4 = None

    # Stage 2: Verify Type 3 predictions for correctly predicted Type 2 instances
    correct_type2_indices = df_test['y2'] == df_test['y2_pred']
    if np.any(correct_type2_indices):
        X_test_type3 = data.X_test[correct_type2_indices]
        y_test_type3 = data.y_test[correct_type2_indices] if len(data.y_test) == len(data.X_test) else data.y[
            correct_type2_indices]

        if len(y_test_type3) > 0:
            y_pred_type3 = model.predict(X_test_type3)
            df_test_type3 = pd.DataFrame({'y3': y_test_type3, 'y3_pred': y_pred_type3.reshape(-1)})

            # Ensure both y3 and y3_pred are of the same data type
            df_test_type3['y3'] = df_test_type3['y3'].astype(str)
            df_test_type3['y3_pred'] = df_test_type3['y3_pred'].astype(str)

            accuracy_type3 = accuracy_score(df_test_type3['y3'], df_test_type3['y3_pred'])
            print(f"Accuracy for Type 3: {accuracy_type3}")
            print("Classification Report for Type 3:")
            print(classification_report(df_test_type3['y3'], df_test_type3['y3_pred'], zero_division=0))
        else:
            print("No correct Type 2 predictions, skipping Type 3 evaluation")
    else:
        print("No correct Type 2 predictions, skipping Type 3 evaluation")

    # Stage 3: Verify Type 4 predictions for correctly predicted Type 2 and Type 3 instances
    if df_test_type3 is not None:
        correct_type3_indices = df_test_type3['y3'] == df_test_type3['y3_pred']
        if np.any(correct_type3_indices):
            X_test_type4 = X_test_type3[correct_type3_indices]
            y_test_type4 = y_test_type3[correct_type3_indices] if len(y_test_type3) == len(X_test_type3) else data.y[
                correct_type3_indices]

            if len(y_test_type4) > 0:
                y_pred_type4 = model.predict(X_test_type4)
                df_test_type4 = pd.DataFrame({'y4': y_test_type4, 'y4_pred': y_pred_type4.reshape(-1)})

                # Ensure both y4 and y4_pred are of the same data type
                df_test_type4['y4'] = df_test_type4['y4'].astype(str)
                df_test_type4['y4_pred'] = df_test_type4['y4_pred'].astype(str)

                accuracy_type4 = accuracy_score(df_test_type4['y4'], df_test_type4['y4_pred'])
                print(f"Accuracy for Type 4: {accuracy_type4}")
                print("Classification Report for Type 4:")
                print(classification_report(df_test_type4['y4'], df_test_type4['y4_pred'], zero_division=0))
            else:
                print("No correct Type 3 predictions, skipping Type 4 evaluation")
        else:
            print("No correct Type 3 predictions, skipping Type 4 evaluation")


def acgfx(df):
    df['text'] = df['Ticket Summary'] + ' ' + df['Interaction content']

    # Vectorize the text data using TF-IDF
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text']).toarray()

    le_y2 = LabelEncoder()
    le_y3 = LabelEncoder()
    le_y4 = LabelEncoder()

    df['y2_encoded'] = le_y2.fit_transform(df['y2'])
    df['y3_encoded'] = le_y3.fit_transform(df['y3'])
    df['y4_encoded'] = le_y4.fit_transform(df['y4'])

    # Target variables
    Y = df[['y2_encoded', 'y3_encoded', 'y4_encoded']].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define the base model
    base_model = GradientBoostingClassifier(random_state=42)

    # Create the ClassifierChain
    chain_model = ClassifierChain(base_model, order='random', random_state=42)

    # Train the model
    chain_model.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = chain_model.predict(X_test)

    # Ensure predictions are integers
    Y_pred = Y_pred.astype(int)

    # Ensure Y_test is also integer
    Y_test = Y_test.astype(int)

    row_accuracies = [calculate_row_accuracy(Y_test[i], Y_pred[i]) for i in range(Y_test.shape[0])]

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Y2_test': le_y2.inverse_transform(Y_test[:, 0]),
        'Y3_test': le_y3.inverse_transform(Y_test[:, 1]),
        'Y4_test': le_y4.inverse_transform(Y_test[:, 2]),
        'Y2_pred': le_y2.inverse_transform(Y_pred[:, 0]),
        'Y3_pred': le_y3.inverse_transform(Y_pred[:, 1]),
        'Y4_pred': le_y4.inverse_transform(Y_pred[:, 2]),
        'row_accuracies': row_accuracies
    })

    # Save the results DataFrame to a CSV file
    results_df.to_csv('true_and_predicted_results.csv', index=False)

    return Y_test, Y_pred


# Custom accuracy calculation function for each row
def calculate_row_accuracy(y_true, y_pred):
    if y_true[0] == y_pred[0]:  # y2
        if y_true[1] == y_pred[1] and y_true[2] == y_pred[2]:  # y3 and y4
            return 1.0
        elif y_true[1] == y_pred[1]:  # y3
            return 2/3
        else:
            return 1/3
    else:
        return 0.0


def print_class_distribution(df):
    print("Class distribution for y2:")
    print(df['y2'].value_counts())
    print("Class distribution for y3:")
    print(df['y3'].value_counts())
    print("Class distribution for y4:")
    print(df['y4'].value_counts())


if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    Y_test, Y_pred = acgfx(df)
