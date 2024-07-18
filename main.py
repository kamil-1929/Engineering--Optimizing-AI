from features.preprocess import *
from features.embeddings import *
from model.catboost import CatBoost
from modelling.modelling import *
from modelling.data_model import *
from model.gradient_boosting import GradientBoosting
import random
from sklearn.metrics import classification_report, accuracy_score

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


def perform_modelling_with_gradientboost(data: Data, df: pd.DataFrame, name):
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


def perform_modelling_with_catboost(data: Data, df: pd.DataFrame, name):
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
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    print_class_distribution(df)  # Add this line to print class distributions
    
    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        print("#####OUTPUT RESULT USING THE GRADIENT BOOSTING#####")
        perform_modelling_with_gradientboost(data, group_df, name)

    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        print("#####OUTPUT RESULT FOR CATBOOST#####")
        perform_modelling_with_catboost(data, group_df, name)
