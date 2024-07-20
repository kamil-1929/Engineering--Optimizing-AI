from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

def verify_predictions(model, X_test, Y_test, le_y2, le_y3, le_y4):
    Y_pred = model.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = Y_test.astype(int)

    row_accuracies = [calculate_row_accuracy(Y_test[i], Y_pred[i]) * 100 for i in range(Y_test.shape[0])]

    Y2_test_str = le_y2.inverse_transform(Y_test[:, 0]).astype(str)
    Y3_test_str = le_y3.inverse_transform(Y_test[:, 1]).astype(str)
    Y4_test_str = le_y4.inverse_transform(Y_test[:, 2]).astype(str)
    Y2_pred_str = le_y2.inverse_transform(Y_pred[:, 0]).astype(str)
    Y3_pred_str = le_y3.inverse_transform(Y_pred[:, 1]).astype(str)
    Y4_pred_str = le_y4.inverse_transform(Y_pred[:, 2]).astype(str)

    results_df = pd.DataFrame({
        'Y2_test': Y2_test_str,
        'Y3_test': Y3_test_str,
        'Y4_test': Y4_test_str,
        'Y2_pred': Y2_pred_str,
        'Y3_pred': Y3_pred_str,
        'Y4_pred': Y4_pred_str,
        'row_accuracies': row_accuracies
    })

    accuracy_Y2 = accuracy_score(results_df['Y2_test'], results_df['Y2_pred'])
    accuracy_Y3 = accuracy_score(results_df['Y3_test'], results_df['Y3_pred'])
    accuracy_Y4 = accuracy_score(results_df['Y4_test'], results_df['Y4_pred'])

    report_Y2 = classification_report(results_df['Y2_test'], results_df['Y2_pred'], zero_division=0)
    report_Y3 = classification_report(results_df['Y3_test'], results_df['Y3_pred'], zero_division=0)
    report_Y4 = classification_report(results_df['Y4_test'], results_df['Y4_pred'], zero_division=0)

    print(f"Overall Accuracy for Type 2: {accuracy_Y2:.2%}")
    print("Classification Report for Type 2:")
    print(report_Y2)
    
    print(f"Overall Accuracy for Type 3: {accuracy_Y3:.2%}")
    print("Classification Report for Type 3:")
    print(report_Y3)
    
    print(f"Overall Accuracy for Type 4: {accuracy_Y4:.2%}")
    print("Classification Report for Type 4:")
    print(report_Y4)

    appgallery_games_group = ['AppGallery-Install/Upgrade', 'AppGallery-Use', 'Games']
    in_app_purchase_group = ['Payment', 'Payment issue', 'In-App Purchase']
    
    appgallery_games_accuracy = calculate_group_accuracy(results_df, 'Y3_test', appgallery_games_group)
    in_app_purchase_accuracy = calculate_group_accuracy(results_df, 'Y3_test', in_app_purchase_group)
    overall_average_accuracy = results_df['row_accuracies'].mean()
    
    print(f"Average Accuracy for AppGallery & Games group: {appgallery_games_accuracy:.2f}%")
    print(f"Average Accuracy for In-App Purchase group: {in_app_purchase_accuracy:.2f}%")
    print(f"Overall Average Accuracy for all groups: {overall_average_accuracy:.2f}%")

    results_df['Predicted Classes'] = list(Y_pred)
    results_df['True Classes'] = list(Y_test)

    results_df.to_csv('true_and_predicted_results.csv', index=False)

    overall_accuracy = np.mean([accuracy_Y2, accuracy_Y3, accuracy_Y4])
    #print(f"Overall Model Accuracy: {overall_accuracy * 100:.2f}%")

    return results_df

def calculate_row_accuracy(y_true, y_pred):
    if y_true[0] == y_pred[0]:
        if y_true[1] == y_pred[1] and y_true[2] == y_pred[2]:
            return 1.0
        elif y_true[1] == y_pred[1]:
            return 2/3
        else:
            return 1/3
    else:
        return 0.0

def calculate_group_accuracy(results_df, column, group):
    group_df = results_df[results_df[column].isin(group)]
    if len(group_df) > 0:
        return group_df['row_accuracies'].mean()
    else:
        return 0.0
