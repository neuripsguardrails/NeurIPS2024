import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df1 = pd.read_csv('llamaguard/results/guardrail_llama_test_results.csv')
df2 = pd.read_csv('nemo/guardrail_nemo_test_results.csv')

# Prepare the combined DataFrame
data = pd.DataFrame({
    'Prompt': df1['prompt'],
    'Llama': df1['predicted'],
    'Nemo': df2['predicted'],
    'True': df1['expected']
})

# Split the data into training and testing sets (80/20 split)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Function to apply Model 1 logic
def model_1(row):
    if row['Llama'] == 'safe':
        return row['Nemo']
    else:
        return row['Llama']

# Function to apply Model 2 logic
def model_2(row):
    if row['Nemo'] == 'safe':
        return row['Llama']
    else:
        return row['Nemo']

# Apply the model logic to the test set
test_data['Model1_Prediction'] = test_data.apply(model_1, axis=1)
test_data['Model2_Prediction'] = test_data.apply(model_2, axis=1)

def custom_confusion_matrix(actual, predicted, positive_label):

    # Initialize the counters
    tp = tn = fp = fn = 0

    # Iterate over the actual and predicted labels
    for a, p in zip(actual, predicted):
        if a == positive_label and p == positive_label:
            tp += 1
        elif a == positive_label and p != positive_label:
            fn += 1
        elif a != positive_label and p == positive_label:
            fp += 1
        elif a != positive_label and p != positive_label:
            tn += 1

    # Print the results
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')

    return tp, fp, fn, tn

# Evaluate each model
def evaluate_model(prompts, predictions, true_labels, model_name):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label='unsafe')
    recall = recall_score(true_labels, predictions, pos_label='unsafe')
    f1 = f1_score(true_labels, predictions, pos_label='unsafe')
    custom_confusion_matrix(true_labels, predictions, "unsafe")
    print(f"{model_name} Metrics (Accuracy, Precision, Recall, F1):", accuracy, precision, recall, f1)

    # Preparing the results DataFrame
    results_df = pd.DataFrame({
        'Prompt': prompts,
        'Predicted': predictions,
        'Actual': true_labels
    })

    # Write the DataFrame to a CSV file
    results_df.to_csv(f"ensemble/{model_name}_model_predictions.csv", index=False)

    # Filter to include only misclassified examples
    failed_predictions_df = results_df[results_df['Predicted'] != results_df['Actual']]

    # Write the misclassified predictions to a CSV file
    failed_predictions_df.to_csv(f"ensemble/failed_{model_name}_model_predictions.csv", index=False)

evaluate_model(test_data['Prompt'], test_data['Model1_Prediction'], test_data['True'], "llama_then_nemo")
evaluate_model(test_data['Prompt'], test_data['Model2_Prediction'], test_data['True'], "nemo_then_llama")