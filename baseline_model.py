# Task 3: Baseline Logistic Regression Model
# Predict Phase 2 clinical trial success or failure

# Step 1: Import libraries
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix


# Step 2: Load the processed training and testing data
X_train = pd.read_csv("data/processed/model_ready/X_train.csv")
X_test = pd.read_csv("data/processed/model_ready/X_test.csv")

y_train = pd.read_csv("data/processed/model_ready/y_train.csv")["label"]
y_test = pd.read_csv("data/processed/model_ready/y_test.csv")["label"]


# Step 3: Check the data
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)


# Step 4: Identify sentiment features
sentiment_columns = []

for column in X_train.columns:
    if column.startswith("pubmed_sentiment_"):
        sentiment_columns.append(column)

print("\nSentiment features detected:", len(sentiment_columns))
print("Sentiment columns:", sentiment_columns)


# Step 5: Create feature matrices without sentiment
X_train_without_sentiment = X_train.drop(columns=sentiment_columns)
X_test_without_sentiment = X_test.drop(columns=sentiment_columns)

print("\nTraining features without sentiment:",
      X_train_without_sentiment.shape)

print("Testing features without sentiment:",
      X_test_without_sentiment.shape)


# Step 6: Define a function to train and evaluate a model
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):

    # Create and train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict clinical trial outcomes
    y_pred = model.predict(X_test)

    # Predict probabilities of success
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Calculate ROC-AUC if both outcomes are present
    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = float("nan")

    # Display model results
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("Number of features:", X_train.shape[1])
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                labels=[0, 1],
                                zero_division=0))

    # Display confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # Return the performance metrics
    return {
        "Model": model_name,
        "Features": X_train.shape[1],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }


# Step 7: Train logistic regression without sentiment
results = []

results_without_sentiment = train_and_evaluate(
    X_train_without_sentiment,
    X_test_without_sentiment,
    y_train,
    y_test,
    "Logistic Regression Without Sentiment"
)

results.append(results_without_sentiment)


# Step 8: Train logistic regression with sentiment
if len(sentiment_columns) > 0:

    results_with_sentiment = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        "Logistic Regression With Sentiment"
    )

    results.append(results_with_sentiment)

else:
    print("\nNo sentiment features available.")
    print("Skipping the model with sentiment.")


# Step 9: Compare model performance
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION MODEL COMPARISON")
print("=" * 50)

print(results_df.round(4).to_string(index=False))

# Step 10: Save model performance results

# Create the output folder if it does not exist
output_folder = Path("outputs/baseline_model")
output_folder.mkdir(parents=True, exist_ok=True)

# Save the model comparison as a CSV file
results_df.to_csv(
    output_folder / "baseline_model_results.csv",
    index=False
)

print("\nModel results saved successfully!")
print("Results saved in:", output_folder)