# Import necessary libraries for data processing and modeling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load and inspect the dataset
dataframe = pd.read_csv('/Users/sarps/Downloads/creditcard_2023.csv')
print(dataframe.head())
print(dataframe.info())

# Check for missing data
missing_values = dataframe.isnull().sum()
print("Missing values:\n", missing_values)

# Prepare feature set and labels
features = dataframe.drop(columns=['id', 'Class'], errors='ignore')
labels = dataframe['Class']
print("Feature columns:\n", features.columns.tolist())

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Display class distribution
class_distribution = pd.Series(y_train).value_counts(normalize=True)
print("Class distribution in training set:\n", class_distribution)

# Initialize Random Forest model with chosen parameters
random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)


# Perform cross-validation
cv_f1_scores = cross_val_score(random_forest_model, X_train_normalized, y_train, cv=5, scoring='f1')
print("\nCross-validation F1 scores:", cv_f1_scores)
print("Average F1 score:", np.mean(cv_f1_scores))

# Train the model on the training data
random_forest_model.fit(X_train_normalized, y_train)

# Make predictions on the test data
predictions = random_forest_model.predict(X_test_normalized)

# Evaluate the model with a classification report
report = classification_report(y_test, predictions)
print("Classification Report:\n", report)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()


# Determine feature importance
feature_importances = random_forest_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", importance_df.head())

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance Ranking')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Compute and plot the correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = features.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Calculate and plot the ROC curve
proba_predictions = random_forest_model.predict_proba(X_test_normalized)[:, 1]
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, proba_predictions)
roc_auc_value = auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()