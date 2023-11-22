# Step 1: Data Loading and Exploration
import pandas as pd
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Step 2: Data Preprocessing
X = iris.data
y = iris.target

# Step 3: Model Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional metrics
print(classification_report(y_test, y_pred))
