# Import Libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_excel('HealthSleep.xlsx')

# Handle Missing Values
data = data.dropna()

# Define Features and Target
X = data.drop(['User ID', 'result'], axis=1)
y = data['result']

# Identify Categorical Variables
categorical_cols = ['Gender', 'Physical Activity Level', 'Dietary Habits', 'Sleep Disorders', 'Medication Usage']

# One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=0)

# Build KNN Model
knn = KNeighborsClassifier()

# Hyperparameter Tuning
param_grid = {
    'n_neighbors': [1, 3, 5],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(knn, param_grid, cv=2)  # cv=2 due to small dataset
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
knn_best = grid.best_estimator_

# Predict
y_pred = knn_best.predict(X_test)

# Evaluate
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Export Model
joblib.dump(knn_best, "healthsleep.pkl")
