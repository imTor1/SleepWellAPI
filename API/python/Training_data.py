# Import Python Libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_excel('D:\\API\\python\\Health_Sleep.xlsx')

# Display the data (optional, for debugging)
print(data.head())

# Encode categorical variables
label_encoder = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = label_encoder.fit_transform(data[column])

# Define features (X) and target (y)
# Assuming the target column is the last one in the dataset ('Sleep Disorders' for prediction)
x = data.iloc[:, 1:len(data.columns)-1]  # Use columns excluding 'User ID' and target column
y = data.iloc[:, len(data.columns)-1]    # Target column (e.g., 'Sleep Disorders')

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Build KNN Model
net = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
net.fit(x_train, y_train)

# Predict on the test set
y_pred = net.predict(x_test)

# Evaluate Model
print('Accuracy metrics:')
print(classification_report(y_test, y_pred))

# Export Model
joblib.dump(net, "health_sleep.pkl")
