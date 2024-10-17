import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info logs

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import keras
from keras.layers import Dense, Input
import joblib

# Load Data
data_train = pd.read_csv(r"C:\Users\shalm\Downloads\train.csv")
print(data_train.info())  # Check the structure of the dataset
print(data_train.describe())  # Basic statistics

# Mapping category (if 'category' column exists)
category_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8}
if 'category' in data_train.columns:
    data_train["category"] = data_train["category"].map(category_map)

# Clean Data (converting to numeric and handling errors)
# Replace commas in numeric columns and convert to numeric
data_train["views"] = pd.to_numeric(data_train["views"].str.replace(',', ''), errors='coerce')
data_train["comment"] = pd.to_numeric(data_train["comment"].str.replace(',', ''), errors='coerce')
data_train["likes"] = pd.to_numeric(data_train["likes"].str.replace(',', ''), errors='coerce')
data_train["dislikes"] = pd.to_numeric(data_train["dislikes"].str.replace(',', ''), errors='coerce')
data_train["adview"] = pd.to_numeric(data_train["adview"], errors='coerce')

# Drop rows with missing or invalid data
data_train.dropna(inplace=True)

# Label encoding for categorical features
data_train['vidid'] = LabelEncoder().fit_transform(data_train['vidid'])
data_train['published'] = LabelEncoder().fit_transform(data_train['published'])

# Duration conversion function
def check_duration(duration):
    if isinstance(duration, str) and duration.startswith('PT'):
        duration = duration[2:]  # Strip the 'PT' prefix
        h, m, s = 0, 0, 0  # Default values for hours, minutes, and seconds

        if 'H' in duration:
            h = int(duration.split('H')[0])
            duration = duration.split('H')[1]

        if 'M' in duration:
            m = int(duration.split('M')[0])
            duration = duration.split('M')[1] if 'M' in duration else ''

        if 'S' in duration:
            s = int(duration.split('S')[0])

        return h * 3600 + m * 60 + s
    return 0

# Ensure the 'duration' column is a string before applying the conversion
data_train['duration'] = data_train['duration'].astype(str)
data_train['duration'] = data_train['duration'].apply(check_duration)

# Additional Data Exploration: Boxplots for Numeric Columns
plt.figure(figsize=(15, 10))
sns.boxplot(data=data_train[["views", "likes", "dislikes", "comment", "adview"]])
plt.title("Boxplot of Numeric Variables")
plt.savefig('boxplot_numeric_columns.png')
plt.show(block=True)

# Histogram for category
plt.ion()
plt.hist(data_train["category"], bins=8)
plt.title("Category Distribution")
plt.savefig('category_histogram.png') 
plt.show(block=True)

# Plot for adview
plt.plot(data_train["adview"])
plt.title("Adview Plot")
plt.show(block=True)

# Remove outliers in adview
data_train = data_train[data_train["adview"] < 2000000]

# Correlation heatmap
f, ax = plt.subplots(figsize=(10, 8))
corr = data_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot=True)
plt.title("Correlation Heatmap")
plt.show(block=True)

# Log Transformation of target (adview) to handle skewness
data_train["adview"] = np.log1p(data_train["adview"])  # log(1 + x)

# Target variable and feature selection
Y_train = data_train['adview']
data_train = data_train.drop(["adview", "vidid"], axis=1)

# Split the dataset into training, validation, and test sets
X_train, X_rem, y_train, y_rem = train_test_split(data_train, Y_train, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Print function for error metrics
def print_error(X_test, y_test, model_name):
    prediction = model_name.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, prediction))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, prediction))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# Linear Regression
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_train, y_train)
print("Linear Regression Performance on Validation Set")
print_error(X_val, y_val, linear_regression)

# Decision Tree Regressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)
print("Decision Tree Performance on Validation Set")
print_error(X_val, y_val, decision_tree)

# Random Forest Regressor with tuned hyperparameters
n_estimators = 500  # Increased estimators
max_depth = 30
min_samples_split = 10
min_samples_leaf = 4
random_forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
random_forest.fit(X_train, y_train)
print("Random Forest Performance on Validation Set")
print_error(X_val, y_val, random_forest)

# Support Vector Regressor (SVR) with kernel and regularization
supportvector_regressor = SVR(kernel='rbf', C=100, gamma=0.1)
supportvector_regressor.fit(X_train, y_train)
print("SVR Performance on Validation Set")
print_error(X_val, y_val, supportvector_regressor)

# Artificial Neural Network (ANN) with Keras
ann = keras.models.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),  # Increased number of neurons
    Dense(32, activation="relu"),
    Dense(1)
])

# Compile the ANN
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.mean_squared_error
ann.compile(optimizer=optimizer, loss=loss, metrics=["mean_squared_error"])

# Train the ANN
history = ann.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=32)  # Adjusted epochs and batch size

# Display model summary
ann.summary()

# Evaluate ANN performance on validation set
print("ANN Performance on Validation Set")
print_error(X_val, y_val, ann)

# Save models
joblib.dump(decision_tree, "decisiontree_youtubeadview.pkl")
ann.save("ann_youtubeadview.keras")

# Make predictions on test set with the best model (Example: Random Forest)
test_predictions = random_forest.predict(X_test)

# Save test predictions to a CSV file
test_results = pd.DataFrame({"Actual": y_test, "Predicted": test_predictions})
test_results.to_csv("test_predictions.csv", index=False)
print("Test predictions saved to 'test_predictions.csv'")