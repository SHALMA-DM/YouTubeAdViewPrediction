This repository contains a machine learning project that predicts YouTube video adviews using several regression models and an Artificial Neural Network (ANN). The goal is to use features of YouTube videos like views, likes, comments, and more to estimate the number of adviews.

📁 Project Files
YouTube Prediction.py: Main Python script for data preprocessing, model training, and predictions.
ann_youtubeadview.keras: Trained Artificial Neural Network (ANN) model file.
boxplot_numeric_columns.png: Boxplot visualization of numeric columns.
category_histogram.png: Histogram showing the distribution of video categories.
decisiontree_youtubeadview.joblib: Trained Decision Tree model file.
test_predictions.csv: CSV file with predictions from the models on the test data.
📝 Project Overview
The goal of the project is to predict YouTube video adviews using various features from the dataset. Multiple regression models are used and compared to identify the one with the best performance. Models used include:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Support Vector Regressor
Artificial Neural Network (ANN)
🔄 Data Preprocessing
The dataset contains features such as:

vidid: Video ID (ignored for modeling).
adview: Target variable representing adviews.
views: Total video views.
likes: Number of likes.
dislikes: Number of dislikes.
comment: Number of comments.
published: Video publication date.
duration: Duration of the video.
category: Video category.
Steps Taken:
Converted non-numeric fields like video duration to numeric.
Handled missing values, outliers, and cleaned data.
Visualized key features with histograms and boxplots.
📊 Model Evaluation
The models were evaluated using the following metrics:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Results on Validation Set:
Model	MAE	RMSE
Linear Regression	3622.09	32114.31
Decision Tree	2141.25	28428.92
Random Forest	[TBD]	[TBD]
Support Vector	[TBD]	[TBD]
Artificial Neural Network	[TBD]	[TBD]
📂 Visualizations
Boxplot: boxplot_numeric_columns.png
Visualizes the distribution of numeric columns.

Category Histogram: category_histogram.png
Shows the distribution of YouTube video categories.

🛠️ Installation
To run this project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/SHALMA-DM/YouTubeAdViewPrediction.git
cd YouTubeAdViewPrediction
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the main Python script:

bash
Copy code
python YouTube\ Prediction.py
🤖 Models
The following machine learning models were used in this project:

Linear Regression: A simple linear model.
Decision Tree Regressor: A tree-based model that predicts based on decisions at nodes.
Random Forest Regressor: An ensemble of decision trees for more accurate predictions.
Support Vector Regressor (SVR): A model that tries to fit the best line within a certain threshold.
Artificial Neural Network (ANN): A neural network model trained to predict adviews.
🔍 Future Work
Hyperparameter tuning to improve model performance.
Exploration of additional features and advanced feature engineering.
Comparison of deep learning methods for improved accuracy.
📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.
