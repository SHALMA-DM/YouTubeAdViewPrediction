This repository contains a project that predicts YouTube video adviews using machine learning models. The goal of the project is to use various attributes of YouTube videos to estimate the number of adviews. Different regression models and an Artificial Neural Network (ANN) were trained and compared to predict the adviews accurately.

**Files in the Repository:**
YouTube Prediction.py: The main Python script that contains the code for preprocessing the data, training the models, and making predictions.
ann_youtubeadview.keras: Saved model file for the trained Artificial Neural Network (ANN).
boxplot_numeric_columns.png: A boxplot visualization of the numeric columns in the dataset.
category_histogram.png: A histogram visualization showing the distribution of video categories.
decisiontree_youtubeadview.joblib: Saved model file for the trained Decision Tree Regressor.
test_predictions.csv: CSV file containing the predictions made by the models on the test set.

#**Project Overview:**
The project uses a dataset containing YouTube video attributes like views, likes, dislikes, comments, and more to predict the number of adviews. Several regression models were trained on this dataset to find the one that yields the best results.

#**Models Implemented:**
   Linear Regression
   Decision Tree Regressor
   Random Forest Regressor
   Support Vector Regressor
   Artificial Neural Network (ANN)

#**Data Preprocessing:**
The dataset includes features like:
   vidid: Video ID (ignored during modeling).
   adview: The target variable representing the number of adviews.
   views: Total views on the video.
   likes: Number of likes.
   dislikes: Number of dislikes.
   comment: Number of comments.
   published: Date when the video was published.
   duration: Duration of the video.
   category: Category of the video.
#**Steps:**
    >Converted non-numeric values such as video duration to a numeric format.
    >Transformed and cleaned the categorical columns.
    >Handled missing values and outliers in the data.
    >Visualized the dataset using histograms and boxplots to understand the 
    distributions of key features.

Model Evaluation:
Each model was evaluated using the following metrics:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)

Installation:
To run this project locally, follow these steps:

Clone the repository:
git clone https://github.com/SHALMA-DM/YouTubeAdViewPrediction.git

cd YouTubeAdViewPrediction

Install the required dependencies:
pip install -r requirements.txt

Run the project:
python YouTube\ Prediction.py

Visualizations:
The repository contains the following visualizations to help understand the dataset:

Boxplot (boxplot_numeric_columns.png): Shows the distribution of numeric columns.
Category Histogram (category_histogram.png): Displays the distribution of YouTube video categories.

Conclusion:
This project demonstrates the use of different machine learning models and neural networks for predicting YouTube adviews. Random Forest and ANN tend to give the best results for this dataset. Further improvements could include hyperparameter tuning and more advanced feature engineering.

