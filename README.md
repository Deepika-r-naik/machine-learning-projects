
### My First ML Project to understand splitting the dataset for testing and training and then used two different models to predict values.

Loading Data: The code starts by importing the necessary libraries and loading a dataset using Pandas from a URL. This dataset contains molecular descriptors and the corresponding solubility values (LogS).

Data Separation: The dataset is split into input variables (X) and the target variable (Y). X contains the molecular descriptors, and Y contains the solubility values.

Data Splitting: Using scikit-learn's train_test_split function, the dataset is split into training and testing sets for both X and Y. This division helps in training the model on one portion and validating its performance on another unseen portion.

Model Building:
a. Linear Regression: The code uses scikit-learn's LinearRegression to build a linear regression model. This model is trained using the training set (x_train and y_train).
b. Random Forest: Another model, RandomForestRegressor, is trained using the same training data to perform regression.

Model Evaluation: The code evaluates the performance of both models using metrics like Mean Squared Error (MSE) and R-squared (R2) score. MSE measures the average squared difference between actual and predicted values. R2 score indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

Model Comparison: The results (MSE and R2 scores) of both models (Linear Regression and Random Forest) are stored in DataFrames and concatenated for comparison.

Data Visualization: A scatter plot is created to visualize the predicted LogS values against the experimental LogS values from the training set. The red line on the plot represents the fitted linear regression line.
