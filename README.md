# Company_Profit

- The goal of this project is to predict the profit of a company based on its city location and expenditures on Research and Development (R&D), Marketing, and Administration.
- Firstly, I obtained the dataset from Kaggle. I then explored the data, removing any null values, duplicate entries, and outliers.
- Next, I used various types of plots to visualize the data and analyze the correlations between the independent variables and the target variable.
- After that, I performed feature engineering on the data by applying standard scaling to the numerical features and converting the categorical features into numerical values using one-hot encoding.
- I then proceeded to build models such as Random Forest, Decision Tree, Linear Regression, and Support Vector Regressor. I fine-tuned these models using Randomized Grid Search to find the best parameters for each one. Finally, I saved the model with the highest accuracy, along with the preprocessing steps applied to the features, using a pipeline and stored it in a pickle file.
- Finally, I deployed the model on my local host using Flask.
