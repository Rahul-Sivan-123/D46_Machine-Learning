# D46_Machine-Learning
Machine Learning assignment submission repo.

# ML Assignment-1(Statistical Measures)
Assignment-1 is about ML statistical measures.

# ML Assignment-2(EDA_and_Preprocessing)
Assignment-2 is about ML EDA_and_Preprocessing.

# ML Assignment-3(Regression)
Assignment-3 is about ML Regression.
The data set used in this assignment is california_housing from sklearn library.
# Pre-Processing:
* Check for missing values using .isnull().sum(), and if any exist, we can fill or drop them.
* Feature Scaling (Standardization)-Appllied StandardScaler from sklearn.preprocessing to normalize the features for better model performance.Standardisation ensures all 
  features have a mean of 0 and a standard deviation of 1 and prevents bias towards variables with larger magnitudes.

# 1. Linear Regression
Linear Regression models the relationship between independent variables and the target variable by fitting a straight-line equation. It minimizes the difference (errors) between actual and predicted values using the least squares method.

**Suitability:**
Since housing prices often depend linearly on factors like population, median income, and number of rooms, Linear Regression can serve as a good baseline model.
It’s interpretable and computationally efficient, but it may struggle with non-linear patterns in the data.

# 2. Decision Tree Regressor
 A Decision Tree splits data into branches based on feature values, making decisions at each node to minimize prediction error. It captures complex relationships but is prone to overfitting.

**Suitability:**
*Suitable when data has non-linear relationships and interactions between features.
*Can capture high-dimensional data patterns but might overfit without proper tuning.

# 3. Random Forest Regressor
 Random Forest is an ensemble learning method that combines multiple Decision Trees. Each tree is trained on a random subset of data, and the final prediction is averaged to improve accuracy and reduce overfitting.

**Suitability:**
Works well when data has complex patterns, reducing overfitting seen in individual decision trees.
Performs better than Decision Trees in generalizing to unseen data.

# 4. Gradient Boosting Regressor
 Gradient Boosting builds models sequentially, correcting errors made by previous models. It optimizes performance using gradient descent and is highly effective for structured data.

**Suitability:**
Useful for datasets where boosting can correct complex relationships.
It performs well with fewer trees but can be computationally expensive.

# 5. Support Vector Regressor (SVR)
SVR finds a hyperplane that best fits the data while allowing a margin of tolerance (epsilon). It works by mapping features into a higher-dimensional space using kernel functions.

**Suitability:**
Suitable for small-to-medium datasets where complex, non-linear relationships exist.

# Comparison and Findings
Best-performing model:Gradient Boosting Regressor or Random Forest Regressor, as ensemble methods tend to achieve high accuracy by reducing overfitting while capturing complex patterns.

Worst-performing model: Typically Linear Regression or SVR, since Linear Regression assumes a linear relationship, which may not hold for housing prices.
  
