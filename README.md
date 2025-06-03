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
Best-performing model: Random Forest Regressor, as ensemble methods tend to achieve high accuracy by reducing overfitting while capturing complex patterns.

Worst-performing model: SVR, since Linear Regression assumes a linear relationship, which may not hold for housing prices.

# ML Assignment-4 (Classification Problem)
Assignment-4 is about ML Classification Problem.

The data set used in this assignment is breast cancer dataset from the sklearn library.

# Loading & Pre-Processing:
check for missing values using .isnull().sum(), and if any exist, we can fill or drop them.Ensures data integrity and prevents errors in model training.

   ## Feature Scaling (Standardization)-
 Apply StandardScaler from sklearn.preprocessing to normalize the features for better model performance.
 
 # Classification Algorithm Implementation
# 1. Logistic Regression
# 2. Decision Tree Classifier
How It Works: Decision Trees split data recursively based on feature thresholds, forming a tree structure. Each internal node represents a decision based on a feature.

**Suitability:**
It is effective when interactions among features are important. It’s also easy to interpret and doesn’t require feature scaling. 

# 3. Random Forest Classifier
How it works: Random Forest is an ensemble method that combines multiple decision trees to improve performance and reduce overfitting.

**Suitability:**

It handles complex patterns well and often outperforms single decision trees by reducing variance.

# 4. Support Vector Machine (SVM)
How it works: SVM finds an optimal hyperplane that maximizes the margin between two classes. It can also use kernel functions for non-linear classification.

**Suitability:**

Breast cancer data has multiple continuous features, and SVM can be effective in high-dimensional spaces.

# 5. k-Nearest Neighbors (k-NN)
How it works: k-NN classifies a new data point by finding the majority class among its k nearest neighbors in feature space.

**Suitability:**
It’s a simple and intuitive approach, effective when patterns exist within feature similarity.

# Comparison and Findings

Created dictionary to store model predictions and evaluated performance.

# Insights
Best Performing Model: Random Forest Classifier is likely to perform the best. It combines multiple decision trees, reduces overfitting, and excels in feature-rich datasets.


# ML Assignment-5(Clustering Algorithm)
Assignment-5 is about ML Clustering Algorithm.
The data set used in this assignment is Iris dataset from the sklearn library.

# Loading & Pre-Processing:
Load the Iris dataset from sklearn. Drop the species column since this is a clustering problem.

# Clustering Algorithm Implementation:

    A).KMeans Clustering:
    K-Means is a popular clustering algorithm that aims to partition data into 𝑘 clusters. 
How it works:   
1. Initializing 𝑘 centroids randomly.  
2. Assigning each data point to the nearest centroid.
3. Updating centroids based on the mean of assigned points.
4. Repeating steps 2 and 3 until centroids stabilize.
   
   Suitability:
1. The Iris dataset has well-defined clusters corresponding to species.
2. K-Means efficiently groups data based on numerical features like petal and sepal measurements.
3. Since the dataset is small and relatively balanced, K-Means performs well in detecting natural groupings.

    B).Hierarchical Clustering:
    How it works:
   Hierarchical clustering builds a hierarchy of clusters in two ways:
#Agglomerative (bottom-up): Each point starts as its own cluster, and similar clusters merge iteratively.          
#Divisive (top-down): All points start in one cluster and split recursively into smaller clusters.

Suitability:
1.The method captures natural hierarchies in the data.  
2. No need to specify the number of clusters beforehand.  
3. Can visualize relationships between clusters using dendrograms.
Worst Performing Model: Decision Tree Classifier could be the weakest due to potential overfitting and lower generalization capability.
  
