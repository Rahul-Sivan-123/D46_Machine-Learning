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
   
   **Suitability:**
   
1. The Iris dataset has well-defined clusters corresponding to species.
2. K-Means efficiently groups data based on numerical features like petal and sepal measurements.
3. Since the dataset is small and relatively balanced, K-Means performs well in detecting natural groupings.

         B).Hierarchical Clustering:
   How it works: Hierarchical clustering builds a hierarchy of clusters in two ways:  
    
#Agglomerative (bottom-up): Each point starts as its own cluster, and similar clusters merge iteratively.          
#Divisive (top-down): All points start in one cluster and split recursively into smaller clusters.

**Suitability:**

1.The method captures natural hierarchies in the data.  
2. No need to specify the number of clusters beforehand.  
3. Can visualize relationships between clusters using dendrograms.
  
# ML Module End Assignment (Car Price Prediction)

# Overview:
This project aims to analyze and predict car prices in the US market using machine learning techniques. A Chinese automobile company is looking to establish a local manufacturing unit and understand the pricing dynamics to compete with existing US and European manufacturers. The goal is to identify key factors affecting car prices and build a predictive model.

Dataset:
The dataset contains various attributes about cars, including technical specifications, fuel type, body type, engine details, and pricing.

# Dataset Source: https://drive.google.com/file/d/1FHmYNLs9v0Enc-UExEMpitOFGsWvB2dP/view?

Key Features:

##  Categorical: fueltype, carbody, drivewheel, enginetype, cylindernumber, etc.

## Numerical: wheelbase, carlength, carwidth, enginesize, horsepower, etc.

## Target variable:price
   

# Methodology
# 1. Data Preprocessing:

. Removed irrelevant columns (car_ID, CarName).

. Encoded categorical features using one-hot encoding.

. Standardized numerical features using StandardScaler.

. Split dataset into training (80%) and testing (20%) sets.

# 2. Model Implementation:
Trained five regression models to predict car prices:

. Linear Regression

. Decision Tree Regressor

. Random Forest Regressor

. Gradient Boosting Regressor

. Support Vector Regressor (SVR)

# 3. Model Evaluation
Compared models using:

. R-squared (R²)

. Mean Squared Error (MSE)

. Mean Absolute Error (MAE)

    # Best Model Identified: Random Forest Regressor (Highest R², Lowest MSE & MAE)

# 4. Feature Importance Analysis:
Used correlation analysis and Random Forest feature importance.

. Most influential features:

. Engine Size

. Horsepower

. Curb Weight

. Drive Wheel Type

. Fuel Type

# 5. Hyperparameter Tuning
. Applied GridSearchCV to optimize model parameters.

. Improved R² score and reduced prediction errors.

# 6. Visualization
. Heatmap: Shows correlation between numerical features.

. Feature Importance Plot: Ranks top predictors of car price.

. Actual vs. Predicted Prices Scatter Plot: Evaluates model accuracy.

# Inferences

    1.Engine Size & Horsepower have the strongest impact on car prices.

    2.The optimized Random Forest model achieved a higher R² score (0.958875) over initial value (0.88831).

    3.Random Forest Regressor performed best(higher R² score)

****-------------------------*****

# ML Final Project (Online Shoppers Purchasing Intention Dataset)

# Overview:
The goal of this project is to predict whether an online shopping session will lead to a purchase, using user behavior and session-level attributes. By analyzing session features such as page visits, durations, bounce rates, and user types, we aim to uncover behavioral patterns that distinguish buyers from non-buyers. These insights can help e-commerce platforms optimize their user experience.

# Dataset Overview:  
The dataset was sourced from the UCI Machine Learning Repository. It includes information on user behavior during online shopping sessions and whether they resulted in a purchase (`Revenue`).

*Dataset Highlights:*
- **12,330** records
- **17 features** + Target column (`Revenue`)
- Includes **numeric** and **categorical** data  

# Dataset Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

# Key Features:

Categorical Data: `Month`,`VisitorType`,`Weekend`, `Revenue`.

Numerical Data: `Administrative`,`Administrative_Duration`,     `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`.

Target variable: `price` ( `True ` if purchase was made, else  `False `)
   

# Methodology

### 1. **Loading & Data Preprocessing**

- Identified number of rows and features in dataset.

- Checked the dataset for null values.

- Checked number of duplicates and dropped it.

- Identified the unique values.

- Gathered information about the numerical data in dataset.

 #### (a) Identified numerical and categorical Features
  #### (b) Outlier Detection(By IQR Method) and Visualization (Boxplot)
  #### (c) Outlier Removal (By Trimming Method)
  #### (d) Outlier Visualization Before and After Trimming

### i. **Data Encoding of Categorical Features**
- Done the data Encoding of Categorical Features (['Month', 'VisitorType', 'Weekend', 'Revenue'])
- Encoded '**Month**' by Mapping.
- Encoded '**Weekend**' & '**Revenue**' from boolean value to int datatype.
- Encoded '**VisitorType**' by One-hot encoding.

#### (a)  Skewness and Kurtosis check of Dataset
- Checked Skewness and Kurtosis 
- Combined into a dataframe(`distribution_metrics`)

### ii. **Data Transformation**
- Done data transformation by log method.
- Created Histogram plot to view the difference before and after transformation.

### 2. **EDA & Visualization**
- Histograms before and after transformation.
- Correlation heatmap for numeric features
- Categorical analysis using bar plots (`Month`, `VisitorType`, `Weekend`).
  -Revenue by Month (Bar Plot)
  -Revenue by Visitor(Bar Plot)
  -Revenue by Weekend ( Count Plot)

### iii. **Feature Scaling**
- Separated Features and Target
-       X = df_log.drop('Revenue', axis=1)
        y = df_log['Revenue']
- Initialize Standard Scaler
-      scaler=StandardScaler()

- Fit and transformed the numerical features.
-     X_scaled=scaler.fit_transform(X_encoded)








    
