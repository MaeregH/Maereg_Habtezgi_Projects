# Customer Segmentation Using K-Means Clustering

## Overview
This project performs **customer segmentation** on a credit card customer dataset using **unsupervised machine learning**. The goal is to identify distinct groups of customers based on their spending behavior, balance patterns, and credit usage, without relying on predefined labels.

By applying **K-Means clustering** and **Principal Component Analysis (PCA)**, the project uncovers meaningful customer segments that can be used for business insights such as targeted marketing, risk assessment, and customer profiling.

---

## Dataset
The dataset contains anonymized credit card customer information, including:

- Account balance  
- Purchase amounts and frequency  
- Cash advance behavior  
- Credit limits  
- Payment behavior  
- Usage frequency metrics  

Each row represents a customer, and all features are numerical.

---

## Methodology

### 1. Data Preprocessing
- Inspected dataset structure and data types
- Standardized all features using `StandardScaler`
- Ensured fair distance calculations for K-Means clustering

---

### 2. Selecting the Number of Clusters
- Used the **Elbow Method** to evaluate different cluster counts
- Plotted inertia (within-cluster sum of squares) versus number of clusters
- Selected **8 clusters** based on the elbow point

---

### 3. K-Means Clustering
- Applied K-Means with the selected number of clusters
- Assigned each customer a cluster label
- Transformed cluster centroids back to the original feature scale for interpretation

---

### 4. Cluster Analysis
- Combined cluster labels with the original dataset
- Visualized feature distributions for each cluster
- Identified behavioral patterns such as:
  - High spenders
  - Low-usage customers
  - High cash-advance users
  - Customers carrying large balances

---

### 5. Dimensionality Reduction and Visualization
- Reduced feature space to two dimensions using **PCA**
- Visualized customers in PCA space colored by cluster assignment
- Assessed separation and structure of customer segments

---

## Results
The model identifies several distinct customer segments with different financial behaviors. These insights can support:

- Targeted marketing strategies
- Credit risk assessment
- Customer value analysis
- Data-driven decision-making

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## How to Run

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn

This step helps identify distinct customer profiles such as:

High spenders

Low-usage customers

Customers with high cash advance usage

Customers who carry high balances

5. Dimensionality Reduction and Visualization

Principal Component Analysis (PCA) reduces the data to two dimensions.

Customers are plotted in PCA space and colored by cluster assignment.

This visualization helps assess the separation and structure of the clusters.

Results

The analysis reveals multiple distinct customer segments with different financial behaviors. These segments can be used to:

Improve customer targeting strategies

Identify high-risk or high-value customers

Support data-driven decision-making in finance and marketing contexts

Technologies Used

Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

How to Run

Clone this repository

Install the required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn


Open and run the Jupyter notebook:

jupyter notebook CustomerProject2.ipynb

Future Improvements

Experiment with alternative clustering methods (e.g., DBSCAN, Hierarchical Clustering)

Add quantitative cluster validation metrics (Silhouette Score, Davies–Bouldin Index)

Perform deeper business-oriented interpretation of each cluster

Incorporate time-based customer behavior if available
