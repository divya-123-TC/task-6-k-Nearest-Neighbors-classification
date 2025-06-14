# task-6-k-Nearest-Neighbors-classification


K-Nearest Neighbors (KNN) Classification – AI & ML Internship Task

Objective:

To implement and understand the K-Nearest Neighbors (KNN) algorithm using the Iris dataset, and:

Normalize features

Experiment with different values of K

Evaluate using accuracy and confusion matrix

Visualize decision boundaries


Dataset Used:

Iris Dataset from sklearn.datasets

Used only the first two features for 2D visualization





Tools & Libraries:

Python

Scikit-learn

NumPy

Pandas

Matplotlib



 Steps Performed:

 1. Data Loading & Preprocessing

Loaded the Iris dataset using sklearn.datasets.load_iris()

Selected first two features for easy visualization

Split the dataset into training and testing sets

Normalized features using StandardScaler


 2. Model Training & Evaluation

Trained KNeighborsClassifier for multiple values of K: 1, 3, 5, 7

Evaluated models using:

Accuracy

Confusion Matrix


Plotted confusion matrices for each K


 3. Decision Boundary Visualization

Visualized the decision boundary for K = 3

Used a meshgrid to classify and plot decision regions

Displayed colored regions indicating how the classifier splits the feature space





Results

K	Accuracy

1	~0.84
3	~0.93
5	~0.89
7	~0.91






 Visuals

Confusion Matrices for K = 1, 3, 5, 7

Decision Boundary plot for K = 3 (using normalized features)










