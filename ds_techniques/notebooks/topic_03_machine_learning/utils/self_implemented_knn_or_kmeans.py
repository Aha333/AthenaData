import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt

class KNNClassifier(BaseEstimator, ClassifierMixin):
    """A simple implementation of K-Nearest Neighbors Classifier"""
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Predict class by majority vote
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
            
        return np.array(predictions)

class KMeans(BaseEstimator):
    """A simple implementation of K-Means Clustering"""
    
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        
    def fit(self, X):
        """Compute k-means clustering."""
        X = np.array(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Initialize centroids randomly
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) 
                                    for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """Predict the closest cluster each sample belongs to."""
        X = np.array(X)
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

def test_against_sklearn():
    """Test implementation against sklearn's implementation"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.datasets import make_classification, make_blobs
    
    # Test KNN
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
    
    custom_knn = KNNClassifier(n_neighbors=3)
    sklearn_knn = KNeighborsClassifier(n_neighbors=3)
    
    custom_knn.fit(X, y)
    sklearn_knn.fit(X, y)
    
    custom_pred = custom_knn.predict(X)
    sklearn_pred = sklearn_knn.predict(X)
    
    knn_accuracy = np.mean(custom_pred == sklearn_pred)
    print(f"KNN accuracy compared to sklearn: {knn_accuracy}")
    
    # Test KMeans
    X, y = make_blobs(n_samples=300, centers=4, random_state=42)
    
    custom_kmeans = KMeans(n_clusters=4, random_state=42)
    sklearn_kmeans = SklearnKMeans(n_clusters=4, random_state=42)
    
    custom_kmeans.fit(X)
    sklearn_kmeans.fit(X)
    
    custom_labels = custom_kmeans.labels_
    sklearn_labels = sklearn_kmeans.labels_
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=custom_labels)
    plt.scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    plt.title('Custom KMeans')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels)
    plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1],
               c='red', marker='x', s=200, linewidths=3)
    plt.title('Sklearn KMeans')
    
    plt.tight_layout()
    plt.show()
    
    return custom_knn, sklearn_knn, custom_kmeans, sklearn_kmeans
