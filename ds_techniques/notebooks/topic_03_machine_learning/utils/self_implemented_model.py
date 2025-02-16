
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """A simple implementation of Decision Tree Regressor"""
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Build decision tree regressor."""
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
        return self

    def predict(self, X):
        """Predict using the decision tree regressor."""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow decision tree."""
        n_samples = len(y)
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return {'value': np.mean(y)}

        best_gain = -np.inf
        best_split = None

        # Try each feature
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)[:-1]  # All possible split points
            
            # Try each threshold
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate gain using variance reduction
                current_var = np.var(y)
                left_var = np.var(y[left_mask])
                right_var = np.var(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gain = current_var - (n_left * left_var + n_right * right_var) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None:
            return {'value': np.mean(y)}

        # Create child nodes
        left = self._grow_tree(
            X[best_split['left_mask']], 
            y[best_split['left_mask']], 
            depth + 1
        )
        right = self._grow_tree(
            X[best_split['right_mask']], 
            y[best_split['right_mask']], 
            depth + 1
        )

        return {
            'feature_idx': best_split['feature_idx'],
            'threshold': best_split['threshold'],
            'left': left,
            'right': right
        }

    def _predict_single(self, x, node):
        """Predict for a single sample."""
        if 'value' in node:
            return node['value']
            
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """A simple implementation of Random Forest Regressor with feature sampling."""
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, n_jobs=None, max_features='auto'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.trees = []

    def _fit_single_tree(self, X, y):
        """Fit a single tree with bootstrap sample and feature sampling."""
        n_samples, n_features = X.shape
        # Bootstrap sampling (sample the data points with replacement)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
        
        # Feature sampling (sample a subset of features)
        if self.max_features == 'auto':  # Default behavior: sqrt(n_features)
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'sqrt':  # Same as 'auto'
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':  # Use log2(n_features)
            max_features = int(np.log2(n_features))
        else:
            max_features = self.max_features  # If an integer is provided
        
        # Randomly select the features to be used for the current tree
        feature_indices = np.random.choice(n_features, size=max_features, replace=False)
        X_bootstrap_selected_features = X_bootstrap[:, feature_indices]

        # Create and train the decision tree regressor on the selected features
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        tree.fit(X_bootstrap_selected_features, y_bootstrap)
        
        # Store the tree and the selected features for later predictions
        self.trees.append((tree, feature_indices))

    def fit(self, X, y):
        """Build the random forest regressor by training multiple decision trees with feature sampling."""
        X = np.array(X)  # Ensure that X is a numpy array
        y = np.array(y)  # Ensure that y is a numpy array
        
        # Parallelize tree fitting using joblib
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_tree)(X, y) for _ in range(self.n_estimators)
        )
        
        return self
    
    def predict(self, X):
        """Predict using the random forest regressor."""
        X = np.array(X)  # Ensure that X is a numpy array
        
        # Collect predictions from each tree
        predictions = []
        for tree, feature_indices in self.trees:
            X_selected_features = X[:, feature_indices]  # Select the features used by the tree
            tree_predictions = tree.predict(X_selected_features)
            predictions.append(tree_predictions)
        
        # Return the average of predictions from all trees
        return np.mean(predictions, axis=0)

def test_against_sklearn():
    """Test implementation against sklearn's implementation"""
    from sklearn.tree import DecisionTreeRegressor as SklearnDTR
    from sklearn.ensemble import RandomForestRegressor as SklearnRF
    from sklearn.datasets import make_regression
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=4, random_state=42)
    print(y)
    
    # Test DecisionTreeRegressor
    custom_dt = DecisionTreeRegressor(max_depth=3)
    sklearn_dt = SklearnDTR(max_depth=3)
    
    
    custom_dt.fit(X, y)
    sklearn_dt.fit(X, y)
    
    custom_pred = custom_dt.predict(X)
    sklearn_pred = sklearn_dt.predict(X)

    # Plot for Decision Tree
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(sklearn_pred, custom_pred)
    plt.plot([min(sklearn_pred), max(sklearn_pred)], [min(sklearn_pred), max(sklearn_pred)], 'r--')
    plt.xlabel('Sklearn Decision Tree Predictions')
    plt.ylabel('Custom Decision Tree Predictions')
    plt.title('Decision Tree: Custom vs Sklearn Predictions')
    
    dt_mse = np.mean((custom_pred - sklearn_pred) ** 2)
    print(f"Decision Tree MSE between implementations: {dt_mse}")
    
    # Test RandomForestRegressor
    custom_rf = RandomForestRegressor(n_estimators=10, max_depth=3)
    sklearn_rf = SklearnRF(n_estimators=10, max_depth=3, random_state=42)
    
    custom_rf.fit(X, y)
    sklearn_rf.fit(X, y)
    
    custom_pred = custom_rf.predict(X)
    sklearn_pred = sklearn_rf.predict(X)
    
    rf_mse = np.mean((custom_pred - sklearn_pred) ** 2)
    print(f"Random Forest MSE between implementations: {rf_mse}")
    
    
    # Plot for Random Forest
    plt.subplot(1, 2, 2)
    plt.scatter(sklearn_pred, custom_pred)
    plt.plot([min(sklearn_pred), max(sklearn_pred)], [min(sklearn_pred), max(sklearn_pred)], 'r--')
    plt.xlabel('Sklearn Random Forest Predictions')
    plt.ylabel('Custom Random Forest Predictions')
    plt.title('Random Forest: Custom vs Sklearn Predictions')
    
    plt.tight_layout()
    plt.show()
    return custom_dt, sklearn_dt, custom_rf, sklearn_rf