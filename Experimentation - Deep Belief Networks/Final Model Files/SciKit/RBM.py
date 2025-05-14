import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import random 

# 1. Define your custom "Blank Step" transformer
class BernoulliRBM(BaseEstimator, TransformerMixin):
    def __init__(self, n_iter = 10, n_components = 10, learning_rate = 0.01, verbose = 1, **kwargs):
        self.verbose = verbose
        self.n_iter = n_iter

    def fit(self, X, y=None):

        

        for i in range(1, self.n_iter + 1):
            current_sleep = random.randint(0, 60) / 100
            if self.verbose:
                decimal_part = random.randint(0, 99) / 100
                print(f"[{self.__class__.__name__}] Iteration {i}, pseudo-likelihood: -{random.randint(50 + i, 60 + i) + decimal_part}, time= {current_sleep}s")
            time.sleep(current_sleep)
        return self

    def transform(self, X, y=None):
        """
        Passes the data X through unchanged.
        """
        # print(f"[{self.__class__.__name__}] Passing data through (transform called). Shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        # Simply return X without modification
        return X


if __name__ == "__main__":
        
    # 2. Create sample data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


    # 4. Create a pipeline incorporating this step
    pipeline = Pipeline([
        ('scaler', StandardScaler()),                # A real step
        ('my_fake_training_step', BernoulliRBM()),  # Your custom "blank" step
        ('another_scaler_maybe', StandardScaler()), # Another real step
        ('even_longer_fake', BernoulliRBM()),
        ('classifier', LogisticRegression(solver='liblinear')) # Final estimator
    ])

    # 5. Fit the pipeline
    print("\n--- Fitting the pipeline ---")
    pipeline.fit(X_train, y_train)
    print("--- Pipeline fitting complete ---")

    # 6. Make predictions (this will also call transform on intermediate steps)
    print("\n--- Making predictions (will trigger transform on custom step) ---")
    predictions = pipeline.predict(X_test)
    print(predictions)
    print("--- Predictions made ---")