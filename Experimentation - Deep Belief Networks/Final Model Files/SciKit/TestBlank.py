import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Define your custom "Blank Step" transformer
class FakeTrainingPassthrough(BaseEstimator, TransformerMixin):
    def __init__(self, steps_to_print=3, delay=0.1):
        """
        A transformer that simulates training steps during fit and passes data through.
        Args:
            steps_to_print (int): How many "steps" to print during fake training.
            delay (float): Seconds to pause between printed steps.
        """
        self.steps_to_print = steps_to_print
        self.delay = delay

    def fit(self, X, y=None):
        """
        Simulates a fitting process by printing steps.
        Does not actually learn anything from X or y.
        """
        print(f"[{self.__class__.__name__}] Starting fake training...")
        for i in range(1, self.steps_to_print + 1):
            print(f"[{self.__class__.__name__}] Fake training step: {i}")
            # Simulate some work being done
            time.sleep(self.delay)
        print(f"[{self.__class__.__name__}] Fake training complete.")
        # The fit method must return self
        return self

    def transform(self, X, y=None):
        """
        Passes the data X through unchanged.
        """
        print(f"[{self.__class__.__name__}] Passing data through (transform called). Shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        # Simply return X without modification
        return X

# 2. Create sample data
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)




# 4. Create a pipeline incorporating this step
pipeline = Pipeline([
    ('scaler', StandardScaler()),                # A real step
    ('my_fake_training_step', FakeTrainingPassthrough(steps_to_print=3, delay=0.2)),  # Your custom "blank" step
    ('another_scaler_maybe', StandardScaler()), # Another real step
    ('even_longer_fake', FakeTrainingPassthrough(steps_to_print=3, delay=0.2)),
    ('classifier', LogisticRegression(solver='liblinear')) # Final estimator
])

# 5. Fit the pipeline
print("\n--- Fitting the pipeline ---")
pipeline.fit(X_train, y_train)
print("--- Pipeline fitting complete ---")

# 6. Make predictions (this will also call transform on intermediate steps)
print("\n--- Making predictions (will trigger transform on custom step) ---")
predictions = pipeline.predict(X_test)
print("--- Predictions made ---")

# 7. Evaluate
score = pipeline.score(X_test, y_test)
print(f"\nPipeline score: {score}")

print("\nPipeline steps:")
for name, step_instance in pipeline.steps:
    print(f"- {name}: {step_instance}")