from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE  # pip install imbalanced-learn
from sklearn.metrics import classification_report

# 1. Generate imbalanced synthetic data
X, y = make_classification(n_samples=1000, n_features=20,
                           n_informative=10, n_redundant=2,
                           n_classes=4, weights=[0.7, 0.2, 0.05, 0.05],  # Imbalanced classes
                           random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Oversample the minority classes (using SMOTE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. DBN (RBM)
rbm = BernoulliRBM(n_components=10, learning_rate=0.06, n_iter=10, random_state=42)
rbm.fit(X_train_resampled)  # Fit on the *resampled* training data
X_train_transformed = rbm.transform(X_train_resampled)
X_test_transformed = rbm.transform(X_test)

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # Use class_weight
rf.fit(X_train_transformed, y_train_resampled)

# 5. Evaluation
y_pred = rf.predict(X_test_transformed)
print(classification_report(y_test, y_pred))