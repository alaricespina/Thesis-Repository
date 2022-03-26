import numpy
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
from sklearn import linear_model
from random import randint 



# Generate Data
# Body language score = postural angle
# Facial expression score = eye condition
samples =  100
max_value = 100
min_value = 1
body_language_score = [randint(min_value, max_value) for i in range(samples)]
facial_expression_score = [randint(min_value, max_value) for i in range(samples)]

# Bias Weights
body_language_weight = randint(1, 10)
facial_expression_weight = randint(1, 10)


# Temporary function for overall health based on body language and facial expression
def health_score(body_language, facial_expression):
    result = body_language_weight * body_language
    result += facial_expression_weight + facial_expression
    return result 

# Store all valeus for overall health
overall_health = []
for i in range(samples):
    health = health_score(body_language_score[i],facial_expression_score[i])
    overall_health.append(health)

# Plot the points in the graph
plt.scatter(body_language_score, overall_health)
plt.scatter(facial_expression_score, overall_health)

# Print Metrics
metrics = f"Body Language Array:\n{body_language_score}\nBody Language Weight: {body_language_weight}\n\n"
metrics += f"Facial Expression Array:\n{facial_expression_score}\nFacial Expression Weight {facial_expression_weight}"
print(f"Metrics:\n\n{metrics}\n")

# Create Linear Model
independent = []

for i in range(samples):
    independent.append((body_language_score[i], facial_expression_score[i]))

dependent = overall_health
regression_model = linear_model.LinearRegression()
regression_model.fit(independent, dependent)

print(f"Regression Coefficient: {regression_model.coef_}")
print(f"Regression Intercepts: {regression_model.intercept_}")

# Show the points plotted by the body language and facial expression scores
plt.show()
