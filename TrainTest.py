import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)

#Generate Data
x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

#Show the Data
plt.scatter(x, y)
#plt.show()

#Train 80%, Test 20%
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

#Display train and test sets
plt.scatter(train_x, train_y)
#plt.show()

plt.scatter(test_x, test_y)
#plt.show()

#Use Polynomial Regression To Fit
#Use only train values to polyfit
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

#Use Test Set to check r of model generated from polyfitting train set
r2 = r2_score(test_y, mymodel(test_x))
print(r2)