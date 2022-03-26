import matplotlib.pyplot as plt
from scipy import stats

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

test = stats.linregress(x,y)

print(test)

def retrievedFunc(x):
    slope = test.slope
    intercept = test.intercept

    return x * slope + intercept 

values = list(map(retrievedFunc, x))
a = retrievedFunc(10)
print(a)

plt.plot(x, values)
plt.show()