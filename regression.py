import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    """It calculates the total cost of a given function for given w and b values
    :rtype: total cost and a numpy array with predicted values for given w and b
    """
    sum_cost = 0
    m = x.shape[0]
    predict = np.zeros(m)
    for i in range(m):
        y_cap = w*x[i] + b
        predict[i] = y_cap
        diff = (y_cap-y[i])**2
        sum_cost = sum_cost + diff
    total_cost = (1/(2*m))*sum_cost
    return total_cost, predict


x_train = np.array([1, 2, 3])
y_train = np.array([110, 230, 310])

current_cost, f_wb = compute_cost(x_train, y_train, 100, 10)
print(current_cost)

plt.plot(x_train, y_train)
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train, f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
