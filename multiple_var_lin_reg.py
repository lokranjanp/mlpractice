import numpy as np
import matplotlib as plt
import copy, math


# training datasets :

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# number of features : n = 4
print(f"x_train shape : {x_train.shape}, y_train shape : {y_train.shape}")
print(f"x_train Type : {type(x_train)}, y_train Type : {type(y_train)}")

w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083

x_vec = x_train[0, :]

def predict(x, w, b):
    """predicts the output for a given input using vectorisation"""
    p = np.dot(w, x)

    return p+b

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0

    for i in range(m):
        y_cap_i = np.dot(w,x[i]) +b
        cost = cost + (y_cap_i - y[i])**2

    cost = cost / (2 * m)
    return cost

cost = compute_cost(x_train,y_train, w_init, b_init)
print(cost)

def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_db = 0.
    dj_dw = np.zeros(n)

    for i in range(m):
        err = (np.dot(x[i], w)+b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*x[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, num_iter, cost_func, grad_func, alpha):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iter):
        dj_dw, dj_db = grad_func(x, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_func(x, y, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history

init_w = np.zeros_like(w_init)
init_b = 0.

iters = 100000
alpha = 5.0e-7
w_final, b_final, J_hist = gradient_descent(x_train, y_train, init_w, init_b, iters, compute_cost,
                                            compute_gradient, alpha)

m,_ = x_train.shape

print(f"final b : {b_final}, final w : {w_final}")

for i in range(m):
    print(f"prediction : {np.dot(x_train[i], w_final)+b_final:0.2f}, target value : {y_train[i]}")


