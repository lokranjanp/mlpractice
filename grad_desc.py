import numpy as np
import matplotlib as plt
import math, copy

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

def compute_cost(x, y, w, b):
    sum_cost = 0
    m = x.shape[0]
    for i in range(m):
        y_cap = w*x[i]+b
        cost = (y_cap - y[i])**2
        sum_cost = sum_cost + cost

    total_cost = (1/(2*m))*sum_cost
    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    total_w = 0
    total_b = 0

    for i in range(m):
        f_wb = w*x[i]+b
        cost_w = (f_wb-y[i])*x[i]
        cost_b = (f_wb-y[i])
        total_b = total_b + cost_b
        total_w = total_w + cost_w

    total_b = (1/m)*total_b
    total_w = (1/m)*total_w

    return total_b,total_w

def gradient_descent(x, y, w_in, b_in, alpha, nums_iter, cost_func, grad_func):
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(nums_iter):
        dj_db, dj_dw = grad_func(x, y, w, b)
        b = b - alpha*dj_db
        w = w - alpha*dj_dw

        if i<10000:
            J_history.append(cost_func(x, y, w, b))
            p_history.append([w, b])

        if i%math.ceil(nums_iter/10) == 0:
            print(f"Iteration {i:4}:  Cost {J_history[-1]:0.2e}",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db:0.3e}",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


w_init = 0
b_init = 0

iters = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iters,
                                                   compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent : ({w_final:8.4f},{b_final:8.4f})")

