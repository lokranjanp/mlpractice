import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_house_data():
    data = np.loadtxt("houses.txt", delimiter=',', skiprows=1)
    x = data[:, :4]
    y = data[:, 4]
    return x, y

x_train, y_train = load_house_data()
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
print(x_train.shape)
print()
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)

print(f"Peak to peak range by column in raw data : {np.ptp(x_train,axis = 0)}")
print(f"Peak to peak range by column in normalised data : {np.ptp(x_norm, axis = 0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(x_norm, y_train)

def custom_repr(model):
    params = ', '.join(f"{param}={value}" for param, value in model.get_params().items())
    return f"{model.__class__.__name__}({params})"

print(custom_repr(sgdr))

print(f"number of iterations completed : {sgdr.n_iter_}, number of weight updates : {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_

print(f"Model parameters : w : {w_norm}, b : {b_norm}")

y_pred_sgd = sgdr.predict(x_norm)
y_pred = np.dot(x_norm, w_norm)+b_norm

print(f"prediction using np.dot and sgdr.predict match : {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set : \n{y_pred[:4]}")
print(f"Target values : \n{y_train[:4]}")

