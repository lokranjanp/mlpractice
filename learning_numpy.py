import numpy as np
import time

# allocates memory to the numpy array "a" and fills the values as zero.
a = np.zeros(4)
print(f"np.zeros(4) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# to fill in random values into the numpy array :
b = np.random.random_sample(5)
print(f"b = {b}")

# to create a numpy array without specifying its shape/size

c = np.arange(5)        #   fills the numpy array with values up to the specified argument
print(f"c = {c}")

# to fill in random values
d = np.random.rand(5)
print(f"d = {d}")

# creating numpy array with user specified values
e = np.array([1, 2, 3, 4])


# slicing of numpy arrays

f = np.arange(10)
# slicing arguments : (start : stop : step)
print(f[2: 4: 1])

# access elements from index 2 to index 5 separated by 2
print(f[2: 5: 2])

# access elements from index 3 and above
print(f[3:])

# access elements below 3
print(f[:3])

# to print elements in reverse order
print(f"printing in reverse order : {f[::-1]}")

# single vector operations are : sum, mean, squaring etc

#   with vectorisation

def my_dot(x, y):
    m = x.shape[0]
    f = 0

    for i in range(0,m):
        f = f + x[i]*y[i]

    return f

x = np.random.rand(100000)
y = np.random.rand(100000)

tic = time.time()
print(f"with vectorisation : {np.dot(x,y)}")
toc = time.time()

print(f"time taken : {1000*(toc - tic):.4f} ms")


#   without vectorisation
tic = time.time()
t = my_dot(x,y)
toc = time.time()

print(f"without vectorisation : {t}")
print(f"time taken : {1000*(toc - tic):.4f} ms")

