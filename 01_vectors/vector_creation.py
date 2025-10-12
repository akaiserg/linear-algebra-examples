import numpy as np

v1 = np.array([1, 2]) # 2D vector
print(f"v1 = {v1}") # [1, 2]
v3 = np.zeros(3) # [0, 0, 0]
print(f"v3 = {v3}")
v4 = np.ones(3) # [1, 1, 1]
print(f"v4 = {v4}")
v5 = np.arange(1, 4) # [1, 2, 3]
print(f"v5 = {v5}")
v6 = np.linspace(0, 1, 3)
print(f"v6 = {v6}")
v7 = np.random.rand(3) # [0.1, 0.2, 0.3]
print(f"v7 = {v7}")
v8 = np.logspace(0, 2, 3)      # [1. 10. 100.] - logarithmic spacing
v9 = np.geomspace(1, 8, 4)     # [1. 2. 4. 8.] - geometric spacing
v10 = np.fromstring('1 2 3', sep=' ')  # [1. 2. 3.]
v11 = np.fromfunction(lambda i: i**2, (3,))  # [0. 1. 4.]
print(f"v10 = {v10}")
print(f"v11 = {v11}")


vr = np.random.rand(10)
for element in vr:
    print(element)

vr = np.append(vr, 10) # add 10 to the end of the vector

for element in vr:
    print(element)

print("---------------1-----------------")

vr = np.insert(vr, 0, 0) # add 0 to the beginning of the vector
for element in vr:
    print(element)

vr = np.delete(vr, 0) # delete the first element of the vector
for element in vr:
    print(element)

print("--------------2------------------")


vr = np.concatenate((vr, [10, 11, 12])) # join the vector with [10, 11, 12]
for element in vr:
    print(element)

print("---------------3-----------------")

vr = np.split(vr, 2) # split the vector into 2 parts
for element in vr:
    print(element)
