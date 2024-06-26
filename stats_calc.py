import numpy as np

#######################
#######################
#######################

JF_vals = np.array(
    [0.700, 0.706, 0.708, 0.708, 0.715, 0.720, 0.726, 0.730, 0.725, 0.724]
)
J_vals = np.array(
    [0.666, 0.668, 0.660, 0.649, 0.661, 0.656, 0.655, 0.647, 0.654, 0.635]
)


#######################
#######################
#######################

J_mean = J_vals.mean()
J_std = J_vals.std()

JF_mean = JF_vals.mean()
JF_std = JF_vals.std()


print("J")
print(f"J mean: {np.round(J_mean, 4)}")
print(f"J std: {np.round(J_std,4)}")
print("*" * 10)
print("J&F")
print(f"J&F mean: {np.round(JF_mean, 4)}")
print(f"J&F std: {np.round(JF_std, 4)}")
