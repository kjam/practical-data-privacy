def laplace_dp_mechanism(value, epsilon, sensitivity=1):
    # Please do not use this function, ever :)
    orig_value = value
    value =  np.random.laplace(value, sensitivity/epsilon) # now you see why the 1 was a poor choice!
    print("Noise: {}".format(value - orig_value))
    return value
