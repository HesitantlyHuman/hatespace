import numpy as np

def interpolate_from_fractionals(input_size, output_size, fraction_list):
    interpolated_list = [int(x) for x in output_size + fraction_list * (input_size - output_size)]
    interpolated_list[0], interpolated_list[-1] = input_size, output_size
    return interpolated_list

def squircle_interpolation(n_samples, power = 1.0):
    #Could change to be symmetric with special case for power < 1.0 : (-(1 - abs(x - 1) ** a) ** 1 / a) + 1
    x = np.linspace(0, 1, n_samples + 1)
    return np.power(1 - np.power(x, power), 1 / power)