import numpy as np

def generate_layer_list(input_size, output_size, num_layers, bias = 1.0):
    return output_size - np.power(np.linspace(1, 0, num_layers), bias) * (output_size - input_size)