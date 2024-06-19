import torch

def group_quantize_round_to_nearest(x, bits=8, group_size=32):
    # assert group_size multiple of hidden dim
    x_shape = x.shape
    group_dim = x_shape[1]//group_size
    max_bin_value = 2**bits - 1
    
    x_reshaped = x.reshape(x_shape[0], group_dim, group_size)
    min_val, max_val = torch.aminmax(x_reshaped, dim=-1)
    zero_points = min_val
    scales = (max_val - min_val)/max_bin_value
    
    zero_points = zero_points.unsqueeze(-1)
    scales = scales.unsqueeze(-1)
    
    quantized_x = torch.round((x_reshaped - zero_points)/scales)
    quantized_x = quantized_x.to(dtype=torch.uint8)

    return quantized_x, zero_points, scales

def group_dequantize(x, scales, zero_points):
    to_reshape_x =  x * scales + zero_points
    return to_reshape_x
    #reshaped_x = to_reshape_x.reshape(x.shape[0], -1)
    #return reshaped_x