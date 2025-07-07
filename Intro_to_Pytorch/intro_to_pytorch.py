import torch
import torch.nn as nn

# Tensors are Pytorch's core data structure and the foundation of deep learning. 
# They're similiar to NumPy arrays but have unique features.
temperatures = [[72, 75, 78], [70, 73, 76]]
temp_tensor = torch.tensor(temperatures)
print(temp_tensor)

# Checking and adding tensors
adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])
print("adjustment shape is: ", adjustment.shape)
print("adjustment type is: ", adjustment.dtype)
print("temperatures shape is: ", temp_tensor.shape)
print("temperatures type is: ", temp_tensor.dtype)
print(adjustment + temp_tensor)

# Create linear layer network that takes 3 features as input and returns 2 outputs
input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])
linear_layer = nn.Linear(
    in_features=3,
    out_features=2
)
output = linear_layer(input_tensor)
print(output)