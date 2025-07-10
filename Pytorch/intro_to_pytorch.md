#### A neural network consists of: 
- **Input Layer**: contains data set features
- **Hidden Layer**: a neural network can contain any numbers of hidden layers
- **Output Layer**: contains predictions


#### Trainable parameters:
- **Weights**: Each weight connects one neuron to another in the next layer. It determines how much the influence of a given input feature has on the output
- **Bias**: It allows the model to shift the activation function to better fit the data.

#### Linear Layer Network
linear_layer = nn.Linear(
    in_features=,
    out_fearures=
)
output = linear_layer(input_tensor)

#### Stacking hidden Layers and parameters
model = nn.Sequential(
    nn.Linear(input_feature_number, b),
    nn.Linear(b, c),
    nn.Linear(c, output_number)
)
output = model(input_tensor)