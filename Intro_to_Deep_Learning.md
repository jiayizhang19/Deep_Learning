### What is a Neural Network   
A neural network consists of:  
- **Input Layer**: contains data set features
- **Hidden Layer**: a neural network can contain any numbers of hidden layers
- **Output Layer**: contains predictions
[Note: Every input feature is connected to every one of neurons in the hidden layer]


### Supervised Learning   
It is one type of machine learning, and basically it has some input x and wants it to learn a function mapping to some output y.  

#### Neural Network Architecture
- Feedforward Neural Networks (FNN) / Dense Neural Network / Standard Neural Network
    - A sequence of layers where each layer is fully connected to the next, and data flows in one direction, from input to output.
- Convolutional Neural Networks **(CNN)**
    - Process image data
- Recurrent Neural Networks **(RNN)**
    - Process one-dimensional sequence data, e.g. audio to text transcipt, and language translation
- Attention-Based Networks **(usually refers to Transformer)**
- Graph Neural Networks **(GNN)** 
- Hybrid Neural Networks 
    - Process image and radar info, e.g. autonomous driving  
![Neural Network Architecture](pics/NN%20Architectures.JPG)

#### Data Type
- Structured Data: tabular data with clear columns and rows
- Unstructured Data: audio, image, and individual text

### Trainable parameters:
- **Weights**: Each weight connects one neuron to another in the next layer. It determines how much the influence of a given input feature has on the output
- **Bias**: It allows the model to shift the activation function to better fit the data.


### Linear Layer Network
linear_layer = nn.Linear(
    in_features=,
    out_fearures=
)
output = linear_layer(input_tensor)


### Stacking hidden Layers and parameters
model = nn.Sequential(
    nn.Linear(input_feature_number, b),
    nn.Linear(b, c),
    nn.Linear(c, output_number)
)
output = model(input_tensor)