import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Función de activación (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación (sigmoid)
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de la red neuronal
def init_network(layers):
    network = []
    for i in range(1, len(layers)):
        layer = {
            'weights': np.random.rand(layers[i-1], layers[i]),
            'bias': np.random.rand(1, layers[i])
        }
        network.append(layer)
    return network

# Propagación hacia adelante
def forward_propagation(network, inputs):
    layer_output = inputs
    for layer in network:
        layer['output'] = sigmoid(np.dot(layer_output, layer['weights']) + layer['bias'])
        layer_output = layer['output']
    return layer_output

# Retropropagación
def backward_propagation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        if i == len(network) - 1:
            layer['error'] = expected - layer['output']
            layer['delta'] = layer['error'] * sigmoid_derivative(layer['output'])
        else:
            next_layer = network[i + 1]
            layer['error'] = np.dot(next_layer['weights'], next_layer['delta'].T).T
            layer['delta'] = layer['error'] * sigmoid_derivative(layer['output'])

# Actualización de pesos y sesgos
def update_weights(network, inputs, learning_rate):
    for i in range(len(network)):
        layer_input = inputs if i == 0 else network[i - 1]['output']
        network[i]['weights'] += learning_rate * np.dot(layer_input.T, network[i]['delta'])
        network[i]['bias'] += learning_rate * network[i]['delta']

# Entrenamiento de la red neuronal
def train_network(network, inputs, expected_outputs, learning_rate, epochs):
    for _ in range(epochs):
        for i in range(len(inputs)):
            inputs_sample = np.array(inputs[i], ndmin=2)
            expected_output = np.array(expected_outputs[i], ndmin=2)
            outputs = forward_propagation(network, inputs_sample)
            backward_propagation(network, expected_output)
            update_weights(network, inputs_sample, learning_rate)

# Ejemplo de uso
if __name__ == "__main__":
    # Definir la estructura de la red neuronal
    layers = [2, 4, 1]  # Dos neuronas en la capa de entrada, cuatro en la capa oculta y una en la capa de salida
    dataset = pd.read_csv("concentlite.csv")
    # Inicializar la red neuronal
    network = init_network(layers)

    # Datos de entrada
    inputs = dataset.iloc[:, :2].values

    # Salidas esperadas
    expected_outputs = dataset.iloc[:, 2].values.reshape(-1, 1)

    # Entrenar la red neuronal
    learning_rate = 0.5
    epochs = 300
    train_network(network, inputs, expected_outputs, learning_rate, epochs)

    # Probar la red neuronal
    for i in range(len(inputs)):
        outputs = forward_propagation(network, np.array(inputs[i], ndmin=2))
        print(f"Input: {inputs[i]}, Output: {outputs}")

    # Probar la red neuronal
    outputs = []
    for i in range(len(inputs)):
        output = forward_propagation(network, np.array(inputs[i], ndmin=2))
        print(f"Input: {inputs[i]}, Output: {output}")
        outputs.append(output)

    outputs = np.array(outputs).flatten()

    # Graficar los resultados
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, cmap='coolwarm')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Neural Network Output')
    plt.colorbar(label='Output')
    plt.show()