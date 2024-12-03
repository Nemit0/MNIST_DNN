import torch
import abc
import matplotlib.pyplot as plt
from torch import Tensor
from tqdm import tqdm
from typing import Dict, List

if torch.cuda.is_available():
    torch.set_default_device('cuda')
else:
    torch.set_default_device('cpu')

### Helper Classes and Functions ###
class OneHotEncoder:
    def __init__(self, num_classes: int, data: Tensor) -> None:
        self.num_classes: int = num_classes
        self.token_map: Dict[any, int] = {token: i for i, token in enumerate(data.unique())}
    
    def encode(self, data: Tensor) -> Tensor:
        encoded = torch.zeros(data.shape[0], self.num_classes)
        for i, token in enumerate(data):
            encoded[i][self.token_map[token]] = 1
        return encoded

### LOSS FUNCTIONS ###
class Loss(abc.ABC):
    @abc.abstractmethod
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass

class MeanSquaredError(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return torch.mean((y_pred - y_true) ** 2)

    def __str__(self) -> str:
        return "MeanSquaredError"

    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return 2 * (y_pred - y_true) / y_true.shape[0]

class CrossEntropyLoss(Loss):
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred_log_softmax = torch.log_softmax(y_pred, dim=1)
        # Compute the loss
        loss = -torch.mean(torch.sum(y_true * y_pred_log_softmax, dim=1))
        return loss

    def __str__(self) -> str:
        return "CrossEntropyLoss"

    def gradient(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Compute softmax probabilities
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        # Compute the gradient
        grad = (y_pred_softmax - y_true) / y_true.shape[0]
        return grad

LOSS_FUNCTIONS = {
    "MeanSquaredError": MeanSquaredError(),
    "CrossEntropyLoss": CrossEntropyLoss()
}

### ACTIVATION FUNCTIONS ###
class Activation(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def gradient(self, x: Tensor) -> Tensor:
        pass

class Tanh(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    def __str__(self) -> str:
        return "Tanh"

    def gradient(self, x: Tensor) -> Tensor:
        y = self.__call__(x)
        return 1 - y ** 2

class Sigmoid(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))

    def __str__(self) -> str:
        return "Sigmoid"

    def gradient(self, x: Tensor) -> Tensor:
        y = self.__call__(x)
        return y * (1 - y)

class ReLU(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return torch.max(x, torch.zeros_like(x))

    def __str__(self) -> str:
        return "ReLU"

    def gradient(self, x: Tensor) -> Tensor:
        return (x > 0).float()

class Softmax(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        exps = torch.exp(x - torch.max(x, axis=1, keepdim=True).values)
        return exps / torch.sum(exps, axis=1, keepdim=True)

    def __str__(self) -> str:
        return "Softmax"

    def gradient(self, x: Tensor) -> Tensor:
        return self.__call__(x) * (1 - self.__call__(x))

class Linear(Activation):
    def __call__(self, x: Tensor) -> Tensor:
        return x

    def __str__(self) -> str:
        return "Linear"

    def gradient(self, x: Tensor) -> Tensor:
        return torch.ones_like(x)

ACTIVATION_FUNCTIONS = {
    "Tanh": Tanh(),
    "Sigmoid": Sigmoid(),
    "ReLU": ReLU(),
    "Softmax": Softmax(),
    "Linear": Linear()
}

### OPTIMIZERS ###
class Optimizer(abc.ABC):
    @abc.abstractmethod
    def step(self, layers: list) -> None:
        pass

class SimpleGradientDescent(Optimizer):
    def __init__(self, lr: float):
        self.lr: float = lr

    def step(self, layers: list) -> None:
        for layer in layers:
            layer.weights = layer.weights - self.lr * layer.grad_weights
            layer.biases = layer.biases - self.lr * layer.grad_biases
    
    def __str__(self) -> str:
        return "SimpleGradientDescent"

class MomentumGradientDescent(Optimizer):
    def __init__(self, lr: float, momentum: float):
        self.lr = lr
        self.momentum = momentum
        self.velocities = []

    def step(self, layers: List) -> None:
        # Initialize velocities for each layer if not already done
        if not self.velocities:
            self.velocities = [{"weights": torch.zeros_like(layer.weights),
                                "biases": torch.zeros_like(layer.biases)}
                               for layer in layers]
        
        for i, layer in enumerate(layers):
            self.velocities[i]["weights"] = self.momentum * self.velocities[i]["weights"] - self.lr * layer.grad_weights
            self.velocities[i]["biases"] = self.momentum * self.velocities[i]["biases"] - self.lr * layer.grad_biases

            layer.weights += self.velocities[i]["weights"]
            layer.biases += self.velocities[i]["biases"]
    
    def __str__(self) -> str:
        return "MomentumGradientDescent"

class AdamOptimizer(Optimizer):
    def __init__(self, lr: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []  # First moment (mean of gradients)
        self.v = []  # Second moment (variance of gradients)

    def step(self, layers: List) -> None:
        self.t += 1  # Increment time step

        if not self.m:
            # Initialize first and second moments
            self.m = [{"weights": torch.zeros_like(layer.weights),
                       "biases": torch.zeros_like(layer.biases)}
                      for layer in layers]
            self.v = [{"weights": torch.zeros_like(layer.weights),
                       "biases": torch.zeros_like(layer.biases)}
                      for layer in layers]
        
        for i, layer in enumerate(layers):
            # Update biased first moment estimate
            self.m[i]["weights"] = self.beta1 * self.m[i]["weights"] + (1 - self.beta1) * layer.grad_weights
            self.m[i]["biases"] = self.beta1 * self.m[i]["biases"] + (1 - self.beta1) * layer.grad_biases

            # Update biased second raw moment estimate
            self.v[i]["weights"] = self.beta2 * self.v[i]["weights"] + (1 - self.beta2) * layer.grad_weights**2
            self.v[i]["biases"] = self.beta2 * self.v[i]["biases"] + (1 - self.beta2) * layer.grad_biases**2

            # Correct bias in first moment
            m_hat_weights = self.m[i]["weights"] / (1 - self.beta1**self.t)
            m_hat_biases = self.m[i]["biases"] / (1 - self.beta1**self.t)

            # Correct bias in second moment
            v_hat_weights = self.v[i]["weights"] / (1 - self.beta2**self.t)
            v_hat_biases = self.v[i]["biases"] / (1 - self.beta2**self.t)

            # Update parameters
            layer.weights -= self.lr * m_hat_weights / (torch.sqrt(v_hat_weights) + self.epsilon)
            layer.biases -= self.lr * m_hat_biases / (torch.sqrt(v_hat_biases) + self.epsilon)
    
    def __str__(self) -> str:
        return "AdamOptimizer"

OPTIMIZERS = {
    "SimpleGradientDescent": SimpleGradientDescent,
    "MomentumGradientDescent": MomentumGradientDescent,
    "AdamOptimizer": AdamOptimizer
}

### LAYER AND NETWORK ###
class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = torch.randn(input_size, output_size, dtype=torch.float64, requires_grad=False)
        self.biases = torch.zeros(output_size, dtype=torch.float64, requires_grad=False)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.z = torch.matmul(inputs, self.weights) + self.biases
        self.output = self.activation(self.z)
        return self.output

    def backward(self, grad_output: Tensor) -> Tensor:
        grad_activation = self.activation.gradient(self.z)
        grad = grad_output * grad_activation
        self.grad_weights = torch.matmul(self.inputs.T, grad)
        self.grad_biases = torch.sum(grad, axis=0)
        grad_input = torch.matmul(grad, self.weights.T)
        return grad_input

class NeuralNetwork:
    def __init__(self, layer_sizes: list, activation: Activation, loss: Loss, optimizer: Optimizer) -> None:
        self.layer_sizes = layer_sizes
        self.layers: List[Layer] = []
        for i in range(len(layer_sizes) - 1):
            act = activation if i < len(layer_sizes) - 2 else Linear()
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], act))
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, inputs: Tensor, targets: Tensor, epochs: int, batch_size: int = 32) -> List[float]:
        loss_history = []
        accuracy_history = []
        for epoch in tqdm(range(epochs)):
            loss_sum = 0
            permutation = torch.randperm(inputs.size(0))
            for i in range(0, inputs.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_inputs = inputs[indices]
                batch_targets = targets[indices]

                outputs = self.forward(batch_inputs)
                loss = self.loss(outputs, batch_targets)
                loss_sum += loss.item()
                accuracy = torch.sum(torch.argmax(outputs, axis=1) == torch.argmax(batch_targets, axis=1)).item() / batch_size
                accuracy_history.append(accuracy)
                grad = self.loss.gradient(outputs, batch_targets)

                self.backward(grad)
                self.optimizer.step(self.layers)

            avg_loss = loss_sum / (inputs.size(0) // batch_size)
            loss_history.append(avg_loss)
            # Omit automatic grad calculation for accuracy calculation
            with torch.no_grad():
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(targets, 1)
                correct = (predicted == labels).sum().item()
                accuracy = 100 * correct / inputs.size(0)
                accuracy_history.append(accuracy)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return loss_history, accuracy_history
    
    def evaluate(self, inputs: Tensor, targets: Tensor) -> float:
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(targets, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / inputs.size(0)
        return accuracy

    def predict(self, input: Tensor):
        output = self.forward(input)
        result = torch.argmax(output, axis=1)
        return result
    
    def save_model(self, path: str) -> None:
        torch.save({
            "layer_sizes": self.layer_sizes,
            "activation": self.layers[0].activation.__str__(),
            "loss": self.loss.__str__(),
            "optimizer": self.optimizer.__str__(),
            "learning_rate": self.optimizer.lr,
            "layers": [
                {
                    "weights": layer.weights,
                    "biases": layer.biases,
                    "activation": layer.activation.__str__(),
                } for layer in self.layers
            ]
        }, path)

    @staticmethod
    def load_model(path: str) -> 'NeuralNetwork':
        model = torch.load(path, weights_only=True)
        layer_sizes = model["layer_sizes"]
        activation = ACTIVATION_FUNCTIONS[model["activation"]]
        loss = LOSS_FUNCTIONS[model["loss"]]
        optimizer = OPTIMIZERS[model["optimizer"]](model["learning_rate"])
        network = NeuralNetwork(layer_sizes, activation, loss, optimizer)

        for layer, model_layer in zip(network.layers, model["layers"]):
            layer.weights = model_layer["weights"]
            layer.biases = model_layer["biases"]
            layer.activation = ACTIVATION_FUNCTIONS[model_layer["activation"]]

        return network

    def __str__(self):
        return f"NeuralNetwork"