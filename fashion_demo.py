from src.nn_objects import *
from src.utils import *

def main():
    model = NeuralNetwork.load_model("./models/mnist_fashion_model.pth")

    # Load the MNIST data
    train_images = load_mnist_images('./data/fashion/train-images-idx3-ubyte')
    train_labels = load_mnist_labels('./data/fashion/train-labels-idx1-ubyte')
    test_images = load_mnist_images('./data/fashion/t10k-images-idx3-ubyte')
    test_labels = load_mnist_labels('./data/fashion/t10k-labels-idx1-ubyte')

    # Normalize images to [0,1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert to torch tensors
    train_x = torch.tensor(train_images, dtype=torch.float64)
    train_y = torch.tensor(train_labels, dtype=torch.int64)
    test_x = torch.tensor(test_images, dtype=torch.float64)
    test_y = torch.tensor(test_labels, dtype=torch.int64)

    # One-hot encode labels
    train_targets_tensor = torch.zeros(train_y.shape[0], 10)
    train_targets_tensor[torch.arange(train_y.shape[0]), train_y] = 1

    test_targets_tensor = torch.zeros(test_y.shape[0], 10)
    test_targets_tensor[torch.arange(test_y.shape[0]), test_y] = 1

    # Evaluate the model
    accuracy = model.evaluate(test_x, test_targets_tensor.float())
    print(f"Test Accuracy: {accuracy}%")

    # Predict a single image
    train_y, test_y = train_targets_tensor, test_targets_tensor

    target_image = test_x[0]
    view_image(target_image, 'fashion_image')

    target_image = target_image.unsqueeze(0)
    prediction = get_fashion_lable(model.predict(target_image).item())
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()