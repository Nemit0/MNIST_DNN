from src.nn_objects import *
from src.utils import *

def main():
    # Load the MNIST data
    train_images = load_mnist_images('./data/fashion/train-images-idx3-ubyte')
    train_labels = load_mnist_labels('./data/fashion/train-labels-idx1-ubyte')
    test_images = load_mnist_images('./data/fashion/t10k-images-idx3-ubyte')
    test_labels = load_mnist_labels('./data/fashion/t10k-labels-idx1-ubyte')

    # Normalize images to [0,1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert to torch tensors
    train_x = torch.tensor(train_images, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.int64)
    test_x = torch.tensor(test_images, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.int64)

    # One-hot encode labels
    train_targets_tensor = torch.zeros(train_y.shape[0], 10)
    train_targets_tensor[torch.arange(train_y.shape[0]), train_y] = 1

    test_targets_tensor = torch.zeros(test_y.shape[0], 10)
    test_targets_tensor[torch.arange(test_y.shape[0]), test_y] = 1

    train_y, test_y = train_targets_tensor, test_targets_tensor

    print(train_x.shape, test_x.shape)
    print(train_y.shape, test_y.shape)

    # Define the neural network with adjusted input and output sizes
    nn = NeuralNetwork([784, 512, 128, 64, 10], Sigmoid(), CrossEntropyLoss(), AdamOptimizer(0.0001))

    # This training takes 10 minutes on RTX 3080.

    # Train the network (you may adjust epochs and batch_size as needed)
    loss, accuracy = nn.train(train_x, train_y.float(), epochs=1000, batch_size=256)
    print(f"Final Training Accuracy: {accuracy[-1]}%")

    # Plot the loss
    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_mnist_fashion.png')

    # Evaluate the model on the test set
    test_accuracy = nn.evaluate(test_x, test_y.float())
    print(f"Test Accuracy: {test_accuracy}%")

    # Save the model
    nn.save_model("./models/mnist_fashion_model.pth")

if __name__ == "__main__":
    main()