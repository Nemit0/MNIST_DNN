import torch
import struct
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, tensor

def view_image(image: Tensor, filename:str) -> None:
    if image.device.type == 'cuda':
        image = image.cpu()
    
    plt.imshow(image.view(28, 28), cmap='gray')
    plt.show()
    plt.savefig(f'{filename}.png')

### Helper Functions to Load MNIST ###
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

def get_fashion_lable(index:int):
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return labels[index]