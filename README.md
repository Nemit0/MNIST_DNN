# MNIST_DNN
DNN trained on MNIST datasets(handwriting, fashion)

## Requirements
This uses pytorch to accelerate calculations, but a cuda enviromnent isn't necessary.

### Enviromnent
The project assumes using python3.10+ and virtualenv & pip for enviromnent setting. To make a venv and install the requirements, run the following commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset
Dataset are from Kaggle MNIST datasets, which are handwriting and fashion datasets.

- [Fashion-mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
-  [Handwriting-mnist](https://www.kaggle.com/c/digit-recognizer/data)

## Training
The main Neural network class and other utils/helpers are stored within /src folder, whilst training and demo are on the root.

The Handwriting is trained with network layer [768, 512, 128, 64, 10] with activation Sigmoid and loss Crossentropy.
It's trained with 500 epoch with learning rate 0.00005 with ADAM optimizer, with batch size 128.

Fashion share same layer configuration with handwriting, but it's trained with lr of 0.0001, 1000 epoch and batch size 256.

Each of training took respectively about 5, 10 for each dataset and training configuration, on a rtx 3080. It might take longer on a cpu.

## Evaluation
The trained models are stored within mnist_fashioni_model.pth, and can be loaded via load_model method on NeuralNetwork class.

Run the demo.py files to see the evaluation and sample prediction for a single image.