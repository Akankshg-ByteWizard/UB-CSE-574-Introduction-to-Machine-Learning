# matplotlib inline
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

def create_cnn():
    class cnn(nn.Module):
        def __init__(self):
            super().__init__()

            # Convolutional Layer 1.
            filter_size1 = 5  # Convolution filters are 5 x 5 pixels.
            num_filters1 = 16  # There are 16 of these filters.

            # Convolutional Layer 2.
            filter_size2 = 5  # Convolution filters are 5 x 5 pixels.
            num_filters2 = 36  # There are 36 of these filters.

            # Fully-connected layer.
            fc_size = 128

            # Number of colour channels for the images: 1 channel for gray-scale.
            num_channels = 1

            # Number of classes, one class for each of 10 digits.
            num_classes = 10

            self.layer_conv1 = nn.Conv2d(num_channels, num_filters1, (filter_size1, filter_size1), padding='same')
            self.layer_conv2 = nn.Conv2d(num_filters1, num_filters2, (filter_size2, filter_size2), padding='same')
            self.pool = nn.MaxPool2d(2)
            self.layer_fc1 = nn.Linear(1764, fc_size)
            self.layer_fc2 = nn.Linear(fc_size, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.layer_conv1(x)))
            x = self.pool(F.relu(self.layer_conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.layer_fc1(x))
            x = self.layer_fc2(x)
            return x

    return cnn()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0:.0f}".format(cls_true[i])
        else:
            xlabel = "True: {0:.0f}, Pred: {1:.0f}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_confusion_matrix(cls_pred, cls_true):
    # This is called from test() below.

    # cls_pred is an list of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def test(dataloader, model, loss_fn, show_example_errors=False, show_confusion_matrix=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    # collecting misclassification examples in case of plotting
    mis_example, mis_example_pred, mis_example_true = [], [], []
    # collecting labels for computing confusion matrix
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X.float())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            mis_example.extend(X[(pred.argmax(1) == y) == False])
            mis_example_pred.extend(pred[(pred.argmax(1) == y) == False].argmax(1))
            mis_example_true.extend(y[(pred.argmax(1) == y) == False])

            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.argmax(1).cpu().numpy())


    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_images(images=mis_example[:9], cls_true=mis_example_true[:9], cls_pred=mis_example_pred[:9])

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=y_pred, cls_true=y_true)


def optimize(iterations, train_dataloader, model, cost, optimizer):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for _ in range(iterations):
        train(train_dataloader, model, cost, optimizer)

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

learning_rate = 1e-4
train_batch_size = 64
test_batch_size = 256

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Construct model
model = create_cnn().to(device)

# Define loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# load data
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
train_dataloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)


test(test_dataloader, model, cost)
optimize(1, train_dataloader, model, cost, optimizer)
test(test_dataloader, model, cost, True, True)

# We already performed 1 iteration above.
optimize(9, train_dataloader, model, cost, optimizer)
test(test_dataloader, model, cost, show_example_errors=False)

# optimize(900, train_dataloader, model, cost, optimizer)
# test(test_dataloader, model, cost, show_example_errors=False)

# optimize(9000, train_dataloader, model, cost, optimizer)
# test(test_dataloader, model, cost, show_example_errors=False)