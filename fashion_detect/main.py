import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#)
device = "cpu"
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_chans = 32
        self.out_chans2 = 64
        self.flatten = nn.Flatten()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.out_chans, (3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_chans, self.out_chans2, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
         )
    
        with torch.no_grad():
            sample_input = torch.rand(1, 1, 28, 28)  # Batch size of 1
            sample_output = self.cnn(sample_input)
            flattened_size = sample_output.view(1, -1).shape[1]
    
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(flattened_size, 256),  # Use computed size
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def training():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model3.pth")
    print("Saved PyTorch Model State to model.pth")

if __name__ == "__main__":
    
    main()
