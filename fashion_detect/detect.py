import torch
from main import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import cv2

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

image = cv2.imread("target.webp")
if image is None:
    print("Image not found!")
    exit()

image_grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
image_resized = cv2.resize(image_grey, (28, 28))
image_inverted = 255 - image_resized
image_resized = image_inverted.copy()


scale_factor = 10  # You can adjust this value to make the image preview bigger
image_preview = cv2.resize(image_resized,
                           (image_resized.shape[1] * scale_factor, image_resized.shape[0] * scale_factor))

cv2.imshow("Resized Grayscale Image", image_preview)

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model3.pth", weights_only=True, map_location=torch.device('cpu')))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert to PIL image first
    transforms.Grayscale(num_output_channels=1),  # Ensure it's single-channel grayscale
    transforms.ToTensor(),              # Convert to tensor, and automatically scales it to [0, 1]
])

tensor_image = transform(image_resized).unsqueeze(0)

print(tensor_image.size())  # Should print torch.Size([1, 1, 28, 28])
print(model)

with torch.no_grad():
    pred = model(tensor_image)
    predicted = classes[pred[0].argmax(0)]
    print(f'Predicted: "{predicted}"')


while True:
    # Wait for the user to press the 'ESC' key to close the window
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of the ESC key
        break

cv2.destroyAllWindows()
