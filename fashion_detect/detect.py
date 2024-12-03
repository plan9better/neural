import torch
from main import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

import cv2

image = cv2.imread("sweater2.webp")
if image is None:
    print("Image not found!")
    exit()

image_grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
image_resized = cv2.resize(image_grey, (28, 28))
image_inverted = 255 - image_resized
image_resized = image_inverted.copy()


# Scaling factor for enlarging the preview
scale_factor = 100  # You can adjust this value to make the image preview bigger

# Resize the image for preview purposes (keeping the original 28x28 image unchanged)
image_preview = cv2.resize(image_resized,
                           (image_resized.shape[1] * scale_factor, image_resized.shape[0] * scale_factor))

cv2.imshow("Resized Grayscale Image", image_preview)

model = NeuralNetwork().to("cpu")
model.load_state_dict(torch.load("model.pth", weights_only=True))

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

# Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )
# # model.eval()
# x, y = test_data[0][0], test_data[0][1]
transform = transforms.Compose([
    transforms.ToPILImage(),            # Convert to PIL image first
    transforms.Grayscale(num_output_channels=1),  # Ensure it's single-channel grayscale
    transforms.ToTensor(),              # Convert to tensor, and automatically scales it to [0, 1]
])

# Apply the transformations
tensor_image = transform(image_resized)

# Add a batch dimension (make it [1, 1, 28, 28] for a single grayscale image)
# tensor_image = tensor_image.unsqueeze(0)

print(tensor_image.size())  # Should print torch.Size([1, 1, 28, 28])

# images = [image_resized]

# print(images, x.shape)
with torch.no_grad():
    # x = x.to("cpu")
    pred = model(tensor_image)
    predicted = classes[pred[0].argmax(0)]
    print(f'Predicted: "{predicted}"')


while True:
    # Wait for the user to press the 'ESC' key to close the window
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of the ESC key
        break

cv2.destroyAllWindows()
