import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# ResNet Model Definition
class ResNet(nn.Module):
    def __init__(self, num_classes):  # Add num_classes as a parameter
        super(ResNet, self).__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

# Assuming you know the number of classes
num_classes = 6  # Update this with the actual number of classes

# If loading a model that was saved with its architecture
model = ResNet(num_classes=num_classes)
model.load_state_dict(torch.load('./features/trash_classifier_model.pt', map_location=torch.device('cpu')))
model.eval()

# Transformations
transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Prediction Function for an Image Tensor
def predict_image(img, model):
    with torch.no_grad():
        xb = img.unsqueeze(0)  # Add a batch dimension
        xb = xb.to(next(model.parameters()).device)
        yb = model(xb)
        _, preds = torch.max(yb, dim=1)
        return preds.item()

# Prediction Function for External Images
def predict_external_image(image_path, model):
    image = Image.open(Path(image_path))
    example_image = transformations(image)
    predicted_class = predict_image(example_image, model)
    class_list = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    return class_list[predicted_class]  # Return the predicted class label