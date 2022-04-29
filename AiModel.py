import torch
import timm
from torch.nn import Linear
from PIL import Image

WEIGHT_PATH = "model1.pth"
N_CLASSES = 7

def create_model():
  model = timm.create_model(model_name = 'efficientnet_b0', pretrained=False)
  model.classifier = Linear(in_features=1280, out_features=N_CLASSES, bias=True)

  device = torch.device('cpu')  
  model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
  model.eval()  

  return model

model = create_model()

from torchvision import transforms
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

def detectIssue(image):
  image = Image.open(image) # Byte Image도 open으로 열 수 있다.
  image = image.convert('RGB')
  image = tf(image)
  image = torch.unsqueeze(image, 0) # batchsize = 1 넣어주기(DataLoader 가 없기 때문)
  H = model(image)
  predict = torch.sigmoid(H).detach().cpu().numpy().squeeze(0).tolist()
  
  return predict