import torch
import timm
from torch.nn import Linear
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


tf = T.Compose([
    T.Resize((224, 224), interpolation=3),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

N_CLASSES = 72


WEIGHT_PATH = "model1.pth"

def create_model():
  model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=False)
  model.head = Linear(in_features=768, out_features=N_CLASSES, bias=True)
  model.eval()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
  model.eval()  

  return model

model = create_model()

def detectIssue(image):
  image = Image.open(image) # Byte Image도 open으로 열 수 있다.
  image = image.convert('RGB')
  image = tf(image)
  image = torch.unsqueeze(image, 0) # batchsize = 1 넣어주기(DataLoader 가 없기 때문)
  H = model(image)
  predict = torch.sigmoid(H).detach().cpu().numpy().squeeze(0).tolist()
  
  return predict