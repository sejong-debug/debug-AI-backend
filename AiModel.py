import torch
import timm
from torch.nn import Linear
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils import get_crop_mask, get_crop_map

crop_mask = get_crop_mask()
inverse_crop_map = {v:k for k,v in get_crop_map().items()}

tf = T.Compose([
    T.Resize((224, 224), interpolation=3),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

N_CLASSES = 72


WEIGHT_PATH = "model1.pth"

def create_model():
  model = timm.create_model(model_name = 'efficientnet_b0', pretrained=False)
  model.classifier = Linear(in_features=1280, out_features=N_CLASSES, bias=True)

  model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
  model.eval()  

  return model

model = create_model()


all_label     = [n for n in range(72)]
healthy_label = [9, 10, 11] + list(range(53, 71+1))
disease_label = list(set(all_label) - set(healthy_label))

import numpy as np
def detectIssue(image, crop_type):
  image = Image.open(image) # Byte Image도 open으로 열 수 있다.
  image = image.convert('RGB')
  image = tf(image)
  image = torch.unsqueeze(image, 0) # batchsize = 1 넣어주기(DataLoader 가 없기 때문)
  H = model(image)
  H = H.detach().cpu().numpy()

  crop_type = inverse_crop_map[crop_type]

  H *= (np.eye(N_CLASSES)[crop_type] @ crop_mask)

  predict = H.squeeze(0)[disease_label]
  # 백엔드에겐 정상을 제외한 '질병만의' 확률값을 전송해야한다.

  return list(predict)