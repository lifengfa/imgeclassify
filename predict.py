
from statistics import mode
import torch
import torch.nn as nn
from torchvision import transforms
from model import SwinTransformer
from PIL import Image
model = torch.load('1.pth')
model.cuda()
model.eval()
print(model)
# feature_model = list(model.classifier.children())
# print(feature_model[-1])
img = Image.open('./data/20201126171929_82/BireView.png')
img = img.resize((224,224), Image.BICUBIC)
img = transforms.ToTensor()(img)
img = img.unsqueeze(0).cuda()
x = model(img)
sig = nn.Sigmoid()
print(sig(x))