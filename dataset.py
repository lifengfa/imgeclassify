import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import random
from torchvision import transforms
import hashlib


path = os.listdir("./data")


def get_unk_mask_indices(image,testing,num_labels,known_labels,epoch=1):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices


class EmoDataset(Dataset):
    def __init__(self,path,input_size,testing=False):
        super(EmoDataset,self).__init__()
        self.path = path
        self.input_size = input_size
        self.testing = testing

    def __getitem__(self, index):
        img = Image.open('./data/'+self.path[index]+'/BireView.png')
        with open('./data/'+self.path[index]+'/'+self.path[index]+'.json','r',encoding='utf8')as f:
            labels = json.load(f)
        label_list = []
        a = labels['themes']
        for j in a:
            label_list.append(j['type'])
        label = np.zeros(8)
        for i,l in enumerate(label_list):
            label[l] = 1
        # img.show()
        img = self.img_agument(img) 
        img = transforms.ToTensor()(img)
        label = torch.FloatTensor(label)
        unk_mask_indices = get_unk_mask_indices(img,self.testing,8,0)
        mask = label.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        # img.show()       
        return img,label,mask

    def __len__(self):
        return len(self.path)

    def img_agument(self,img):
        w,h = img.size[0],img.size[1]
        ratio = random.uniform(.5,2)
        nh = int(ratio*h)
        scale = w/nh
        s = self.input_size
        if scale > 1:
            image = img.resize((s,int(s/scale)), Image.BICUBIC)
            dh = s-int(s/scale)
            dy = int(random.uniform(0,dh))
            new_image = Image.new('RGB', (s,s), (128,128,128))
            new_image.paste(image, (0, dy))
        else:
            image = img.resize((int(s*scale),s), Image.BICUBIC)
            dw = int(s*scale)-s
            dx = int(random.uniform(0,dw))
            new_image = Image.new('RGB', (s,s), (128,128,128))
            new_image.paste(image, (dx,0))

        return new_image



if __name__ == '__main__':
    dataset = EmoDataset(path,224)
    img,label,mask = dataset.__getitem__(1)
    print((label))
    print(dataset.__len__())