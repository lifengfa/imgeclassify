import os
import torch
import torch.nn as nn
from dataset import EmoDataset
from model import SwinTransformer
from torch.utils.data import DataLoader
import random
from random import shuffle
from utils import load_pretrained
import torchvision
from unet import ResNet34UnetPlus,UNet
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

random.seed(99)
torch.manual_seed(99)  # 为cpu设置随机种子
torch.cuda.manual_seed(99)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(99)  # 为所有GPU设置随机种子

#选择模型：resnet50 or swin transformer or swin transformer-cat or Unet or resnet50
backbone = 'Unet'

Cuda = True
input_size = 224
batch_size = 8
lr = 0.0005
epochs = 20
freeze_epoch = 20
path = os.listdir("./data")
shuffle(path)
train_path = path[:100]
val_path = path[100:]

train_dataset = EmoDataset(train_path,input_size)
val_dataset = EmoDataset(val_path,input_size,True)
trainset_size = train_dataset.__len__()
valset_size = val_dataset.__len__()

train_dataloader =  DataLoader(train_dataset,batch_size,True)
val_dataloader =  DataLoader(val_dataset,1,False)




def calculate_acuracy_mode_one(pre,label,th=0.5):
    pre_result = pre>th
    pre_result = pre_result.float()
    pre_num = torch.sum(pre_result)
    if pre_num == 0:
        return 0,0
    target_num = torch.sum(label)
    true_pre_sum = torch.sum(pre_result*label)
    precision = true_pre_sum/pre_num
    recall = true_pre_sum/target_num
    return precision.item(), recall.item()


def train_model(model,criterion,optimizer,backbone):
    losses_t = []
    losses_v = []
    f1s = []
    Unfreez_flag = True
    sigmoid = nn.Sigmoid()
    maxf1 = 0
    for epoch in range(epochs):
        if epoch>freeze_epoch and Unfreez_flag:
            print('*'*10+'unfreezeing...')
            for param in model.parameters():
                param.requires_grad = True
            Unfreez_flag = False
        
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train','val']:
            running_loss = 0.0
            running_precision = 0.0
            running_recall = 0.0
            batch_num = 0

            if phase == 'train':
                

                
                model.train()
                for data,label,mask in train_dataloader:
                    if Cuda:
                        data,label = data.cuda(),label.cuda()
                    optimizer.zero_grad()
                    if backbone == 'CTran':
                        mask_in = mask.clone().cuda()
                        outputs,_,_ = model(data,mask_in)
                    else:
                        outputs = model(data)
                    loss = criterion(sigmoid(outputs),label)
                    precision, recall = calculate_acuracy_mode_one(sigmoid(outputs), label)
                    running_precision += precision
                    running_recall += recall
                    batch_num += 1
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    running_loss += loss.item()*data.shape[0]
                    

            else:
                with torch.no_grad():
                    model.eval()
                    for data,label,mask in val_dataloader:
                        if Cuda:
                            data,label = data.cuda(),label.cuda()
                        if backbone == 'CTran':
                            mask_in = mask.clone().cuda()
                            outputs,_,_ = model(data,mask_in)
                        else:
                            outputs = model(data)
                        loss = criterion(sigmoid(outputs),label)
                        running_loss += loss.item() * data.size(0)
                        precision, recall = calculate_acuracy_mode_one(sigmoid(outputs), label)
                        running_precision += precision
                        running_recall += recall
                        batch_num += 1

            if phase == 'train':
                epoch_loss = running_loss/trainset_size
                print('trainloss is',epoch_loss)
                epoch_precision = running_precision/batch_num
                epoch_recall = running_recall/batch_num
                print('trainprecision is',epoch_precision)
                print('trainrecall is',epoch_recall)
                losses_t.append(epoch_loss)
            else:
                epoch_loss = running_loss/valset_size
                print('valloss is',epoch_loss)
                epoch_precision = running_precision/batch_num
                epoch_recall = running_recall/batch_num
                f1 = 2*(epoch_precision*epoch_recall)/(epoch_precision+epoch_recall)
                print('valprecision is',epoch_precision)
                print('valrecall is',epoch_recall)
                print('F1 is',f1)
                losses_v.append(epoch_loss)
                f1s.append(f1)
                if f1 > maxf1:
                    torch.save(model,'1.pth')
                    maxf1 = f1
                    print('model save at{} epoch'.format(epoch))

    return losses_t,losses_v,f1s


    

if __name__ == '__main__':

    if backbone == 'swin transformer':
        print('using swin transformer')
        model = SwinTransformer(False,num_classes=8)
        load_pretrained('swin_tiny_patch4_window7_224.pth',model)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
        nn.Linear(768, 8),
        )
    elif backbone == 'swin transformer-cat':
        print('using swin transformer-cat')
        model = SwinTransformer(True,num_classes=8)
        load_pretrained('swin_tiny_patch4_window7_224.pth',model)
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Sequential(
        nn.Linear(1920, 8),
        )
    elif backbone == 'Unet':
        print('using Unet')
        model = ResNet34UnetPlus(1)
        model.require_encoder_grad(False)

    else:
        print('using resnet50')
        model = torchvision.models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 8),
        )
    if Cuda:
        model.cuda()

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr)
    loss_t,loss_v,f1 = train_model(model, criterion, optimizer,backbone)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(loss_t, label='train loss')
    ax1.plot(loss_v, label='val loss')

    # ax1.set_ylim([0.35,0.6])
    ax1.legend()
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(f1, label='f1')
    # ax2.plot(loss_v, label='val acc')

    # ax2.set_ylim([0.5,0.7])
    ax2.legend()
    ax2.set_ylabel('F1')
    ax2.set_xlabel('Epoch')

    fig.suptitle('Training History')
    plt.show()

    
        

        









            




