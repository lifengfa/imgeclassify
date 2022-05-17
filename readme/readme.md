## 实验结果（所有backbone都在imagenet1k预训练）



实验所用超参数：

input_size = 224#输入图片大小

batch_size = 8#每次训练输入图片数

lr = 0.0005#训练学习率

epochs = 20#训练次数

shuffle(path)

#100张图片训练 37张图片验证

train_path = path[:100]

val_path = path[100:]

### Swin transformer作backbone

![image-20220517143507072](.\stresults.png)

trainloss is 0.4220248103141785
trainprecision is 0.7083527629192059
trainrecall is 0.45994975245915926
valloss is 0.42788094740647536
valprecision is 0.7307692307692307
valrecall is 0.5683760711779962
F1 is 0.6394230786961718
model save at 13 epoch



### Resnet50作backbone

![image-20220517141924488](.\resnet50.png)



trainloss is 0.3784354758262634
trainprecision is 0.7416666746139526
trainrecall is 0.48795553812613857
valloss is 0.4010630836471533
valprecision is 0.7948717948717948
valrecall is 0.5598290623762668
F1 is 0.6569602864744241
model save at 16 epoch



### Swin transformer-cat提取特征

考虑到沙盘场景中既要考虑整体，又要考虑局部(个体单位的受伤，状态等...)。所以将swin transformer每层layer输出cat起来，最后进行分类。这样就既包含图片整体信息，也包含swin trnasformer中每个小窗口单独的信息。

![image-20220517142727575](.\stcatresults.png)

trainloss is 0.39959718704223635
trainprecision is 0.7356837632564398
trainrecall is 0.47936428968723005
valloss is 0.4157009140039102
valprecision is 0.7564102564102564
valrecall is 0.5598290623762668
F1 is 0.6434398951223324
model save at 11 epoch



### Unet作backbone

![image-20220517144314065](.\unet.png)

trainloss is 0.49883436918258667
trainprecision is 0.604364982018104
trainrecall is 0.510168575323545
valloss is 0.5350971838984734
valprecision is 0.6581196593932617
valrecall is 0.6111111136583182
F1 is 0.6337448579272745
model save at 5 epoch