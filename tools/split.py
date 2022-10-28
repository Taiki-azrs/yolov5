import os
import glob
import random
train_ratio=0.8
val_ratio=0.2
path_list = glob.glob('images/*.png')
random.shuffle(path_list)
val_list=path_list[:int(len(path_list)*0.2)]
train_list=path_list[int(len(path_list)*0.2):]
with open('train.txt','w') as f:
    for i in train_list:
        f.write('./{}\n'.format(i))



with open('val.txt','w') as f:
    for i in val_list:
        f.write('./{}\n'.format(i))
