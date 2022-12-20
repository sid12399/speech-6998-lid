import kaldiio
import sys, os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pickle import Unpickler
from tdnn import TDNN
from collections import defaultdict

random.seed(1)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

in_set = ['ENG', 'GER', 'ICE', 'FRE', 'SPA', 'ARA', 'RUS', 'BEN', 'KAS', 'GRE', 'CAT', 'KOR', 'TUR', 'TAM', 'TEL', 'CHI', 'TIB', 'JAV', 'EWE', 'HAU', 'LIN', 'YOR', 'HUN', 'HAW', 'MAO', 'ITA', 'URD', 'SWE', 'PUS', 'GEO', 'HIN', 'THA']
out_of_set = ['DUT', 'HEB', 'UKR', 'BUL', 'PER', 'ALB', 'UIG', 'MAL', 'BUR', 'IBA', 'ASA', 'AKU', 'ARM', 'HRV', 'FIN', 'JPN', 'NOR', 'NEP', 'RUM']

root_dir = "db/cu-multilang-dataset/"


data = []
for i,lang in enumerate(in_set + out_of_set, 0):
    print(lang, "(In-set)" if lang in in_set else "(Out-of-set)")
    
    for f_idx in range(1, 30 + 1):
        
        filepath = root_dir + lang + '/data/raw_mfcc_pitch_' + lang + '.' + str(f_idx) + '.ark'

        for key, numpy_array in kaldiio.load_ark(filepath):
            inputs = torch.from_numpy(np.expand_dims(numpy_array, axis=0))
            inputs.to(device)
            labels = torch.from_numpy(np.array([i if lang in in_set else -1]))
            labels.to(device)
            data.append((inputs, labels))


print("\ncombining all mfccs by label")
random.shuffle(data)
data_concatenated = dict()
for iter,i in enumerate(data): 
    label = i[1].numpy()[0]
    mfcc = np.squeeze(i[0].numpy(), axis=0)
    if label in data_concatenated:
        data_concatenated[label].append(mfcc)
    else:
        data_concatenated[label] = [mfcc]

for i in data_concatenated:
    data_concatenated[i] = np.vstack(data_concatenated[i])

del data

def chunkify_tensor(tensor, size=400):
    return torch.split(tensor, size, dim=1)[:-1] # except last one bc that isn't the right size

'''
Train (tdnn):       95% of in-set
Train (lda/plda):   80% of out-of-set
Test:               5% of in-set + 20% out-of-set
'''

print("chunking and splitting into 3 groups:")
train1, train2, test = [], [], []
for i in data_concatenated:
    label = torch.from_numpy(np.array([i]))
    mfcc = torch.from_numpy(np.expand_dims(data_concatenated[i], axis=0))
    chunks = chunkify_tensor(mfcc)
    if i >= 0:
        cutoff = int(len(chunks) * 0.95)
        for chunk in chunks[:cutoff]:
            train1.append((chunk.to(device), label.to(device)))
        for chunk in chunks[cutoff:]:
            test.append((chunk.to(device), label.to(device)))
    else:
        cutoff = int(len(chunks) * 0.8)
        for chunk in chunks[:cutoff]:
            train2.append((chunk.to(device), label.to(device)))
        for chunk in chunks[cutoff:]:
            test.append((chunk.to(device), label.to(device)))

del data_concatenated

print("\ttrain1:", len(train1))
print("\ttrain2:", len(train2))
print("\ttest:", len(test))
print("\t\ttest (in-set):", len([i for i in test if i[1].to('cpu').numpy()[0] >= 0]))
print("\t\ttest (oos):", len([i for i in test if i[1].to('cpu').numpy()[0] < 0]))
print()
print("labels in test:", set([i[1].to('cpu').numpy()[0] for i in test]))
print(test[0][0].size())
asdfasdfasdf = defaultdict(int)
for i in test:
    asdfasdfasdf[i[1].to('cpu').numpy()[0]] += 1
print(sorted(asdfasdfasdf.items()))
print()


class Net(nn.Module):
    def __init__(self, in_size, num_classes):
        super().__init__()
        
        self.layer1 = TDNN(input_dim=in_size, output_dim=256, context_size=3)
        self.layer2 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=1)
        self.layer3 = TDNN(input_dim=256, output_dim=256, context_size=3, dilation=1)
        self.layer4 = TDNN(input_dim=256, output_dim=256, context_size=1)
        self.layer5 = TDNN(input_dim=256, output_dim=256, context_size=1)
        self.final_layer = TDNN(input_dim=256, output_dim=num_classes, context_size=1)
        
    def forward(self, x):
        forward_pass = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            self.layer2,
            nn.ReLU(),
            self.layer3,
            nn.ReLU(),
            self.layer4,
            nn.ReLU(),
            self.layer5,
            nn.ReLU(),
            self.final_layer)
        
        return forward_pass(x)



# LOAD_PATH = 'saved-models/tdnn-256-4s-25epochs'
print('Loading the model')

if len(sys.argv) != 2:
    print("Usage: python3 decode.py <model_name> \n Please refer to the saved_models directory to see available models and include the name without any extension.")
    exit(1)

model_name = sys.argv[1]
    
infile = open("saved_models/" + model_name + ".pickle", "rb")
net = Unpickler(infile).load()
infile.close()

print('Finished loading')
print()



#only test on the in-set ones
test = [i for i in test if i[1].to('cpu').numpy()[0] >= 0]
random.shuffle(test)
print(len(test), '\n')



print("Testing...")
correct, total = 0, 0

for i, data in enumerate(test,0):
    if i%2000 == 0 and total > 0:
        print("iter:", i, "\tacc:", correct/total)

    inputs, labels = data[0].to(device), data[1].to(device)
    
    outputs = net(inputs)
    outputs = torch.mean(outputs, 1)
    outputs = F.softmax(outputs, dim=1)
    predicted = outputs.argmax(1, keepdim=True)

    total += 1
    if predicted == labels:
        correct += 1

print("\nAcc:", correct/total)
