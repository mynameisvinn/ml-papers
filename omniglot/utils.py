import os
import random

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

# saving and loading checkpoint mechanisms
def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')

def load_checkpoint(model, optimizer):
    save_path = f'siameseNet-batchnorm50.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    print(f'Model loaded from <== {save_path}')
    
    return val_loss

class OmniglotDataset(Dataset):    
    def __init__(self, categories, root_dir, setSize, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize
    
    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        img1 = None
        img2 = None
        label = None
        
        if idx % 2 == 0: # select two examples of the same character
            category = random.choice(self.categories)  # a category is the alphabet
            character = random.choice(category[1])  # character is the character, so the english alphabet has 26 characters
            
            # pick two random examples of a given character in this alphabet
            imgDir = self.root_dir + category[0] + '/' + character  
            img1Name = random.choice(os.listdir(imgDir))
            img2Name = random.choice(os.listdir(imgDir))
            img1 = Image.open(imgDir + '/' + img1Name)
            img2 = Image.open(imgDir + '/' + img2Name)

            label = 1.0
            
        else: # select characters from different alphabets for both images
            category1, category2 = random.choice(self.categories), random.choice(self.categories)
            character1, character2 = random.choice(category1[1]), random.choice(category2[1])
            imgDir1, imgDir2 = self.root_dir + category1[0] + '/' + character1, self.root_dir + category2[0] + '/' + character2
            img1Name = random.choice(os.listdir(imgDir1))
            img2Name = random.choice(os.listdir(imgDir2))
            while img1Name == img2Name:
                img2Name = random.choice(os.listdir(imgDir2))
            label = 0.0
            img1 = Image.open(imgDir1 + '/' + img1Name)
            img2 = Image.open(imgDir2 + '/' + img2Name)
#         plt.imshow(img1)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))          



# creates n-way one shot learning evaluation
class NWayOneShotEvalSet(Dataset):
    def __init__(self, categories, root_dir, setSize, numWay, transform=None):
        self.categories = categories
        self.root_dir = root_dir
        self.setSize = setSize
        self.numWay = numWay
        self.transform = transform
    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        # find one main image
        category = random.choice(categories)  # which alphabet do we want to go with
        character = random.choice(category[1])  # which letter/character in the alphabet do we want to look at
        imgDir = root_dir + category[0] + '/' + character  # now we've decided on a specific letter in a specific alphabet
        imgName = random.choice(os.listdir(imgDir))  # randomly choose an exammple as our main image
        mainImg = Image.open(imgDir + '/' + imgName)

        if self.transform:
            mainImg = self.transform(mainImg)
        
        # find n numbers of distinct images, 1 in the same set as the main
        testSet = []  # the main image is compared against this list of TestSet
        label = np.random.randint(self.numWay)  # numway represents number of alphabets we want to compare to
        
        for i in range(self.numWay):  # how many examples (each example is a unique alphabet) do we get to use
            testImgDir = imgDir
            testImgName = ''
            
            # we want at least one example from the same letter of the same alphabet in the testset
            if i == label:  
                testImgName = random.choice(os.listdir(imgDir))
            
            # all other examples should come from other alphabets/characters
            else:
                testCategory = random.choice(categories)  # now we choose another random category
                testCharacter = random.choice(testCategory[1])  # with another random letter from that alphabet
                testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                while testImgDir == imgDir:
                    testImgDir = root_dir + testCategory[0] + '/' + testCharacter
                testImgName = random.choice(os.listdir(testImgDir))  # with a random example
            
            testImg = Image.open(testImgDir + '/' + testImgName)
            
            if self.transform:
                testImg = self.transform(testImg)
            
            testSet.append(testImg)
        # plt.imshow()
        return mainImg, testSet, torch.from_numpy(np.array([label], dtype = int))