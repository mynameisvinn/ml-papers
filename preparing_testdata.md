# preparing test/eval data
we hold out 20 alphabets from training. 

from this collection of 20 alphabets we select a `mainImg`, which is the image that we will test against 20 images. 
```python
class NWayOneShotEvalSet(Dataset):
...

    category = random.choice(categories)  # choose a random alphabet
    character = random.choice(category[1])  # choose a letter/character from that the alphabet
    imgDir = root_dir + category[0] + '/' + character
    imgName = random.choice(os.listdir(imgDir))  # randomly choose an exammple as our main image
    mainImg = Image.open(imgDir + '/' + imgName)
```
## construct `testSet` by selecting "support" images
we colllect `numWay` examples, where `numWay` represents the number of support images to compare `mainImg`. 

the neural net computes similarity scores between `mainImg` and every image in `testSet`. the pair with the highest similarity score is the model's prediction.
```python
class NWayOneShotEvalSet(Dataset):
    ...

    testSet = []  # similarity score between main image is computed for each "support" image in testSet
    label = np.random.randint(self.numWay)  # numway represents number of alphabets we want to compare to

    for i in range(self.numWay):  # how many examples (each example is a unique alphabet) do we get to use

        testImgDir = imgDir
        testImgName = ''
        
        # we want at least one example in testSet thats from the same letter
        if i == label:  
            testImgName = random.choice(os.listdir(imgDir))
        
        # all other examples should come from other alphabets/characters
        else:
            testCategory = random.choice(categories)  # choose a random alphabet
            testCharacter = random.choice(testCategory[1])  # with another random letter from that alphabet
            testImgDir = root_dir + testCategory[0] + '/' + testCharacter
            
            # dont want to draw another sample thats from the same letter as mainImg
            while testImgDir == imgDir:
                testImgDir = root_dir + testCategory[0] + '/' + testCharacter
            testImgName = random.choice(os.listdir(testImgDir))
        
        testImg = Image.open(testImgDir + '/' + testImgName)
        
        if self.transform:
            testImg = self.transform(testImg)
        
        testSet.append(testImg)
```