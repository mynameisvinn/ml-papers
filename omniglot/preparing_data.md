# Training data

## Similar images
50% of the dataset consists "similar" pairs of objects, which means they are instances drawn from a single character:

```python
category = random.choice(categories)  # randomly select an alphabet
character = random.choice(category[1])  # choose a char from this alphabet

# pick two random instances of this character
imgDir = root_dir + category[0] + '/' + character  
img1Name = random.choice(os.listdir(imgDir))
img2Name = random.choice(os.listdir(imgDir))

img1 = Image.open(imgDir + '/' + img1Name)
img2 = Image.open(imgDir + '/' + img2Name)

label = 1.0  # label 1 indicates they are two instances from a single character
```
## Dissimilar images
The other half consists of "dissimilar" objects. These are images are drawn from different alphabets.
```python
category1 = random.choice(categories)
character1 = random.choice(category1[1])  # pick the second example because example 0 is used for similar images
imgDir1 = root_dir + category1[0] + '/' + character1

category2 = random.choice(categories)
character2 = random.choice(category2[1])
imgDir2 = root_dir + category2[0] + '/' + character2

img1Name = random.choice(os.listdir(imgDir1))
img2Name = random.choice(os.listdir(imgDir2))

# we dont want images to be identical
while img1Name == img2Name:
    img2Name = random.choice(os.listdir(imgDir2))

img1 = Image.open(imgDir1 + '/' + img1Name)
img2 = Image.open(imgDir2 + '/' + img2Name)

label = 0.0  # label 0 indicates they are dissimilar images
```