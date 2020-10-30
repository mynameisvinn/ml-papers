# evaluating one-shot learning
```python
# evaluation metrics
def eval(model, test_loader):
    with torch.no_grad():
        model.eval()  # set to eval() model since model contains batchnorm and dropout layers
        correct = 0

        for mainImg, imgSets, label in test_loader:
            mainImg = mainImg.to(device)
            predVal = 0
            pred = -1
            
            # go through each support image in testsets (there are 20) and compute similarity score
            for i, testImg in enumerate(imgSets):
                testImg = testImg.to(device)
                output = model(mainImg, testImg)  # output is [0, 1] and represents similarity score
                if output > predVal:
                    pred = i  # we want the idx with the highest similarity
                    predVal = output
            label = label.to(device)
            if pred == label:
                correct += 1
```