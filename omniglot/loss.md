# Computing loss
The Siamese model outputs an embedding that represents the similarity between `x1` and `x2`.
```python
def forward(self, x1, x2):
    x1 = self.convs(x1)
    x1 = x1.view(-1, 256 * 6 * 6)
    output1 = self.sigmoid(self.fc1(x1))
    
    x2 = self.convs(x2)
    x2 = x2.view(-1, 256 * 6 * 6)
    output2 = self.sigmoid(self.fc1(x2))

    d = torch.abs(output1 - output2)
    similarity = self.fcOut(d)
    return similarity
```
unlike typical models, we dont update parameters after a single pass. with siamese networks, we pass two images `input1` and `input2` consecutively through a single network to get `output1` and `output2`.

`output1` and `output2` are then used to compute a distance `d`, which is passed through a single fully connected layer.

# Objective
The objective function is designed so that the model outputs 1 when the label is 1 (which indicates the two images are instances of the same character) and 0 when the label is 0.
```python
criterion = nn.BCEWithLogitsLoss()  # binary cross entropy
...
for epoch in range(num_epochs):
    for img1, img2, labels in train_loader:
    	...
    	similarity = model(img1, img2)
        loss = criterion(similarity, labels)
```


