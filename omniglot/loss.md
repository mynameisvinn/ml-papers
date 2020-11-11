# computing loss
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

# whats the objective?
loss is computed by comparing this predicted similarity score against ground truth labels (where `y=1` if the images are examples of the same character and `y=0` if the images are not).
```python
criterion = nn.BCEWithLogitsLoss()
...
for epoch in range(num_epochs):  # number of passes through training dataset
    for img1, img2, labels in train_loader:
    	...
    	similarity = model(img1, img2)
        loss = criterion(similarity, labels)
```


