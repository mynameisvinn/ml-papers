# Instance Segmentation
Exploring instance segmentation with Oxford's IIT Pets dataset.

## Oxford Dataset
```bash
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz
tar -xf annotations.tar.gz
```

## Inference
```python

dataset = PetsDataset(get_transform(train=True))

data_loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    num_workers=0, 
    collate_fn=utils.collate_fn)
```
Let's get our model with weights updated specifically for the pets dataset.
```python
model = get_model_instance_segmentation(num_classes=2)
checkpoint = torch.load('model/0_Apr-03.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```
Get a batch of data and pass it to our model.
```python
images, _ = next(iter(data_loader))
prediction = model(images)
```
### Examining results
The model returns up to 100 predictions, descending in order of confidence.