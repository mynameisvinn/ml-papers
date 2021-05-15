import torch

from utils import save_checkpoint

def train(model, optimizer, train_loader, val_loader, num_epochs, criterion, out_path, device):
    for epoch in range(num_epochs):
        print("Starting epoch " + str(epoch+1))
        
        model.train()
        for x1, x2, labels in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_loss = 0.0
        with torch.no_grad():
            model.eval()  # switch to eval() since there are dropout and batchnorm layers in the model
            for x1, x2, labels in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                labels = labels.to(device)
                outputs = model(x1, x2)
                loss = criterion(outputs, labels)
                batch_loss += loss.item()
        avg_val_loss = batch_loss / len(val_loader)
        print(f'Validation loss: {avg_val_loss}')
    save_checkpoint(out_path, model, optimizer, avg_val_loss)
    print("Finished Training")