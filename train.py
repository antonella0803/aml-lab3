import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset import get_dataloaders
from models.custom_model import CustomCNN
from utils.visualization import plot_training_curves, show_predictions

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)
        
    return running_loss / total_samples, running_corrects / total_samples

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
    return running_loss / total_samples, running_corrects / total_samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Dispositivo in uso: {device}")
    
    dataloaders, class_names = get_dataloaders(data_dir='dataset/tiny-imagenet-200', batch_size=64)
    
    model = CustomCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5  # Impostato a 5 per una prova rapida
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    os.makedirs('checkpoints', exist_ok=True)
    
    print("--> Inizio addestramento...")
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, dataloaders['val'], criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoca [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
              
    # Salvataggio modello
    torch.save(model.state_dict(), 'checkpoints/custom_model.pth')
    print("--> Modello salvato in checkpoints/custom_model.pth")
    
    # Visualizzazioni
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    show_predictions(model, dataloaders['val'], class_names, device)

if __name__ == '__main__':
    main()
