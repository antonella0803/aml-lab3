import torch
import torch.nn as nn
from dataset.dataset import get_dataloaders
from models.custom_model import CustomCNN

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Dispositivo per la valutazione: {device}")
    
    # Caricamento dei dataloader
    dataloaders, class_names = get_dataloaders(data_dir='dataset/tiny-imagenet-200', batch_size=64)
    val_loader = dataloaders['val']
    
    # Istanziazione e caricamento dei pesi del modello
    model = CustomCNN(num_classes=len(class_names)).to(device)
    checkpoint_path = 'checkpoints/custom_model.pth'
    
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"--> Modello caricato con successo da {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ Errore: File {checkpoint_path} non trovato! Esegui prima train.py.")
        return

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
    final_loss = running_loss / total_samples
    final_acc = running_corrects / total_samples
    
    print("\n" + "="*40)
    print(f"📊 RISULTATI FINALI DI VALUTAZIONE")
    print(f"Validation Loss:     {final_loss:.4f}")
    print(f"Validation Accuracy: {final_acc * 100:.2f}%")
    print("="*40)

if __name__ == '__main__':
    evaluate()
