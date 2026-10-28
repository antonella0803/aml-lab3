import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Mostra i grafici di Loss e Accuracy nel corso delle epoche.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def show_predictions(model, dataloader, class_names, device, num_images=6):
    """
    Mostra un campione di predizioni sul set di validazione.
    """
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
    plt.figure(figsize=(12, 6))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 3, i + 1)
        
        # Denormalizzazione dell'immagine per la visualizzazione
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        title_color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}", color=title_color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
