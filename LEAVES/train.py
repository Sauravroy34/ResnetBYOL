import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from models.BYOL import BYOL

class ECGBYOLDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Convert the filtered numpy/list signal back to a tensor
        signal = torch.tensor(item['ecg_signal'], dtype=torch.float)
        
        # If your signal is 1D (length), BYOL usually expects (channels, length)
        # Uncomment the next line if you get dimensionality errors in your ResNet1D:
        signal = signal.unsqueeze(0) 
        
        # Duplicating the signal since your view makers handle the transformations
        x1 = signal.clone()
        x2 = signal.clone()
    
        
        return x1, x2

print("Downloading dataset from Hugging Face...")
hf_data = load_dataset("Codemaster67/ECGFiltered")

train_dataset = ECGBYOLDataset(hf_data['train'])
valid_dataset = ECGBYOLDataset(hf_data['validation'])
# test_dataset = ECGBYOLDataset(hf_data['test']) # Available when you need it

BATCH_SIZE = 64

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

model = BYOL()
LR = 1e-3
epochs = 1000
base_save_path = "Resnet50_1d_BYOL"


def set_requires_grad(modules, val):
    for mod in modules:
        for param in mod.parameters():
            param.requires_grad = val

def trainBYOL(model = model, trainloader = trainloader, validloader =validloader , device = "cpu"):
    view_parameters = list(model.view1.parameters()) + list(model.view2.parameters())
    if hasattr(model.view1, 'params'):
        view_parameters += model.view1.params + model.view2.params
        
    optimizer_view = torch.optim.Adam(view_parameters, lr=LR)

    online_parameters = (
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.online_predictor.parameters())
    )
    optimizer_online = torch.optim.Adam(online_parameters, lr=LR)

    view_modules = [model.view1, model.view2]
    online_modules = [
        model.online_encoder, 
        model.online_projector, 
        model.online_predictor
    ]

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=3, delta=0.001)
    model.to(device)
    for e in range(epochs):
        # ---------------- TRAINING PHASE ----------------
        model.train()
        epoch_loss_online, epoch_loss_view = 0.0, 0.0

        loop = tqdm(trainloader, desc=f"Epoch {e+1}/{epochs} [Train]")
        for batch in loop:
            x1, x2, label = batch
            
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)

            # 1. Zero all gradients at the start
            optimizer_view.zero_grad()
            optimizer_online.zero_grad()

            # 2. Forward pass
            loss = model(x1, x2)

            # 3. Compute gradients for View Makers (Maximize loss)
            set_requires_grad(online_modules, False)
            set_requires_grad(view_modules, True)
            
            view_maker_loss = -loss
            view_maker_loss.backward(retain_graph=True)

            # 4. Compute gradients for Online Network (Minimize loss)
            set_requires_grad(view_modules, False)
            set_requires_grad(online_modules, True)
            
            loss.backward()

            # 5. Step both optimizers ONLY AFTER all backward passes are done
            optimizer_view.step()
            optimizer_online.step()

            # 6. EMA update
            model.update_moving_average()

            # Logging
            epoch_loss_view += view_maker_loss.item()
            epoch_loss_online += loss.item()
            loop.set_postfix(Online_Loss=loss.item(), View_Loss=view_maker_loss.item())
        avg_train_online_loss = epoch_loss_online / len(trainloader)
        avg_train_view_loss = epoch_loss_view / len(trainloader)
        
        # ---------------- VALIDATION PHASE ----------------
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            val_loop = tqdm(validloader, desc=f"Epoch {e+1}/{epochs} [Valid]", leave=False)
            for batch in val_loop:
                x1, x2, _ = batch
                x1 = x1.to(device, dtype=torch.float)
                x2 = x2.to(device, dtype=torch.float)
                
                # Forward pass for validation (loss evaluation only)
                loss = model(x1, x2)
                epoch_val_loss += loss.item()
                
        avg_val_loss = epoch_val_loss / len(validloader)

        print(f'End of Epoch {e+1} - Train Loss: {avg_train_online_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

        # ---------------- CHECKPOINTS & EARLY STOPPING ----------------
        if e % 10 == 9:
            checkpoint_path = f"{base_save_path}_checkpoint_{e + 1}.pth"
            torch.save(model.online_encoder.state_dict(), checkpoint_path)

        # Evaluate Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {e+1}! Best validation loss was {early_stopping.best_loss:.4f}.")
            break

    # Save final model
    torch.save(model.online_encoder.state_dict(), f"{base_save_path}_final.pth")
    print("Training Complete. Final model saved.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainBYOL(device = device)
