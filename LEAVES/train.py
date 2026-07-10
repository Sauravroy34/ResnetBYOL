import os
import torch
import torch.nn as nn
from tqdm import tqdm
from models.BYOL import BYOL



model = BYOL()


# Constants
LR = 1e-3
epochs = 1000
base_save_path = "Resnet50_1d_BYOL"

def set_requires_grad(modules, val):
    for mod in modules:
        for param in mod.parameters():
            param.requires_grad = val

def trainBYOL(model, trainloader, testloader, device):
    

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

    for e in range(epochs):
        model.train()
        epoch_loss_online, epoch_loss_view = 0.0, 0.0

        loop = tqdm(trainloader, desc=f"Epoch {e+1}/{epochs}")
        for batch in loop:
            x1, x2, label = batch
            
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)

            optimizer_view.zero_grad()
            optimizer_online.zero_grad()

            # Forward pass
            loss = model(x1, x2)


            set_requires_grad(online_modules, False)
            set_requires_grad(view_modules, True)

            view_maker_loss = -loss
            view_maker_loss.backward(retain_graph=True)
            optimizer_view.step()


            set_requires_grad(view_modules, False)
            set_requires_grad(online_modules, True)

            optimizer_view.zero_grad()

            loss.backward()
            optimizer_online.step()

            model.update_moving_average()

            epoch_loss_view += view_maker_loss.item()
            epoch_loss_online += loss.item()
            
            loop.set_postfix(Online_Loss=loss.item(), View_Loss=view_maker_loss.item())

        avg_online_loss = epoch_loss_online / len(trainloader)
        avg_view_loss = epoch_loss_view / len(trainloader)
        print(f'End of Epoch {e+1} - Avg Online Loss: {avg_online_loss:.4f}, Avg View Loss: {avg_view_loss:.4f}')

        if e % 10 == 9:
            checkpoint_path = f"{base_save_path}_checkpoint_{e + 1}.pth"
            torch.save(model.online_encoder.state_dict(), checkpoint_path)

    torch.save(model.online_encoder.state_dict(), f"{base_save_path}_final.pth")