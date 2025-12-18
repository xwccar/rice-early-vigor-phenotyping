import torch
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device,
                epochs=100, save_path="results/saved_models", model_name="model", use_time=True):
    os.makedirs(save_path, exist_ok=True)
    best_r2 = -1
    if scheduler is None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)



    for epoch in range(epochs):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            if use_time:
                points, time_tensor, labels = batch
                time_tensor = time_tensor.to(device)
            else:
                points, labels = batch
                time_tensor = None

            points = points.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if use_time:
                outputs = model(points, time_tensor)
            else:
                outputs = model(points)

            loss = criterion(outputs.view(-1), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        train_r2 = r2_score(y_true, y_pred)

        val_loss, val_r2 = validate_model(model, val_loader, criterion, device, use_time)
        current_lr = optimizer.param_groups[0]['lr']
        if val_r2 > best_r2:
            best_r2 = val_r2
            torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}_best.pth"))
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, R2: {train_r2:.4f} | Val Loss: {val_loss:.4f}, R2: {val_r2:.4f}, 📉 当前学习率: {current_lr}")

        scheduler.step()
        # scheduler.step(val_loss)

def validate_model(model, dataloader, criterion, device, use_time=True):
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            if use_time:
                points, time_tensor, labels = batch
                time_tensor = time_tensor.to(device)
            else:
                points, labels = batch
                time_tensor = None

            points = points.to(device)
            labels = labels.to(device)

            if use_time:
                outputs = model(points, time_tensor)
            else:
                outputs = model(points)

            loss = criterion(outputs.view(-1), labels.view(-1))
            total_loss += loss.item()

            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    r2 = r2_score(y_true, y_pred)
    return avg_loss, r2

def collate_with_time(batch):
    if len(batch[0]) == 3:
        points_list, time_list, label_list = zip(*batch)
        points_padded = pad_sequence(points_list, batch_first=True)
        times = torch.stack(time_list)
        labels = torch.stack(label_list)
        return points_padded, times, labels
    else:
        points_list, label_list = zip(*batch)
        points_padded = pad_sequence(points_list, batch_first=True)
        labels = torch.stack(label_list)
        return points_padded, labels