
import os
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, dataloader, device, save_path, use_time=True):
    model.eval()
    y_true, y_pred, filenames, days, a_list, b_list = [], [], [], [], [], []

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

            outputs = model(points, time_tensor) if use_time else model(points)
            y_true.extend(labels.view(-1).cpu().numpy())
            y_pred.extend(outputs.view(-1).cpu().numpy())

            if hasattr(batch, 'filenames'):
                filenames.extend(batch.filenames)
            if hasattr(batch, 'days'):
                days.extend(batch.days)
            if hasattr(batch, 'a_list'):
                a_list.extend(batch.a_list)
                b_list.extend(batch.b_list)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame({
        "A": a_list if a_list else list(range(len(y_true))),
        "B": b_list if b_list else list(range(len(y_true))),
        "True": y_true,
        "Predicted": y_pred,
        "Error": [p - t for t, p in zip(y_true, y_pred)],
        "Day": days if days else [None]*len(y_true),
        "File": filenames if filenames else [None]*len(y_true)
    })
    df.to_csv(save_path, index=False)

    return mse, r2
