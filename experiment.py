import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer import train_model
from dataloader import RiceTillerDataset
from evaluator import evaluate_model
from utils.visualize import plot_from_csv

from models.dgcnn import DGCNN
from models.pointconv import PointConvRegressor
from models.pointtransformer import PointTransformerRegressor
from models.pct import PctRegressor
from models.pointnet import PointNet
from trainer import collate_with_time


def custom_collate_fn(batch):
    if len(batch[0]) == 3:  # 使用时间
        points, times, labels = zip(*batch)
        return list(points), torch.stack(times), torch.stack(labels)
    else:
        points, labels = zip(*batch)
        return list(points), torch.stack(labels)


def run_experiments(configs, data_root, label_files_dict, day_mapping=None, results_dir="results"):

    for config in configs:
        model_name = config["model"]
        print(f"🚀 正在运行模型: {model_name} | 配置: {config}")

        dataset = RiceTillerDataset(
            root_dir=data_root,
            label_files_dict=label_files_dict,
            day_mapping=day_mapping,
            use_rgb=config.get("use_rgb", False),
            use_time=config.get("use_time", True)
        )

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            drop_last=True,
            batch_size=config.get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_with_time
        )
        val_loader = DataLoader(
            val_dataset,
            drop_last=True,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            collate_fn=collate_with_time
        )

        # 选择模型
        if model_name.lower() == "dgcnn":
            model = DGCNN(
                use_time=config.get("use_time", True),
                use_attention=config.get("use_attention", False),
                use_residual=config.get("use_residual", False),
                activation=config.get("activation", "relu"),
                use_feat_norm=config.get("use_feat_norm", False),
            )
        elif model_name.lower() == "pointconv":
            model = PointConvRegressor(
                use_time=config.get("use_time", True),
                use_attention=config.get("use_attention", False),
                use_residual=config.get("use_residual", False),
                activation=config.get("activation", "relu"),
                use_feat_norm=config.get("use_feat_norm", False),
            )
        elif model_name.lower() == "pointtransformer":
            model = PointTransformerRegressor(
                use_time=config.get("use_time", True),
                use_attention=config.get("use_attention", False),
                use_residual=config.get("use_residual", False),
                activation=config.get("activation", "relu"),
                use_feat_norm=config.get("use_feat_norm", False),
            )
        elif model_name.lower() == "pct":
            model = PctRegressor(
                use_time=config.get("use_time", True),
                use_attention=config.get("use_attention", False),
                use_residual=config.get("use_residual", False),
                activation=config.get("activation", "relu"),
                use_feat_norm=config.get("use_feat_norm", False),
            )
        elif model_name.lower() == "pointnet":
            model = PointNet(
                use_time=config.get("use_time", True),
                use_attention=config.get("use_attention", False),
                use_residual=config.get("use_residual", False),
                activation=config.get("activation", "relu"),
                use_feat_norm=config.get("use_feat_norm", False),
            )
        else:
            raise ValueError("Unsupported model type")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

        model_tag = f"{model_name}_rgb{config.get('use_rgb', False)}_time{config.get('use_time', True)}"
        save_path = os.path.join(results_dir, "saved_models", model_tag)

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=nn.MSELoss(),
            scheduler=scheduler,
            device=device,
            epochs=config.get("epochs", 100),
            save_path=save_path,
            model_name=model_tag,
            use_time=config.get("use_time", True)
        )

        pred_csv = os.path.join(results_dir, "predictions", model_tag + ".csv")
        mse, r2 = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            save_path=pred_csv,
            use_time=config.get("use_time", True)
        )
        print(f"✅ [{model_tag}] 测试 MSE: {mse:.4f} | R2: {r2:.4f}")

        plot_from_csv(pred_csv)