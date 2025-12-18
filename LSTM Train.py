import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import csv
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

# 初始化 TensorBoard
writer = SummaryWriter("runs/rice_growth")
log_dir = "tensorboard_logs"
os.makedirs(log_dir, exist_ok=True)


# 自适应加权损失函数
class AdaptiveLossWeighting(nn.Module):
    def __init__(self):
        super(AdaptiveLossWeighting, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))  # 两个损失项的可训练参数

    def forward(self, ms_loss, lidar_loss):
        ms_weight = torch.exp(-self.log_vars[0])
        lidar_weight = torch.exp(-self.log_vars[1])
        loss = ms_weight * ms_loss + lidar_weight * lidar_loss + self.log_vars.sum()+4
        return loss


# 自定义数据加载器
class RiceGrowthDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        self.time_steps = 10

        # 提取光谱数据（NDVI、LCI、GNDVI、OSAVI、NDRE）
        spectral_features = ['NDVI', 'LCI', 'GNDVI', 'OSAVI', 'NDRE']
        self.spectral_columns = [f"{feature}_t{t}" for feature in spectral_features for t in range(1, 11)]

        # 提取LiDAR数据（株高、分蘖、叶龄）
        lidar_features = ['株高', '分蘖', '叶龄']
        self.lidar_columns = [f"{feature}_t{t}" for feature in lidar_features for t in range(1, 11)]

        # 提取目标变量
        self.target_column = "early_growth_score"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ms_data = self.data.iloc[idx][self.spectral_columns].values.reshape(self.time_steps, 5).astype(np.float32)
        lidar_data = self.data.iloc[idx][self.lidar_columns].values.reshape(self.time_steps, 3).astype(np.float32)
        target = self.data.iloc[idx][self.target_column].astype(np.float32)
        return torch.tensor(ms_data), torch.tensor(lidar_data), torch.tensor([target])

# 定义时间注意力机制
class TimeAttention(nn.Module):
    def __init__(self, input_dim):
        super(TimeAttention, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.attention_weights(x))
        scores = torch.softmax(scores, dim=1)
        weighted = x * scores
        return torch.sum(weighted, dim=1), scores


# 定义 LSTM 分支
class LSTMBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TimeAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out)
        return F.relu(self.fc(attn_out)), attn_weights


# 定义多模态 LSTM 模型
class MultimodalLSTM(nn.Module):
    def __init__(self, ms_input_dim, lidar_input_dim, hidden_dim, output_dim):
        super(MultimodalLSTM, self).__init__()
        self.ms_branch = LSTMBranch(ms_input_dim, hidden_dim, output_dim)
        self.lidar_branch = LSTMBranch(lidar_input_dim, hidden_dim, output_dim)
        self.final_fc = nn.Linear(output_dim * 2, 1)

    def forward(self, ms_data, lidar_data):
        ms_features, ms_attention = self.ms_branch(ms_data)
        lidar_features, lidar_attention = self.lidar_branch(lidar_data)
        merged = torch.cat((ms_features, lidar_features), dim=1)
        return torch.sigmoid(self.final_fc(merged)), ms_attention, lidar_attention

def compute_rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))

# 初始化模型
ms_input_dim = 5  # 光谱特征数
lidar_input_dim = 3  # LiDAR特征数
time_steps = 10
hidden_dim = 64
output_dim = 32

model = MultimodalLSTM(ms_input_dim, lidar_input_dim, hidden_dim, output_dim)
loss_function = AdaptiveLossWeighting()
# optimizer = optim.Adam(list(model.parameters()) + list(loss_function.parameters()), lr=0.001)
optimizer = optim.AdamW(list(model.parameters()) + list(loss_function.parameters()), lr=0.001, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # 每50个 epoch 学习率下降一半

# 加载训练和测试数据
dataset_train = RiceGrowthDataset("L:\杨非凡pointnet项目\早生快发模型/rice_early_growth_template_per_sample.xlsx")
dataset_test = RiceGrowthDataset("L:\杨非凡pointnet项目\早生快发模型/rice_early_growth_template_per_sample - 重复区.xlsx")

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)




# 训练函数
def train_model(model, dataloader, optimizer, loss_function, epochs=3000, save_path="saved_models"):
    os.makedirs(save_path, exist_ok=True)  # 确保保存目录存在
    log_data = []  # 用于存储 TensorBoard 记录的数据
    global writer  # 确保 writer 变量可用

    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0
    model.train()

    for epoch in range(epochs):
        total_loss, ms_loss, lidar_loss = 0, 0, 0
        total_rmse, ms_rmse, lidar_rmse = 0, 0, 0

        for batch_idx, (ms_data, lidar_data, targets) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs, ms_attention, lidar_attention = model(ms_data, lidar_data)
            ms_pred, _ = model.ms_branch(ms_data)
            lidar_pred, _ = model.lidar_branch(lidar_data)

            ms_loss_value = F.mse_loss(ms_pred, targets)
            lidar_loss_value = F.mse_loss(lidar_pred, targets)
            loss = loss_function(ms_loss_value, lidar_loss_value)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ms_loss += ms_loss_value.item()
            lidar_loss += lidar_loss_value.item()

            # 计算 RMSE
            total_rmse += compute_rmse(outputs, targets).item()
            ms_rmse += compute_rmse(ms_pred, targets).item()
            lidar_rmse += compute_rmse(lidar_pred, targets).item()

            # # 记录损失和 RMSE 到 TensorBoard
            # writer.add_scalar("Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx)
            # writer.add_scalar("Loss/MS", ms_loss_value.item(), epoch * len(dataloader) + batch_idx)
            # writer.add_scalar("Loss/LiDAR", lidar_loss_value.item(), epoch * len(dataloader) + batch_idx)
            # writer.add_scalar("RMSE/Total", compute_rmse(outputs, targets).item(), epoch * len(dataloader) + batch_idx)
            # writer.add_scalar("RMSE/MS", compute_rmse(ms_pred, targets).item(), epoch * len(dataloader) + batch_idx)
            # writer.add_scalar("RMSE/LiDAR", compute_rmse(lidar_pred, targets).item(),
            #                   epoch * len(dataloader) + batch_idx)
            # writer.add_histogram("Attention/MS", ms_attention.detach().cpu().numpy(),
            #                      epoch * len(dataloader) + batch_idx)
            # writer.add_histogram("Attention/LiDAR", lidar_attention.detach().cpu().numpy(),
            #                      epoch * len(dataloader) + batch_idx)

            # 记录时间步注意力权重
            if batch_idx == 0:  # 只打印第一个 batch
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}")
                print(f"MS Attention: {ms_attention.mean(dim=0).detach().cpu().numpy()}")  # 解决 requires_grad=True 的问题
                print(f"LiDAR Attention: {lidar_attention.mean(dim=0).detach().cpu().numpy()}")

            # 记录 t1-t10 注意力到 TensorBoard
            attention_values = {f"Attention/MS_t{t + 1}": ms_attention[:, t].mean().item() for t in range(10)}
            attention_values.update(
                {f"Attention/LiDAR_t{t + 1}": lidar_attention[:, t].mean().item() for t in range(10)})
            log_entry = {"Epoch": epoch + 1, "Batch": batch_idx + 1, "Loss_Total": loss.item(), **attention_values}
            log_data.append(log_entry)

            for key, value in attention_values.items():
                writer.add_scalar(key, value, epoch * len(dataloader) + batch_idx)

            # 记录损失和 RMSE 到 TensorBoard
            writer.add_scalar("Loss/Total", loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/MS", ms_loss_value.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("Loss/LiDAR", lidar_loss_value.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("RMSE/Total", compute_rmse(outputs, targets).item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("RMSE/MS", compute_rmse(ms_pred, targets).item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar("RMSE/LiDAR", compute_rmse(lidar_pred, targets).item(),
                              epoch * len(dataloader) + batch_idx)
            writer.add_histogram("Attention/MS", ms_attention.detach().cpu().numpy(),
                                 epoch * len(dataloader) + batch_idx)
            writer.add_histogram("Attention/LiDAR", lidar_attention.detach().cpu().numpy(),
                                 epoch * len(dataloader) + batch_idx)

            # 记录时间步注意力权重曲线
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), ms_attention.mean(dim=0).detach().cpu().numpy(), label='MS Attention', marker='o')
            ax.plot(range(1, 11), lidar_attention.mean(dim=0).detach().cpu().numpy(), label='LiDAR Attention', marker='s')
            ax.set_xlabel("Time Steps (t1-t10)")
            ax.set_ylabel("Attention Weight")
            ax.set_title(f"Attention Weights at Epoch {epoch+1}, Batch {batch_idx+1}")
            ax.legend()
            writer.add_figure("Attention Weights", fig, epoch * len(dataloader) + batch_idx)
            plt.close(fig)

        epoch_avg_rmse = total_rmse / len(dataloader)

        # 更新最佳模型逻辑
        if epoch_avg_rmse < best_rmse:
            best_rmse = epoch_avg_rmse
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            print(f"New best model at epoch {best_epoch} with RMSE: {best_rmse:.4f}")


        print(
            f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(dataloader):.4f}, RMSE: {total_rmse / len(dataloader):.4f}")

        # 每个 epoch 结束后保存模型
        model_save_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        # 保存最佳模型到独立文件
        if best_model_state is not None:
            best_model_path = os.path.join(save_path, "best_model_rmse.pth")
            torch.save(best_model_state, best_model_path)
        scheduler.step()  # 更新学习率
        print(f"Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}")
        print(f"Model saved at {model_save_path}")

    # 保存 TensorBoard 数据到 CSV 文件
    log_file = os.path.join(log_dir, "tensorboard_logs.csv")
    with open(log_file, mode='w', newline='') as file:
        writer_csv = csv.DictWriter(file, fieldnames=log_data[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(log_data)
    print(f"TensorBoard logs exported to {log_file}")


# 训练模型
train_model(model, dataloader_train, optimizer, loss_function)
print("Run 'tensorboard --logdir=runs/rice_growth' in the terminal to visualize logs.")

