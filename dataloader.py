import os
import laspy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RiceTillerDataset(Dataset):
    def __init__(self, root_dir, label_files_dict, day_mapping, use_rgb=False, use_time=True):
        """
        root_dir: 主目录，包含多个日期子文件夹（如0806、0810等）
        label_files_dict: dict，key为文件夹名，value为上传的Excel文件对象
        day_mapping: dict，key为文件夹名，value为插秧后天数
        use_rgb: 是否使用RGB特征
        use_time: 是否加入“插秧后天数”特征
        """
        self.use_rgb = use_rgb
        self.use_time = use_time
        self.samples = []

        for folder_name, label_file in label_files_dict.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            try:
                labels_df = pd.read_excel(label_file)
                labels_df.columns = [col.strip() for col in labels_df.columns]
            except Exception as e:
                print(f"❌ 无法读取标签表格 {folder_name}: {e}")
                continue

            days_after = day_mapping.get(folder_name, None)
            if days_after is None:
                print(f"⚠️ 缺少 {folder_name} 的插秧后天数，跳过")
                continue

            for file in os.listdir(folder_path):
                if not file.endswith('.las'):
                    continue
                try:
                    A, B = map(int, os.path.splitext(file)[0].split('-'))
                    match = labels_df[(labels_df["A"] == A) & (labels_df["B"] == B)]
                    if not match.empty and pd.notna(match["提取的数据"].values[0]):
                        label = float(match["提取的数据"].values[0])
                        self.samples.append((os.path.join(folder_path, file), label, days_after))
                except Exception as e:
                    print(f"⚠️ 无法解析文件 {file}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, time_value = self.samples[idx]
        las = laspy.read(path)
        points = np.vstack([las.x, las.y, las.z]).T

        if self.use_rgb:
            rgb = np.vstack([las.red, las.green, las.blue]).T
            points = np.concatenate((points, rgb), axis=1)

        # 中心化
        centroid = np.mean(points[:, :3], axis=0)
        points[:, :3] -= centroid

        points = torch.tensor(points, dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        time_tensor = torch.tensor([time_value], dtype=torch.float32)

        if self.use_time:
            return points, time_tensor, label
        else:
            return points, label
