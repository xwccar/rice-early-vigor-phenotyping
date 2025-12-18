import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 读取 Excel 文件
df = pd.read_excel("得分汇总.xlsx", sheet_name='Sheet2')  # 替换为你的文件路径

# 选择输入特征（去除前两列编号）和目标
X = df.iloc[:, 2:-1].values
y = df.iloc[:, -1].values

# 归一化输入数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# **增强 t1-t5 的影响**
# 找到 t1-t5 相关的特征索引
t4_t6_features = [i for i, col in enumerate(df.columns[2:-1]) if any(f"_t{t}" in col for t in range(4, 7))]
X[:, t4_t6_features] *= 4  # 放大 t1-t5 特征值

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 MLPRegressor 模型
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                   max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# 预测
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

# 计算 R² 和 RMSE
r2_train = r2_score(y_train, y_train_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"训练集 R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}")
print(f"测试集 R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

# **使用 SHAP 计算特征重要性**
def model_predict(X):
    return mlp.predict(X)

explainer = shap.KernelExplainer(model_predict, X_train[:50])  # 只取部分数据作为背景集，提升计算速度
shap_values = explainer.shap_values(X_test[:200])  # 计算前 200 个样本的 SHAP 值

# 计算 SHAP 重要性
shap_importance = np.abs(shap_values).mean(axis=0)

# 归一化 SHAP 重要性（不按时间点归一化）
shap_importance_normalized = shap_importance / shap_importance.sum()

# 创建 SHAP 重要性 DataFrame
shap_feature_importance_df = pd.DataFrame({
    "Feature": df.columns[2:-1],
    "SHAP_Importance": shap_importance_normalized
})

# 按 t1-t10 排序
time_points = [f"_t{i}" for i in range(1, 11)]
sorted_features = sorted(shap_feature_importance_df["Feature"], key=lambda x: next((i for i, t in enumerate(time_points) if t in x), 999))

shap_feature_importance_df_sorted = shap_feature_importance_df.set_index("Feature").loc[sorted_features].reset_index()

# 导出 SHAP 重要性结果
shap_feature_importance_df_sorted.to_excel("SHAP_Feature_Importance-MS.xlsx", index=False)
print("SHAP 特征重要性已保存为 SHAP_Feature_Importance.xlsx")
