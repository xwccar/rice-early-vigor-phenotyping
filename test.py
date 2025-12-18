import os
path = r"L:\杨非凡pointnet项目\代码\dataset\train"
if os.path.exists(path):
    print("路径存在")
    print("子文件夹：", os.listdir(path))
else:
    print("路径不存在")