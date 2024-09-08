import pandas as pd
import numpy as np


# 读取 Excel 文件
df = pd.read_excel("./output1.xlsx", header=None)


# 定义极坐标转换为直角坐标的函数
def polar_to_cartesian(theta):
    r = (0.55 * theta) / (2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


# 创建一个列表用于存储结果
result_list = []

for index, row in df.iterrows():
    # 计算每一行的 x, y 坐标
    x_values, y_values = polar_to_cartesian(row)

    # 添加到结果列表中
    result_list.append(pd.Series(x_values))
    result_list.append(pd.Series(y_values))


# 将结果列表转换为 DataFrame
result = pd.concat(result_list, axis=1).T.reset_index(drop=True)

# 保存到新的 Excel 文件
result.to_excel("output4.xlsx", index=False, header=False)
