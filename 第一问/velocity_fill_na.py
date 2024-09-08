import pandas as pd

df = pd.read_excel("./output2.xlsx", header=None, index_col=None)
df = df.astype(float)

# 使用 fillna 方法和 ffill（前向填充）来填充缺失值
df.fillna(method="ffill", inplace=True)

# 控制输出精度为小数点后六位
df = df.round(6)

# 将 DataFrame 扩展到 224 行
df = df.reindex(range(224)).fillna(method="ffill")

# 保存到新的 Excel 文件
df.to_excel("output3.xlsx", index=False, header=False)
