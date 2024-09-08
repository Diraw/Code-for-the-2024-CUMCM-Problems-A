import pandas as pd

df = pd.read_excel("./output4.xlsx", header=None)
# print(df)
df = df.astype(float)
# 填充到 448 行
df = df.reindex(range(448))

# 遍历每一列
for col in df.columns:
    # 获取非空值
    last_two_values = df[col].dropna().iloc[-2:]

    if len(last_two_values) < 2:
        continue

    # 奇数行填充倒数第二个值
    odd_fill_value = last_two_values.iloc[0]

    # 偶数行填充等差数列
    even_start_value = last_two_values.iloc[1]
    difference = 1.65
    # print(odd_fill_value, even_start_value)

    for i in range(len(df)):
        if pd.isna(df.at[i, col]):
            if i % 2 == 0:
                df.at[i, col] = odd_fill_value
            else:
                df.at[i, col] = even_start_value + (i // 2) * difference
            # print(df)

# 控制输出精度为小数点后六位
df = df.round(6)
df.to_excel("output5.xlsx", index=False, header=False)
