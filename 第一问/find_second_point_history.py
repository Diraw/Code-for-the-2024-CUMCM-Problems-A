import math
import pandas as pd
import numpy as np
from scipy.integrate import quad

TOL = 1e-10

def distance(theta1, theta2):
    # 计算两个点之间的距离
    r1 = 0.55 * theta1 / (2 * np.pi)
    r2 = 0.55 * theta2 / (2 * np.pi)
    cos_value = math.cos(float(theta2 - theta1))
    distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * cos_value)
    return float(distance)


def fine_theta_for_distance(theta1, target_distance, tol=TOL):
    # 对于龙身，利用二分法找到满足距离的点
    lower = theta1
    upper = theta1+np.pi
    while upper - lower > tol:
        mid = (lower + upper) / 2
        length = distance(theta1, mid)

        if length < target_distance:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2


def is_enough_distance_to_add_new_point(theta, target_distance, tol=TOL):
    if distance(theta, 36 * np.pi) - target_distance < tol:
        return False
    else:
        return True


def arc_length(theta1, theta2):
    # 计算两个点之间的弧长
    return (0.55 * quad(lambda theta: np.sqrt(theta**2 + 1), theta2, theta1)[0]) / (
        2 * np.pi
    )


def read_from_csv(file_path):
    # 从指定路径的CSV文件读取数据
    data = pd.read_csv(file_path, header=None)
    return data

def add_na_to_list(keyword_list, times):
    for _ in range(times):
        keyword_list.append(pd.NA)

def output_to_df(data_list, column_name):
    global df
    df[column_name] = data_list


def append_to_row(df, row_index, new_data):
    # 获取第 row_index 行的现有数据并转换为列表
    existing_data = df.loc[row_index].tolist()
    # 将新数据追加到现有数据
    combined_data = existing_data + new_data
    # 确保 DataFrame 列数足够
    if len(combined_data) > len(df.columns):
        # 扩展列数
        additional_columns = len(combined_data) - len(df.columns)
        for i in range(additional_columns):
            df[f"new_col_{i+1}"] = pd.NA
    # 更新第 row_index 行的数据
    df.loc[row_index] = combined_data
    return df


def expand_dataframe(original_df, new_rows, new_cols):
    # 创建一个空的 DataFrame，大小为 new_rows x new_cols
    expanded_df = pd.DataFrame(
        np.nan,
        index=range(new_rows),  # 行索引从 0 到 new_rows-1
        columns=range(new_cols),  # 列名从 0 到 new_cols-1
    )
    # 将原始 DataFrame 的数据复制到扩展 DataFrame 中
    for i in range(min(original_df.shape[0], new_rows)):
        for j in range(min(original_df.shape[1], new_cols)):
            expanded_df.iloc[i, j] = original_df.iloc[i, j]
    return expanded_df


def find_second_point_enter(sec):
    # first_point_r_history = read_from_csv("./data/龙头的r数据.csv").iloc[0].values
    first_point_theta_history = (
        read_from_csv("./data/龙头的theta数据.csv").iloc[0].values
    )
    # print(first_point_r_history)
    target_distance = 2.86
    for first_point_theta in first_point_theta_history:
        list=[] # 储存这一秒的各点数据
        list.append(first_point_theta)
        arc0 = arc_length(32 * np.pi, first_point_theta)
        print(
            f"秒数: {np.round(arc0,decimals=0)}",
            "截距: ",
            distance(first_point_theta, 32 * np.pi),
            "是否可以加入新龙身：",
            is_enough_distance_to_add_new_point(first_point_theta, target_distance),
        )
        if is_enough_distance_to_add_new_point(first_point_theta, target_distance):
            new_point = fine_theta_for_distance(
                first_point_theta, target_distance=2.86, tol=TOL
            )
            print("新一节龙身的theta位置为", new_point)
            list.append(new_point)
            add_na_to_list(list, 224 - len(list))
            output_to_df(list, sec)
            sec = sec + 1
            break
        add_na_to_list(list, 224-len(list))
        output_to_df(list, sec) # 加入一列的数据
        # print(list,len(list))
        # print(df)
        sec=sec+1
    return sec

def find_second_point_history(new_point_sec):
    first_point_theta_history = (
        read_from_csv("./data/龙头的theta数据.csv").iloc[0].values
    )
    list = []  # 储存这一点每秒的数据
    for first_point_theta in first_point_theta_history[new_point_sec:]:
        new_point_theta = fine_theta_for_distance(
            first_point_theta, target_distance=2.86, tol=TOL
        )
        list.append(new_point_theta)
    append_to_row(df, 1, list)

if __name__ == "__main__":
    df = pd.DataFrame()  # 全局df，往下是各节点，往右是秒数
    sec=0 # 全局sec，储存现在是第几秒的状态
    new_point_sec=find_second_point_enter(sec) # 找到第二个点的开始时刻，返回进场时间
    find_second_point_history(new_point_sec) # 在第二个点进场之后，其余点均可直接用截距找到
    df = expand_dataframe(df, 224, 301)
    # print(new_point_sec)
    # print(df)
    # df.to_excel("output.xlsx", index=False, header=False)
    df.to_excel("output.xlsx")
