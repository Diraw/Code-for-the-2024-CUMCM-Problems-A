import math
import pandas as pd
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

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
    upper = theta1 + np.pi
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


def find_enter_sec(data, point):
    first_point_theta = data.iloc[0]
    prevous_point_history = data.iloc[point]
    target_distance = 1.65
    start = prevous_point_history.first_valid_index()
    for sec in range(len(first_point_theta)):
        prevous_point = prevous_point_history[sec + start]
        first_point = first_point_theta[sec]
        arc0 = arc_length(32 * np.pi, first_point)
        if is_enough_distance_to_add_new_point(prevous_point, target_distance):
            break
    return sec + start


def find_point_history(data_1, sec, point):
    list = []  # 储存这一点每秒的数据
    prevous_point_history = data_1.iloc[point]
    for prevous_point in prevous_point_history[sec:]:
        new_point_theta = fine_theta_for_distance(
            prevous_point, target_distance=1.65, tol=TOL
        )
        list.append(new_point_theta)
    list_0 = [pd.NA] * sec
    list = list_0 + list
    return list


def new_data(result_list, data):
    data = data.iloc[:2]
    result_list = pd.DataFrame(result_list, columns=data.columns)
    data = pd.concat([data, result_list], ignore_index=True)
    return data


if __name__ == "__main__":
    data = pd.read_excel("./output.xlsx", header=0, index_col=0)
    result_list = []  # 储存最终的数据
    for point in tqdm(range(1, 180)):  # 从第1行循环到223行，依次计算所有点
        sec = find_enter_sec(data, point)  # 计算该点的进场时刻
        list = find_point_history(data, sec, point)  # 计算该点的历史轨迹
        result_list.append(list)
        data = new_data(result_list, data)

    # 使用 pandas 将列表写入 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存到 Excel 文件
    df.to_excel("output1.xlsx", index=False, header=False)
