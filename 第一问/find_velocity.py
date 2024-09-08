import numpy as np
import pandas as pd
import math
from itertools import zip_longest
from tqdm import tqdm

TOL = 1e-10

def velocity(v_prevous, alpha, beta):
    v = (math.cos(alpha) * v_prevous) / (math.cos(beta))
    return abs(v)


def find_alpha_2(theta_prevous, theta):
    r_prevous = (0.55 * theta_prevous) / (2 * np.pi)
    r = (0.55 * theta) / (2 * np.pi)
    tan_alpha_2 = (r_prevous * math.sin(theta_prevous) - r * math.sin(theta)) / (
        r_prevous * math.cos(theta_prevous) - r * math.cos(theta)
    )
    alpha_2 = math.atan(tan_alpha_2)
    if alpha_2 < 0:
        alpha_2 = alpha_2 + np.pi
    return alpha_2


def find_alpha(theta_prevous, theta):
    tan_alpha_1 = (
        math.sin(theta_prevous) + theta_prevous * math.cos(theta_prevous)
    ) / (math.cos(theta_prevous) - theta_prevous * math.sin(theta_prevous))
    alpha_1 = math.atan(tan_alpha_1)
    if alpha_1 < 0:
        alpha_1 = alpha_1 + np.pi

    alpha_2 = find_alpha_2(theta_prevous, theta)

    alpha = alpha_1 - alpha_2

    return alpha


def find_beta(theta_prevous, theta, l=1.65):
    tan_phi = (math.sin(theta) + theta * math.cos(theta)) / (
        math.cos(theta) - theta * math.sin(theta)
    )
    phi = math.atan(tan_phi)
    if phi < 0:
        phi = phi + np.pi

    phi2 = find_alpha_2(theta_prevous, theta)

    # print(theta % (2 * np.pi))
    beta = abs(phi - phi2)
    # print("phi, gamma, theta_0: ", phi, gamma)
    return beta


def column_generator(file_path):
    df = pd.read_excel(file_path, header=None, index_col=None)
    columns = df.columns
    for column in columns:
        yield df[column]


def add_list_to_df(df, new_list):
    # 将现有的 DataFrame 转换为列表
    existing_data = df.values.T.tolist()
    # 将新的列表添加到现有数据中
    combined_data = list(zip_longest(*existing_data, new_list, fillvalue=None))
    # 创建新的 DataFrame
    new_df = pd.DataFrame(combined_data)
    return new_df


if __name__ == "__main__":
    column_gen = column_generator("./output1.xlsx")
    df = pd.DataFrame()  # 初始化df用于储存各点的速度
    for column_data in column_gen:
        v = 1  # 初始化龙头的速度为1m/s
        # print(column_data)
        list_0 = []  # list用于储存每一秒各点的速度
        list_0.append(1)
        for theta_prevous, theta in zip(column_data, column_data[1:]):
            # print(theta_prevous, theta)
            if math.isnan(theta):
                break
            alpha = find_alpha(theta_prevous, theta)
            beta = find_beta(theta_prevous, theta, l=1.65)
            v = velocity(v, alpha, beta)
            # print("v: ", v)
            # print("alpha, beta: ", alpha, beta)
            list_0.append(v)
        df = add_list_to_df(df, list_0)
        # print(df)
    df.to_excel("output2.xlsx", index=False, header=False)
