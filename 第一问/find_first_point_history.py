import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import font_manager

TOL = 1e-10

def arc_length(theta1, theta2):
    # 计算两个点之间的弧长
    return (0.55 * quad(lambda theta: np.sqrt(theta**2 + 1), theta2, theta1)[0]) / (
        2 * np.pi
    )


def find_theta_for_arc_length(theta1, target_length=1, tol=TOL):
    # 对于龙头，利用二分法找到满足弧长的点，这些点就是龙头的历史点位
    lower = 0
    upper = theta1
    while upper - lower > tol:
        mid = (lower + upper) / 2
        length = arc_length(theta1, mid)

        if length < target_length:
            upper = mid
        else:
            lower = mid
    return (lower + upper) / 2


def show_plot(theta_values, r_values, time_steps):
    # 加载字体文件
    font_path = "./SimHei.ttf"
    font_prop = font_manager.FontProperties(fname=font_path)
    # 画出龙头的历史轨迹图
    plt.figure(figsize=(10, 6))
    plt.style.use("seaborn-white")
    ax = plt.subplot(111, polar=True)
    ax.plot(theta_values, r_values, label="龙头的移动轨迹")
    ax.scatter(
        theta_values,
        r_values,
        color="red",
        s=10,
        zorder=5,
    )
    ax.legend(prop=font_prop)

    plt.show()


def calculate_coordinates(theta, r):
    # 将theta r转化为x y坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def output_to_excel(file_path="output.xlsx", *arrays):
    # 将数据输入指定路径的excel文件
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    data = pd.DataFrame(arrays)
    data.to_excel(file_path, header=False, index=False)
    print(f"文件已成为保存至{file_path}")


def output_to_csv(file_path, *arrays):
    # 将数据输入指定路径的CSV文件
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    data = pd.DataFrame(arrays)
    data.to_csv(file_path, header=False, index=False)
    print(f"文件已成为保存至{file_path}")


def plot_curve(values):
    # 画出转换之后的xy，作为验证
    plt.figure()
    plt.plot(range(len(values)), values, marker="o")
    plt.grid(True)
    plt.show()


def find_first_point_history():
    # 初始条件
    theta1 = 32 * np.pi
    time_steps = 300
    target_length_per_second = 1

    # 计算每秒的坐标
    theta_values = [theta1]
    for _ in range(time_steps):
        next_theta = find_theta_for_arc_length(
            theta_values[-1], target_length_per_second
        )
        theta_values.append(next_theta)

    # output_to_excel(theta_values) # 查看theta角的正确性
    r_values = [(theta * 0.55) / (2 * np.pi) for theta in theta_values]
    x_values, y_values = calculate_coordinates(theta_values, r_values)
    # print(x_values, y_values)
    output_to_excel(
        "./data/龙头的x y数据.xlsx", x_values, y_values
    )  # 保存龙头的x y数据
    output_to_csv(
        "./data/龙头的r数据.csv", r_values
    )  # 保存龙头的r数据，用于下个程序求龙身的数据
    output_to_csv(
        "./data/龙头的theta数据.csv", theta_values
    )  # 保存龙头的theta数据，用于下个程序求龙身的数据

    plot_curve(x_values)
    plot_curve(y_values)
    show_plot(theta_values, r_values, time_steps)


if __name__ == "__main__":
    find_first_point_history()
