# 第一问

`data` 文件夹以及 `output` 那几个 `excel` 都是程序自动生成，请不要随意删除，若需运行程序请一定将终端移动到该文件夹下，否则将找不到数据文件，因为代码使用的是相对路径！！！

第一问写的有点屎山……不如看看[第二问](../第二问/README.md)和[第三问](../第三问/README.md)的~

## 代码运行顺序
1. 运行 `find_first_point_history.py`，将会生成 `龙头的x y数据.xlsx`，`龙头的r数据.csv`，`龙头的theta数据.csv` 三个文件在当前目录的 `data` 文件夹下（没有的话会自动创建）
2. 运行 `find_second_point_history.py`，将会读取 `data` 文件夹中的 `龙头的theta数据.csv`，之后生成 `output.xlsx` 在当前文件夹下
3. 需要手动打开 `data` 文件夹下 `龙头的theta数据.csv` 文件，选择第一行，手动将数据拷贝到 `output.xlsx` 的第一行（不然数据不是完整的，后面的程序运行不了）
4. 运行 `find_other_points_history.py`，将读取 `output.xlsx` 中前两个把手的theta信息，生成所有点位的theta信息，将结果保存在当前目录下的 `output1.xlsx`
5. 运行 `find_velocity.py`，将读取 `output1.xlsx` 中各点位的theta信息，生成各点位的速度信息，将结果保存在当前目录下的 `output2.xlsx`
6. 运行 `velocity_fill_na.py`，将读取 `output2.xlsx`，填补缺失值之后将结果保存在当前目录下的 `output3.xlsx`
7. 运行 `theta_to_xy.py`，将读取 `output1.xlsx`，将theta值转换为xy值之后保存在当前目录下的 `output4.xlsx`
8. 运行 `xy_fill_na.py`，将读取 `output4.xlsx`，填补缺失值之后将结果保存在当前目录下的 `output5.xlsx`
9. 手动将 `output3.xlsx` `output5.xlsx` 中的结果拷贝到 `result1.xlsx` 里边去

## 注

- 由于代码的更新迭代，部分脚本运行会出现 `FutureWarning` 等字样，这是正常现象！！！
- 请一定按顺序进行运行！！！否则出现无法复现论文结果的情况！！！
- 为了减小支撑材料的大小，已将所有的字体文件删去，因此部分plot中将出现中文字体无法显示的情况！！！请谅解！！！