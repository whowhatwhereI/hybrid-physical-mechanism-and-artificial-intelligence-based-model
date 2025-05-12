import os
import sys
import stat

# 创建目录
def make_dir(path: str):
    if os.path.isdir(path) is True:
        print("已有数据, 退出项目")
        sys.exit()
    else:
        os.mkdir(path)


# 写入csv数据
def write_statistic_data(path: str,
                         lists: list):
    with open(path, "a+") as f:
        f.write(",".join([str(i) for i in lists]) + "\n")
    f.close()


# 写入日志
def write_log(path: str,
              string: str):
    # print(string)
    with open(path, "a") as f:
        f.write(string + "\n")


# 清空文件夹
def remove_all(file_dir: str):
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        file_path = os.path.join(file_dir, file_name)
        os.remove(file_path)


# 限制范围
def clamp_value(value, min_value, max_value):
    value_return = value
    if value < min_value:
        value_return = min_value
    elif value > max_value:
        value_return = max_value
    return value_return
