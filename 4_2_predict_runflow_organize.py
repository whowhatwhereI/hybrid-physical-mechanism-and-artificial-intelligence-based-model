import datetime

import numpy as np
import os
import pandas as pd
import torch

from _parameters import parameters
from _method import write_statistic_data, write_log
from _algorithm import RunflowPredictDataset, Runflow

# 创建用于输出径流量的 降雨数据序列 1 - 216
if __name__ == '__main__':
    # rainfall
    predicted_runflow_output_dir = parameters.predicted_runflow_output_dir
    predicted_runflow_output_1_dir = os.path.join(predicted_runflow_output_dir, "1_rainfall")
    predicted_runflow_output_1_names = "rainfall_series_{}.csv"
    # area
    area_dir = parameters.area_dir
    area_path = os.path.join(area_dir, "area.csv")
    # area_data = pd.read_csv(area_path, delimiter=",", encoding="utf8", index_col="OBJECTID")
    # predict_output
    predicted_runflow_output_dir = parameters.predicted_runflow_output_dir
    predicted_runflow_output_2_dir = os.path.join(predicted_runflow_output_dir, "2_runflow")
    predicted_runflow_output_2_names = "runflow_{}.csv"

    # Runflow
    train_output_dir = parameters.train_output_dir
    train_output_3_dir = os.path.join(train_output_dir, "3_train_model_files")
    train_output_3_path = os.path.join(train_output_3_dir, "epoch60.pth")
    runflow = Runflow(1, 1, 20)
    state_dict = torch.load(train_output_3_path)
    runflow.load_state_dict(state_dict)
    # for name, parameters in runflow.named_parameters():
    #     print(name, parameters)

    # 遍历 每个区域
    # a = [870, 3718, 3716]
    for area_id in range(1, 3722 + 1):
        # output path
        predicted_runflow_output_2_name = predicted_runflow_output_2_names.format(str(area_id))
        predicted_runflow_output_2_path = os.path.join(predicted_runflow_output_2_dir, predicted_runflow_output_2_name)
        # model
        runflowPredictDataset = RunflowPredictDataset(area_path=area_path,
                                                      rainfall_dir=predicted_runflow_output_1_dir,
                                                      rainfall_name=predicted_runflow_output_1_names,
                                                      area_id=area_id)
        # write
        write_statistic_data(predicted_runflow_output_2_path, ["time", "runflow", "prec", "id"])
        # 遍历 每个区域的 每一场雨
        for i in range(len(runflowPredictDataset)):
            target, times_str = runflowPredictDataset[i]
            # print(target)
            # print(times_str)
            output = runflow(target, 60)
            output = output.reshape(len(output)).detach().numpy()
            # print("output", output.shape)
            # print(output)
            # write
            for row in range(len(output)):
                prec = target[row, 6].detach().item()
                time = times_str[row]
                pred_runflow = output[row]
                write_statistic_data(predicted_runflow_output_2_path, [str(time), str(pred_runflow), str(prec), str(i)])
