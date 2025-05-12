import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

from _method import clamp_value

torch.set_default_tensor_type(torch.DoubleTensor)


# 数据归一化 - [0, 1]
class BatchNorm1d(nn.Module):  # 仅标准化一个输入项, 不支持动态修改 【n, 1, 1】
    def __init__(self):
        super(BatchNorm1d, self).__init__()

    def forward(self, x):
        q = x  # [n, 1, 1]
        q_min = torch.min(q)
        q_max = torch.max(q)
        if q_max - q_min != 0:
            q = (q - q_min) / (q_max - q_min)
        else:
            q = q - q + 1
            q_min = torch.tensor(0)
        return q, q_max, q_min


# torch - runoff 读取训练数据
class RunflowTrainDataset(nn.Module):
    def __init__(self,
                 series_path: str):
        self.series_data = pd.read_csv(series_path, delimiter=",", index_col="id")
        print(self.series_data)

    def __len__(self):
        return int(self.series_data.index[-1])

    def __getitem__(self, item):
        series_data = self.series_data.loc[item + 1]  # [n, 9]
        series_target_data = series_data.loc[:,
                             ["greenland", "water", "impervious", "area", "slope", "duration", "prec", "total_prec",
                              "area_id"]]  # [n, 9]
        series_label_data = series_data.loc[:, ["flow"]]  # [n, 1]
        series_target_values = series_target_data.values
        series_label_values = series_label_data.values
        series_target_tensor = torch.tensor(series_target_values)
        series_label_tensor = torch.tensor(series_label_values)
        return series_target_tensor, series_label_tensor, item + 1


# torch - runoff 读取预测数据
class RunflowPredictDataset(nn.Module):
    def __init__(self,
                 area_path: str,
                 rainfall_dir: str,
                 rainfall_name: str,
                 area_id: int):
        self.area_data = pd.read_csv(area_path, delimiter=",", index_col="OBJECTID")
        rainfall_id = self.area_data.loc[area_id, "id"]
        print("area_id:", area_id)
        # print("rainfall_id:", rainfall_id)
        rainfall_path = os.path.join(rainfall_dir, rainfall_name.format(str(rainfall_id)))
        self.rainfall_data = pd.read_csv(rainfall_path, delimiter=",", index_col="id")
        # print("rainfall_data", self.rainfall_data)
        # print(self.area_data)
        self.area_data_filter = self.area_data.loc[
            area_id, ["greenland", "water", "impervious", "area", "slope", "blockId"]]  # Series
        # print(self.area_data_filter)
        # 修改特定区域的降雨数据
        if area_id == 870:
            self.rainfall_data["prec"] = self.rainfall_data["prec"] / 2 / 2 / 2 / 2
            self.rainfall_data["total_prec"] = self.rainfall_data["total_prec"] / 2 / 2 / 2 / 2

    def __len__(self):
        return int(self.rainfall_data.index[-1] - 1)

    def __getitem__(self, item):
        series_data = self.rainfall_data.loc[item + 1]  # [n, 9]
        series_target_time = series_data.loc[:, "time"].values
        series_data.insert(loc=4, column="greenland", value=self.area_data_filter["greenland"])
        series_data.insert(loc=5, column="water", value=self.area_data_filter["water"])
        series_data.insert(loc=6, column="impervious", value=self.area_data_filter["impervious"])
        series_data.insert(loc=7, column="area", value=self.area_data_filter["area"])
        series_data.insert(loc=8, column="slope", value=self.area_data_filter["slope"])
        series_data.insert(loc=9, column="area_id", value=self.area_data_filter["blockId"])
        series_target_data = series_data.loc[:,
                             ["greenland", "water", "impervious", "area", "slope", "duration", "prec", "total_prec",
                              "area_id"]]  # [n, 9]
        # print(series_target_data)
        series_target_values = series_target_data.values
        series_target_tensor = torch.tensor(series_target_values)
        return series_target_tensor, series_target_time


# torch - runoff 径流量模拟
class Runflow(nn.Module):
    def __init__(self, input_size, hidden_size, interval_epoch):
        super(Runflow, self).__init__()
        self.interval_epoch = interval_epoch
        # 绿地参数 —— 霍顿下渗  f_t = f_0 + (f_0 - f_c) * e^{-kt}
        self.f_0 = torch.nn.Parameter(torch.tensor([0.9], requires_grad=True))
        self.register_parameter("f_0", self.f_0)
        self.f_c = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.register_parameter("f_c", self.f_c)
        self.k = torch.nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.register_parameter("k", self.k)
        # 不透水面参数 —— 曼宁公式  Q = W * 1.49 / n * (d - d_p)^{5/2} * S^{1/2}
        self.n = torch.nn.Parameter(torch.tensor([0.9], requires_grad=True))
        self.register_parameter("n", self.n)
        self.d_p = torch.nn.Parameter(torch.tensor([0.0001], requires_grad=True))
        self.register_parameter("d_p", self.d_p)
        # 随时间消失
        self.evap = torch.nn.Parameter(torch.tensor([0.29], requires_grad=True))
        self.register_parameter("evap", self.evap)
        self.evap_constant = torch.nn.Parameter(torch.tensor([0.02], requires_grad=True))
        self.register_parameter("evap_constant", self.evap_constant)
        # 过量流量更快流失
        self.missing_restrict = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))
        self.register_parameter("missing_restrict", self.missing_restrict)
        # self.missing_max = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))
        # self.register_parameter("missing_max", self.missing_max)
        self.missing = torch.nn.Parameter(torch.tensor([0.2], requires_grad=True))
        self.register_parameter("missing", self.missing)
        # GRU
        self.w_r_1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_r_1", self.w_r_1)
        self.w_r_2 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_r_2", self.w_r_2)
        self.w_z_1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_z_1", self.w_z_1)
        self.w_z_2 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_z_2", self.w_z_2)
        self.w_1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_1", self.w_1)
        self.w_2 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.register_parameter("w_2", self.w_2)
        # 计算
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.gru_ = torch.nn.GRU(1, 1)
        self.BatchNorm1d = BatchNorm1d()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        # self.lstm_green = nn.LSTM(input_size=input_size,
        #                           hidden_size=hidden_size,
        #                           num_layers=1)
        # self.lstm_imper = nn.LSTM(input_size=input_size,
        #                           hidden_size=hidden_size,
        #                           num_layers=1)
        # self.lstm = nn.LSTM(input_size=input_size,
        #                     hidden_size=hidden_size,
        #                     num_layers=1,
        #                     bias=True)

    def forward(self, x, epoch):
        # 禁止物理过程训练
        if epoch > self.interval_epoch:
            self.f_0.requires_grad = False
            self.f_c.requires_grad = False
            self.k.requires_grad = False
            self.n.requires_grad = False
            self.d_p.requires_grad = False
            self.evap.requires_grad = False
            self.evap_constant.requires_grad = False
            self.missing_restrict.requires_grad = False
            self.missing.requires_grad = False

        # # self.missing_max.data = torch.clamp(self.missing_max.data, 1.0, 2.0)
        # print("epoch", epoch)
        # print("f_0", self.f_0)
        # print("f_c", self.f_c)
        # print("k", self.k)
        # print("n", self.n)
        # print("d_p", self.d_p)
        # print("evap", self.evap)
        # print("evap_constant", self.evap_constant)
        # print("missing_restrict", self.missing_restrict)
        # print("missing", self.missing)
        # 提取 [n]
        greenland = x[:, 0]
        water = x[:, 1]
        impervious = x[:, 2]
        area = x[:, 3]
        slope = x[:, 4]
        duration = x[:, 5]
        prec = x[:, 6]
        total_prec = x[:, 7]
        area_id = x[0, 8]
        # 计算在当前时间前积蓄下的流量
        prec_pre = total_prec - prec
        for i in range(1, len(prec_pre)):
            last_prec = prec_pre[i - 1]
            if last_prec > self.missing_restrict:
                last_prec = self.missing_restrict + (last_prec - self.missing_restrict) * self.missing
            prec_pre[i] = (last_prec + prec[i - 1]) * (1 - self.evap) - self.evap_constant
        prec_pre = self.relu(prec_pre)

        # 绿地
        greenland_area = greenland * area
        f_t = (self.f_0 - self.f_c) * (np.e ** -(self.k * total_prec)) + self.f_c
        q_greenland = prec_pre / 1000 * greenland_area * (1 - f_t)  # mm -> m
        q_greenland = self.relu(q_greenland)
        # 不透水面
        impervious_area = impervious * area
        d_impervious = (prec_pre / 1000) - self.d_p
        d_impervious = self.relu(d_impervious)
        q_impervious = torch.sqrt(impervious_area) * (1.49 / self.n) * d_impervious ** (5 / 3) * torch.sqrt(slope)
        q_impervious = self.relu(q_impervious)
        q_impervious = q_impervious.reshape(q_impervious.shape[0], 1, 1)
        q_greenland = q_greenland.reshape(q_greenland.shape[0], 1, 1)

        q = (q_impervious + q_greenland)  # m^2
        # # q = (self.lstm_imper(q_impervious)[0] + self.lstm_green(q_greenland)[0])  # m^2
        # # print("f_t", f_t)
        # # print("total_prec", total_prec)
        # # print("d_impervious", d_impervious)
        # # print("prec", prec)
        # # print("prec_pre", prec_pre)
        # print("q_greenland", q_greenland)
        # print("q_impervious", q_impervious)

        # output, (h, c) = self.lstm(q)
        # q, q_var, q_mean = self.BatchNorm1d(q)
        # output[output < 0] = 0
        # q = q * q_var + q_mean
        if str(round(area_id.detach().item())) == "3723" or str(round(area_id.detach().item())) == "870":
            q = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)
        # elif str(round(area_id.detach().item())) == "3718" or str(round(area_id.detach().item())) == "3716":
        else:
            q = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)
            q = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)
            q = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)
            q = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)
        if epoch > self.interval_epoch:
            # h = torch.cat((torch.tensor([[[0]]]), q[:-1, :, :]), 0)  # i 前一个时间点
            h_t = torch.tensor([[[0.0]]], requires_grad=True)  # 记录信息
            q2 = q[0, :, :].clone().reshape(1, 1, 1)
            for i in range(1, len(q)):
                h_t_ = (h_t[i - 1, 0, 0] * self.w_r_1 + q[i, 0, 0] * self.w_r_2).reshape(1, 1, 1)
                h_t = torch.cat((h_t, h_t_), 0)
                q2_ = (h_t_ * self.w_z_1 + q[i, 0, 0] * self.w_z_2).reshape(1, 1, 1)
                q2 = torch.cat((q2, q2_), 0)

            #     q = self.gru_(q)
            #     q, q_max, q_min = self.BatchNorm1d(q)
            #     q, h = self.rnn(q)
            #     q = q_min + (q * (q_max - q_min))
            return q2
        return q


# torch - pesticide 读取农药训练数据
class PesticideTrainDataSet(nn.Module):
    def __init__(self, area_path: str, pesticide_usage_path: str, pesticide_dir: str, runflow_dir: str, runflow_observe_dir: str,
                 pesticide_name: str):
        super(PesticideTrainDataSet, self).__init__()
        self.area_data = pd.read_csv(area_path, delimiter=",", encoding="utf8", index_col="OBJECTID")  # 区域数据
        self.usage_data = pd.read_csv(pesticide_usage_path, delimiter=",", encoding="utf8",  # 使用量数据
                                      index_col=r"pesticide(ug/m2/d)")
        self.pesticide_dir = pesticide_dir  # 灭蚊剂检测数据 - [THA...]
        self.runflow_dir = runflow_dir  # 模拟出的流量数据 - 1-3722
        self.runflow_observe_dir = runflow_observe_dir  # 观测的流量数据
        self.pesticide_name = pesticide_name  # 要模拟的灭蚊剂
        # print("usage_data:", self.usage_data)
        # print("area_data:", self.area_data)

    def __len__(self):
        return 3

    def __getitem__(self, areas_id):
        pesticide_output_data = pd.DataFrame([])
        runflow_output_observe_data = pd.DataFrame([])
        runflow_output_data = pd.DataFrame([])
        for area_id in areas_id:
            #
            pesticide_path = os.path.join(self.pesticide_dir, "{}_modify.csv".format(str(area_id)))
            pesticide_data = pd.read_csv(pesticide_path, delimiter=",", encoding="utf8", index_col="time")
            pesticide_data = pesticide_data.loc[:, ["times", self.pesticide_name]]
            pesticide_data.insert(loc=2, column="area_id", value=area_id)
            pesticide_output_data = pd.concat((pesticide_output_data, pesticide_data), axis=0)
            # print("pesticide_data:", pesticide_data)
            runflow_observe_path = os.path.join(self.runflow_observe_dir, "modify_time_{}.csv".format(area_id))
            runflow_observe_data = pd.read_csv(runflow_observe_path, delimiter=",", encoding="utf8", index_col="time")
            runflow_observe_data.insert(loc=1, column="area_id", value=area_id)
            runflow_observe_data.index = [
                datetime.datetime.strftime(datetime.datetime.fromisoformat(i), "%Y-%m-%dT%H:%M")
                for i in runflow_observe_data.index]
            runflow_output_observe_data = pd.concat((runflow_output_observe_data, runflow_observe_data), axis=0)
            # print("runflow_observe_data:", runflow_observe_data)
            runflow_path = os.path.join(self.runflow_dir, "runflow_{}.csv".format(str(area_id)))
            runflow_data = pd.read_csv(runflow_path, delimiter=",", encoding="utf8")
            runflow_data.insert(loc=3, column="greenland", value=self.area_data.loc[area_id, "greenland"])
            runflow_data.insert(loc=4, column="greenland_5m", value=self.area_data.loc[area_id, "greenland_5m"])
            runflow_data.insert(loc=5, column="water", value=self.area_data.loc[area_id, "water"])
            runflow_data.insert(loc=6, column="impervious", value=self.area_data.loc[area_id, "impervious"])
            runflow_data.insert(loc=7, column="area", value=self.area_data.loc[area_id, "area"])
            runflow_data.insert(loc=8, column="slope", value=self.area_data.loc[area_id, "slope"])
            runflow_data.insert(loc=9, column="name", value=self.area_data.loc[area_id, "name"])
            runflow_data.insert(loc=10, column="area_id", value=self.area_data.loc[area_id, "blockId"])
            runflow_data.insert(loc=11, column="class", value=self.area_data.loc[area_id, "class"])
            runflow_output_data = pd.concat((runflow_output_data, runflow_data), axis=0)
            # print("runflow_data:", runflow_data)
        runflow_output_data.index = [i for i in range(len(runflow_output_data))]
        return runflow_output_data, pesticide_output_data, runflow_output_observe_data, self.usage_data


# torch - pesticide 读取农药预测数据
class PesticideTotalDataSet(nn.Module):
    def __init__(self, area_path: str, pesticide_usage_path: str, runflow_dir: str):
        super(PesticideTotalDataSet, self).__init__()
        self.area_data = pd.read_csv(area_path, delimiter=",", encoding="utf8", index_col="OBJECTID")  # 区域数据
        self.usage_data = pd.read_csv(pesticide_usage_path, delimiter=",", encoding="utf8",
                                      index_col=r"pesticide(ug/m2/d)")  # 使用量数据
        self.runflow_dir = runflow_dir  # 模拟出的流量数据 - 1-3722  runflow_{}.csv
        # self.pesticide_names = pesticide_names  # 要模拟的灭蚊剂名称

    def __len__(self):
        return len(self.pesticide_names)

    def __getitem__(self, area_id):  # 获得对应区域的用于预测的数据
        # 读取模拟出来的流量数据
        runflow_path = os.path.join(self.runflow_dir, "runflow_{}.csv".format(str(area_id)))
        runflow_data = pd.read_csv(runflow_path, delimiter=",", encoding="utf8")
        runflow_data.insert(loc=3, column="greenland", value=self.area_data.loc[area_id, "greenland"])
        runflow_data.insert(loc=4, column="greenland_5m", value=self.area_data.loc[area_id, "greenland_5m"])
        runflow_data.insert(loc=5, column="water", value=self.area_data.loc[area_id, "water"])
        runflow_data.insert(loc=6, column="impervious", value=self.area_data.loc[area_id, "impervious"])
        runflow_data.insert(loc=7, column="area", value=self.area_data.loc[area_id, "area"])
        runflow_data.insert(loc=8, column="slope", value=self.area_data.loc[area_id, "slope"])
        runflow_data.insert(loc=9, column="name", value=self.area_data.loc[area_id, "name"])
        runflow_data.insert(loc=10, column="area_id", value=self.area_data.loc[area_id, "blockId"])
        runflow_data.insert(loc=11, column="class", value=self.area_data.loc[area_id, "class"])
        # print(self.usage_data)
        # print(runflow_data)
        return runflow_data, self.usage_data


# torch - pesticide 灭蚊剂训练模拟
class PesticideTrainPredict(nn.Module):
    def __init__(self, params: list):
        super(PesticideTrainPredict, self).__init__()
        # 污染物累积
        # self.accumulate_max = torch.nn.Parameter(torch.tensor([40.440791974599927], requires_grad=True))  # mg/m3
        # self.register_parameter("accumulate_max", self.accumulate_max)
        # self.accumulate_n = torch.nn.Parameter(torch.tensor([4.672340705329424], requires_grad=True))  # day
        # self.register_parameter("accumulate_n", self.accumulate_n)
        #
        self.accumulate_m0_day = torch.nn.Parameter(torch.tensor([3.04219055811343], requires_grad=True))  # mg/m3
        self.register_parameter("accumulate_m0_day", self.accumulate_m0_day)
        self.accumulate_retain = torch.nn.Parameter(torch.tensor([0.907313539587267], requires_grad=True))  # %/d
        self.register_parameter("accumulate_retain", self.accumulate_retain)
        self.accumulate_degrade = torch.nn.Parameter(torch.tensor([0.895373541685726], requires_grad=True))  # %/d
        self.register_parameter("accumulate_degrade", self.accumulate_degrade)
        # 喷洒修正系数
        self.accumulate_usage_plus_870 = torch.nn.Parameter(torch.tensor([1.33118995142345], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_870", self.accumulate_usage_plus_870)
        self.accumulate_usage_plus_3718 = torch.nn.Parameter(torch.tensor([0.901858392588013], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_3718", self.accumulate_usage_plus_3718)
        self.accumulate_usage_plus_3716 = torch.nn.Parameter(torch.tensor([1.19306950845391], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_3716", self.accumulate_usage_plus_3716)
        # 污染物冲刷
        self.scour_C1 = torch.nn.Parameter(torch.tensor([0.227983188296579], requires_grad=True))
        self.register_parameter("scour_C1", self.scour_C1)
        self.scour_C2 = torch.nn.Parameter(torch.tensor([0.262647971835829], requires_grad=True))
        self.register_parameter("scour_C2", self.scour_C2)
        self.scour_B1 = torch.nn.Parameter(torch.tensor([0.40876438184221], requires_grad=True))
        self.register_parameter("scour_B1", self.scour_B1)
        self.scour_B2 = torch.nn.Parameter(torch.tensor([0.630350367766635], requires_grad=True))
        self.register_parameter("scour_B2", self.scour_B2)
        if len(params) != 0:
            self.accumulate_m0_day.data = torch.tensor([params[0]])
            self.accumulate_retain.data = torch.tensor([params[1]])
            self.accumulate_degrade.data = torch.tensor([params[2]])
            self.accumulate_usage_plus_870.data = torch.tensor([params[3]])
            self.accumulate_usage_plus_3718.data = torch.tensor([params[4]])
            self.accumulate_usage_plus_3716.data = torch.tensor([params[5]])
            self.scour_C1.data = torch.tensor([params[6]])
            self.scour_C2.data = torch.tensor([params[7]])
            self.scour_B1.data = torch.tensor([params[8]])
            self.scour_B2.data = torch.tensor([params[9]])

    def forward(self, runflow_data, pesticide_data, runflow_observe_data, usage_data, pesticide_name, areas_id):
        #
        # self.accumulate_usage_plus_870.requires_grad = False
        # self.accumulate_usage_plus_3716.requires_grad = False
        # self.accumulate_usage_plus_3718.requires_grad = False
        # self.accumulate_retain.requires_grad = False
        # self.accumulate_degrade.requires_grad = False
        # self.accumulate_m0_day.requires_grad = False
        # self.scour_C1.requires_grad = False
        # self.scour_C2.requires_grad = False
        # self.scour_B1.requires_grad = False
        # self.scour_B2.requires_grad = False
        # 污染物累积
        # self.accumulate_max.data = torch.clamp(self.accumulate_max.data, 0.01, 500)
        # self.accumulate_n.data = torch.clamp(self.accumulate_n.data, 2, 15)
        #
        self.accumulate_m0_day.data = torch.clamp(self.accumulate_m0_day.data, 0.01, 10)
        self.accumulate_retain.data = torch.clamp(self.accumulate_retain, 0.5, 0.99)
        self.accumulate_degrade.data = torch.clamp(self.accumulate_degrade, 0.5, 0.99)
        # 喷洒修正系数
        self.accumulate_usage_plus_870.data = torch.clamp(self.accumulate_usage_plus_870.data, 0.001, 1000)
        self.accumulate_usage_plus_3716.data = torch.clamp(self.accumulate_usage_plus_3716.data, 0.001, 1000)
        self.accumulate_usage_plus_3718.data = torch.clamp(self.accumulate_usage_plus_3718.data, 0.001, 1000)
        # 污染物冲刷
        self.scour_C1.data = torch.clamp(self.scour_C1.data, 0.001, 100)
        self.scour_C2.data = torch.clamp(self.scour_C2.data, 0.2, 20)
        self.scour_B1.data = torch.clamp(self.scour_B1.data, 0.1, 10)
        self.scour_B2.data = torch.clamp(self.scour_B2.data, 0.000001, 10)

        # 记录数据
        total_pesticide_output_tensor = torch.tensor([])  # 输出污染物浓度
        total_pesticide_observe_tensor = torch.tensor([])  # 输出原始污染物浓度
        total_runflow_tensor = torch.tensor([])  # 输出流量
        total_prec_tensor = torch.tensor([])  # 输出雨量
        total_accumulation_tensor = torch.tensor([])  # 输出区域污染物累积
        total_accumulation_sub_tensor = torch.tensor([])  # 输出区域污染物冲刷
        total_time_str_list = []  # 输出时间
        total_area_id_list = []  # 输出区域位置
        training_pesticide_predict_tensor = torch.tensor([])  # 输出训练的污染物浓度
        training_pesticide_observe_tensor = torch.tensor([])  # 输出训练的原始污染物浓度
        training_runflow_predict_tensor = torch.tensor([])  # 输出训练的流量
        training_runflow_observe_tensor = torch.tensor([])  # 输出训练的原始流量
        training_prec_tensor = torch.tensor([])  # 输出训练的雨量
        training_accumulation_tensor = torch.tensor([])  # 输出训练的污染物累积
        training_accumulation_sub_tensor = torch.tensor([])  # 输出训练的污染物冲刷
        training_time_str_list = []  # 输出训练的时间
        training_area_id_list = []  # 输出训练的区域位置
        validation_pesticide_predict_tensor = torch.tensor([])  # 输出验证的污染物浓度
        validation_pesticide_observe_tensor = torch.tensor([])  # 输出验证的原始污染物浓度
        validation_runflow_predict_tensor = torch.tensor([])  # 输出验证的流量
        validation_runflow_observe_tensor = torch.tensor([])  # 输出验证的原始流量
        validation_prec_tensor = torch.tensor([])  # 输出验证的雨量
        validation_accumulation_tensor = torch.tensor([])  # 输出验证的污染物累积
        validation_accumulation_sub_tensor = torch.tensor([])  # 输出验证的污染物冲刷
        validation_time_str_list = []  # 输出验证的时间
        validation_area_id_list = []  # 输出验证的区域位置
        compare_pesticide_predict_tensor = torch.tensor([])  # 输出总对应的污染物浓度
        compare_pesticide_observe_tensor = torch.tensor([])  # 输出总对应的原始污染物浓度
        compare_runflow_predict_tensor = torch.tensor([])  # 输出总对应的流量
        compare_runflow_observe_tensor = torch.tensor([])  # 输出总对应的原始流量
        compare_prec_tensor = torch.tensor([])  # 输出总对应的雨量
        compare_accumulation_tensor = torch.tensor([])  # 输出总对应的污染物累积
        compare_accumulation_sub_tensor = torch.tensor([])  # 输出总对应的污染物冲刷
        compare_time_str_list = []  # 输出总对应的时间
        compare_area_id_list = []  # 输出总对应的区域位置
        for area_id in areas_id:
            # 过滤特定的区域
            runflow_data_filter = runflow_data[runflow_data["area_id"] == area_id]
            # print(runflow_data_filter)
            pesticide_data_filter = pesticide_data[pesticide_data["area_id"] == area_id]
            runflow_observe_data_filter = runflow_observe_data[runflow_observe_data["area_id"] == area_id]
            # 提取固定的区域参数
            area_name = runflow_data_filter.loc[runflow_data_filter.index[0], "name"]
            greenland_5m = torch.tensor([runflow_data_filter.loc[runflow_data_filter.index[0], "greenland_5m"]])
            water = torch.tensor([runflow_data_filter.loc[runflow_data_filter.index[0], "water"]])
            impervious = torch.tensor([runflow_data_filter.loc[runflow_data_filter.index[0], "impervious"]])
            area = torch.tensor([runflow_data_filter.loc[runflow_data_filter.index[0], "area"]])
            slope = torch.tensor([runflow_data_filter.loc[runflow_data_filter.index[0], "slope"]])
            classification = str(runflow_data_filter.loc[runflow_data_filter.index[0], "class"])
            pesticide_usage = torch.tensor([usage_data.loc[pesticide_name, area_name]])
            greenland = greenland_5m * area
            # 初始时间
            time = datetime.datetime.strptime(runflow_data_filter.loc[runflow_data_filter.index[0], "time"],
                                              "%Y-%m-%dT%H:%M")
            last_time_begin = datetime.datetime.strptime(runflow_data_filter.loc[runflow_data_filter.index[0], "time"],
                                                         "%Y-%m-%dT%H:%M")
            # 修改喷洒量
            if area_id == 870:
                pesticide_usage = pesticide_usage * self.accumulate_usage_plus_870
            elif area_id == 3718:
                pesticide_usage = pesticide_usage * self.accumulate_usage_plus_3718
            elif area_id == 3716:
                pesticide_usage = pesticide_usage * self.accumulate_usage_plus_3716
            # 污染物初始累积 (ug/m2)
            accumulation = self.accumulate_m0_day * pesticide_usage * \
                           torch.pow(self.accumulate_degrade, self.accumulate_m0_day) * \
                           torch.pow(self.accumulate_retain, self.accumulate_m0_day)
            # 筛选雨次 0 - ?
            rain_num = runflow_data_filter.loc[runflow_data_filter.index[-1], "id"]
            for rain_th in range(rain_num + 1):
                runflow_data_filter_th = runflow_data_filter[runflow_data_filter["id"] == rain_th]
                v_t = 0  # 流量累积
                # 计算与上次降雨间隔的时间
                time_begin_str = runflow_data_filter_th.loc[runflow_data_filter_th.index[0], "time"]
                time_begin = datetime.datetime.strptime(time_begin_str, "%Y-%m-%dT%H:%M")
                time_interval = torch.tensor(
                    (time_begin.timestamp() - last_time_begin.timestamp()) / 60 / 60 / 24)  # day
                # 污染物累积 (ug/m2)  B = C1 * (1 - e^(-C2 * t))
                # accumulation_ug = accumulation_ug + ((self.accumulate_max * 10 * pesticide_usage - accumulation) * greenland * time_interval) / (
                #         self.accumulate_n + time_interval)
                # 污染物累积 (ug/m2)  B = C1 * (1 - e^(-C2 * t))
                accumulation = accumulation + pesticide_usage * time_interval
                accumulation = accumulation * torch.pow(self.accumulate_degrade, time_interval)  # 降解损耗
                accumulation = accumulation * torch.pow(self.accumulate_retain, time_interval)  # 喷洒损耗
                accumulation_ug = accumulation * greenland
                accumulation_sub = 0  # 在该场雨中失去的污染物
                # 记录该次降雨开始的时间, 用作下一场雨的时间计算
                last_time_begin_str = runflow_data_filter_th.loc[runflow_data_filter_th.index[0], "time"]
                last_time_begin = datetime.datetime.strptime(last_time_begin_str, "%Y-%m-%dT%H:%M")
                # step in
                for i in runflow_data_filter_th.index:
                    # 当前时间
                    time_str = runflow_data_filter_th.loc[i, "time"]
                    time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
                    # time_interval = torch.tensor((time.timestamp() - last_time.timestamp()) / 60 / 60 / 24)  # day
                    runflow = torch.tensor([runflow_data_filter_th.loc[i, "runflow"]])  # m3 10min内
                    prec = torch.tensor([runflow_data_filter_th.loc[i, "prec"]])
                    # 污染物冲刷 (ug/L)  W = C1 * Qt^C2 * (P0 * delta_t  - Pwt)  ***  delta_t = 1 / (1 + B1 * e^ (-B2 * Vt))
                    v_t = v_t + runflow  # m3
                    delta_t = 1 / (1 + self.scour_B1 * torch.pow(np.e, -self.scour_B2 * v_t))
                    scouring_ug = self.scour_C1 * torch.pow(runflow / 10 / 60, self.scour_C2) * (
                            (accumulation_ug * delta_t - accumulation_sub) / greenland)
                    scouring_ug = clamp_value(scouring_ug, torch.tensor([0.0]), np.inf)
                    scouring = scouring_ug  # ug/L
                    # print("scouring:", scouring)
                    # print("scouring_ug:", scouring_ug)
                    # 记录失去的污染物
                    accumulation_sub = accumulation_sub + scouring_ug * runflow
                    accumulation_sub = clamp_value(accumulation_sub, torch.tensor([0.0]), np.inf)
                    # print("accumulation_sub,", str(accumulation_sub.item()))
                    # print("accumulation,", str(accumulation.item()))
                    # print("scouring_ug,", str(scouring_ug.item()))

                    # scouring_ug *= wetting
                    # print("scouring_ug: ", scouring_ug)
                    # 记录数据
                    total_pesticide_output_tensor = torch.cat((total_pesticide_output_tensor, scouring), 0)
                    total_runflow_tensor = torch.cat((total_runflow_tensor, runflow), 0)
                    total_prec_tensor = torch.cat((total_prec_tensor, prec), 0)
                    total_time_str_list.append(time_str)
                    total_area_id_list.append(str(area_id))
                    total_accumulation_tensor = torch.cat((total_accumulation_tensor, accumulation_ug), 0)
                    total_accumulation_sub_tensor = torch.cat((total_accumulation_sub_tensor, accumulation_sub), 0)
                    # 是否为已有灭蚊剂浓度的时间点
                    if time_str in pesticide_data_filter.index:
                        total_pesticide_observe_tensor = torch.cat(
                            (total_pesticide_observe_tensor,
                             torch.tensor([pesticide_data_filter.loc[time_str, pesticide_name]])), 0)
                        # 总数据
                        compare_time_str_list.append(time_str)
                        compare_area_id_list.append(str(area_id))
                        compare_runflow_predict_tensor = torch.cat((compare_runflow_predict_tensor, runflow), 0)
                        compare_pesticide_predict_tensor = torch.cat((compare_pesticide_predict_tensor, scouring), 0)
                        compare_prec_tensor = torch.cat((compare_prec_tensor, prec), 0)
                        compare_accumulation_tensor = torch.cat((compare_accumulation_tensor, accumulation_ug), 0)
                        compare_accumulation_sub_tensor = torch.cat((compare_accumulation_sub_tensor, accumulation_sub),
                                                                    0)
                        compare_pesticide_observe_tensor = torch.cat(
                            (compare_pesticide_observe_tensor,
                             torch.tensor([pesticide_data_filter.loc[time_str, pesticide_name]])), 0)
                        if datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M") in runflow_observe_data_filter.index:
                            compare_runflow_observe_tensor = torch.cat((compare_runflow_observe_tensor, torch.tensor([
                                runflow_observe_data_filter.loc[
                                    datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M"), "flow"]])), 0)
                        else:
                            compare_runflow_observe_tensor = torch.cat((compare_runflow_observe_tensor, runflow), 0)
                        # 率定数据
                        if time.timestamp() - datetime.datetime(2022, 8, 16).timestamp() < 0:  # 8.16 之前
                            training_time_str_list.append(time_str)
                            training_area_id_list.append(str(area_id))
                            training_runflow_predict_tensor = torch.cat((training_runflow_predict_tensor, runflow), 0)
                            training_pesticide_predict_tensor = torch.cat((training_pesticide_predict_tensor, scouring), 0)
                            training_prec_tensor = torch.cat((training_prec_tensor, prec), 0)
                            training_accumulation_tensor = torch.cat((training_accumulation_tensor, accumulation_ug), 0)
                            training_accumulation_sub_tensor = torch.cat((training_accumulation_sub_tensor, accumulation_sub),
                                                                         0)
                            training_pesticide_observe_tensor = torch.cat(
                                (training_pesticide_observe_tensor,
                                 torch.tensor([pesticide_data_filter.loc[time_str, pesticide_name]])), 0)
                            if datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M") in runflow_observe_data_filter.index:
                                training_runflow_observe_tensor = torch.cat((training_runflow_observe_tensor, torch.tensor([
                                    runflow_observe_data_filter.loc[
                                        datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M"), "flow"]])), 0)
                            else:
                                training_runflow_observe_tensor = torch.cat((training_runflow_observe_tensor, runflow), 0)
                        else:
                            validation_time_str_list.append(time_str)
                            validation_area_id_list.append(str(area_id))
                            validation_runflow_predict_tensor = torch.cat((validation_runflow_predict_tensor, runflow), 0)
                            validation_pesticide_predict_tensor = torch.cat((validation_pesticide_predict_tensor, scouring), 0)
                            validation_prec_tensor = torch.cat((validation_prec_tensor, prec), 0)
                            validation_accumulation_tensor = torch.cat((validation_accumulation_tensor, accumulation_ug), 0)
                            validation_accumulation_sub_tensor = torch.cat((validation_accumulation_sub_tensor, accumulation_sub),
                                                                           0)
                            validation_pesticide_observe_tensor = torch.cat(
                                (validation_pesticide_observe_tensor,
                                 torch.tensor([pesticide_data_filter.loc[time_str, pesticide_name]])), 0)
                            if datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M") in runflow_observe_data_filter.index:
                                validation_runflow_observe_tensor = torch.cat((validation_runflow_observe_tensor, torch.tensor([
                                    runflow_observe_data_filter.loc[
                                        datetime.datetime.strftime(time, "%Y-%m-%dT%H:%M"), "flow"]])), 0)
                            else:
                                validation_runflow_observe_tensor = torch.cat((validation_runflow_observe_tensor, runflow), 0)
                    else:
                        total_pesticide_observe_tensor = torch.cat(
                            (total_pesticide_observe_tensor, torch.tensor([0.0])), 0)
                # print("accumulation_ug:", str(accumulation_ug.item()))
                # print("accumulation_sub:", str(accumulation_sub.item()))
                time_rainfall_length = (time.timestamp() - time_begin.timestamp()) / 60 / 60 / 24
                # 去除降雨期间降解的污染物
                accumulation_ug = (accumulation_ug - accumulation_sub) * torch.pow(self.accumulate_degrade,
                                                                                   time_rainfall_length)
                accumulation = accumulation_ug / greenland
        return [total_time_str_list,
                total_pesticide_output_tensor,
                total_pesticide_observe_tensor,
                total_runflow_tensor,
                total_prec_tensor,
                total_accumulation_tensor,
                total_accumulation_sub_tensor,
                total_area_id_list], \
               [compare_pesticide_predict_tensor,
                compare_pesticide_observe_tensor,
                compare_runflow_predict_tensor,
                compare_runflow_observe_tensor,
                compare_prec_tensor,
                compare_accumulation_tensor,
                compare_accumulation_sub_tensor,
                compare_time_str_list,
                compare_area_id_list], \
               [training_pesticide_predict_tensor,
                training_pesticide_observe_tensor,
                training_runflow_predict_tensor,
                training_runflow_observe_tensor,
                training_prec_tensor,
                training_accumulation_tensor,
                training_accumulation_sub_tensor,
                training_time_str_list,
                training_area_id_list], \
               [validation_pesticide_predict_tensor,
                validation_pesticide_observe_tensor,
                validation_runflow_predict_tensor,
                validation_runflow_observe_tensor,
                validation_prec_tensor,
                validation_accumulation_tensor,
                validation_accumulation_sub_tensor,
                validation_time_str_list,
                validation_area_id_list]


# torch - pesticide 灭蚊剂预测
class PesticideTotalPredict(nn.Module):
    def __init__(self):
        super(PesticideTotalPredict, self).__init__()
        # 污染物累积
        # self.accumulate_max = torch.nn.Parameter(torch.tensor([40.440791974599927], requires_grad=True))  # mg/m3
        # self.register_parameter("accumulate_max", self.accumulate_max)
        # self.accumulate_n = torch.nn.Parameter(torch.tensor([4.672340705329424], requires_grad=True))  # day
        # self.register_parameter("accumulate_n", self.accumulate_n)
        #
        self.accumulate_m0_day = torch.nn.Parameter(torch.tensor([3.05018056500963], requires_grad=True))  # mg/m3
        self.register_parameter("accumulate_m0_day", self.accumulate_m0_day)
        self.accumulate_retain = torch.nn.Parameter(torch.tensor([0.94184679376275], requires_grad=True))  # %/d
        self.register_parameter("accumulate_retain", self.accumulate_retain)
        self.accumulate_degrade = torch.nn.Parameter(torch.tensor([0.904118422535808], requires_grad=True))  # %/d
        self.register_parameter("accumulate_degrade", self.accumulate_degrade)
        # 喷洒修正系数
        self.accumulate_usage_plus_870 = torch.nn.Parameter(torch.tensor([1.08440819110883], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_870", self.accumulate_usage_plus_870)
        self.accumulate_usage_plus_3718 = torch.nn.Parameter(torch.tensor([1.58718310554308], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_3718", self.accumulate_usage_plus_3718)
        self.accumulate_usage_plus_3716 = torch.nn.Parameter(torch.tensor([2.1140102328541], requires_grad=True))
        self.register_parameter("accumulate_usage_plus_3716", self.accumulate_usage_plus_3716)
        # 污染物冲刷
        self.scour_C1 = torch.nn.Parameter(torch.tensor([0.472591131935476], requires_grad=True))
        self.register_parameter("scour_C1", self.scour_C1)
        self.scour_C2 = torch.nn.Parameter(torch.tensor([0.407377747374919], requires_grad=True))
        self.register_parameter("scour_C2", self.scour_C2)
        self.scour_B1 = torch.nn.Parameter(torch.tensor([0.445718540597239], requires_grad=True))
        self.register_parameter("scour_B1", self.scour_B1)
        self.scour_B2 = torch.nn.Parameter(torch.tensor([0.00667679290725365], requires_grad=True))
        self.register_parameter("scour_B2", self.scour_B2)

    def forward(self, runflow_data, usage_data, pesticide_name, params):
        # 读取传入的参数
        if len(params) != 0:
            self.accumulate_m0_day.data = torch.tensor(params[0])
            self.accumulate_retain.data = torch.tensor(params[1])
            self.accumulate_degrade.data = torch.tensor(params[2])
            self.accumulate_usage_plus_870.data = torch.tensor(params[3])
            self.accumulate_usage_plus_3716.data = torch.tensor(params[4])
            self.accumulate_usage_plus_3718.data = torch.tensor(params[5])
            self.scour_C1.data = torch.tensor(params[6])
            self.scour_C2.data = torch.tensor(params[7])
            self.scour_B1.data = torch.tensor(params[8])
            self.scour_B2.data = torch.tensor(params[9])
        # 记录数据
        total_pesticide_output_tensor = torch.tensor([])  # 输出污染物浓度
        # 提取固定的区域参数
        area_name = runflow_data.loc[runflow_data.index[0], "name"]
        area_id = runflow_data.loc[runflow_data.index[0], "area_id"]
        greenland_5m = torch.tensor([runflow_data.loc[runflow_data.index[0], "greenland_5m"]])
        water = torch.tensor([runflow_data.loc[runflow_data.index[0], "water"]])
        impervious = torch.tensor([runflow_data.loc[runflow_data.index[0], "impervious"]])
        area = torch.tensor([runflow_data.loc[runflow_data.index[0], "area"]])
        slope = torch.tensor([runflow_data.loc[runflow_data.index[0], "slope"]])
        classification = str(runflow_data.loc[runflow_data.index[0], "class"])
        pesticide_usage = torch.tensor([usage_data.loc[pesticide_name, area_name]])
        greenland = greenland_5m * area
        # 初始时间
        time = datetime.datetime.strptime(runflow_data.loc[runflow_data.index[0], "time"],
                                          "%Y-%m-%dT%H:%M")
        last_time_begin = datetime.datetime.strptime(runflow_data.loc[runflow_data.index[0], "time"],
                                                     "%Y-%m-%dT%H:%M")
        # 修改喷洒量
        # print(time)
        datetime.timedelta()
        # 修改喷洒量
        if area_id == 870:
            pesticide_usage = pesticide_usage * self.accumulate_usage_plus_870
        elif area_id == 3718:
            pesticide_usage = pesticide_usage * self.accumulate_usage_plus_3718
        elif area_id == 3716:
            pesticide_usage = pesticide_usage * self.accumulate_usage_plus_3716
        # 污染物初始累积 ug/m3
        accumulation = self.accumulate_m0_day * pesticide_usage * \
                       torch.pow(self.accumulate_degrade, self.accumulate_m0_day) * \
                       torch.pow(self.accumulate_retain, self.accumulate_m0_day)
        # 筛选雨次 0 - ?
        rain_num = runflow_data.loc[runflow_data.index[-1], "id"]
        for rain_th in range(rain_num + 1):
            runflow_data_th = runflow_data[runflow_data["id"] == rain_th]
            v_t = 0  # 流量累积
            # 计算与上次降雨间隔的时间
            time_begin_str = runflow_data_th.loc[runflow_data_th.index[0], "time"]
            time_begin = datetime.datetime.strptime(time_begin_str, "%Y-%m-%dT%H:%M")
            time_interval = torch.tensor(
                (time_begin.timestamp() - last_time_begin.timestamp()) / 60 / 60 / 24)  # day
            # 污染物累积 (ug/m3)  B = C1 * (1 - e^(-C2 * t))
            # accumulation_ug = accumulation_ug + ((self.accumulate_max * 10 * pesticide_usage - accumulation) * greenland * time_interval) / (
            #         self.accumulate_n + time_interval)
            # 污染物累积 (ug/m3)  B = C1 * (1 - e^(-C2 * t))
            accumulation = accumulation + pesticide_usage * time_interval
            accumulation = accumulation * torch.pow(self.accumulate_degrade, time_interval)  # 降解损耗
            accumulation = accumulation * torch.pow(self.accumulate_retain, time_interval)  # 喷洒损耗
            accumulation_ug = accumulation * greenland
            accumulation_sub = 0  # 在该场雨中失去的污染物
            # 记录该次降雨开始的时间, 用作下一场雨的时间计算
            last_time_begin_str = runflow_data_th.loc[runflow_data_th.index[0], "time"]
            last_time_begin = datetime.datetime.strptime(last_time_begin_str, "%Y-%m-%dT%H:%M")
            # step in
            for i in runflow_data_th.index:
                # 当前时间
                time_str = runflow_data_th.loc[i, "time"]
                time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
                # time_interval = torch.tensor((time.timestamp() - last_time.timestamp()) / 60 / 60 / 24)  # day
                runflow = torch.tensor([runflow_data_th.loc[i, "runflow"]])  # m3 10min内
                prec = torch.tensor([runflow_data_th.loc[i, "prec"]])
                # 污染物冲刷 (ug/L)  W = C1 * Qt^C2 * (P0 * delta_t  - Pwt)  ***  delta_t = 1 / (1 + B1 * e^ (-B2 * Vt))
                v_t = v_t + runflow  # m3
                delta_t = 1 / (1 + self.scour_B1 * torch.pow(np.e, -self.scour_B2 * v_t))
                if greenland != 0:
                    scouring_ug = self.scour_C1 * torch.pow(runflow / 10 / 60, self.scour_C2) * (
                            (accumulation_ug * delta_t - accumulation_sub) / greenland)
                else:
                    scouring_ug = torch.tensor([0.0])
                scouring_ug = clamp_value(scouring_ug, torch.tensor([0.0]), np.inf)
                scouring = scouring_ug  # ug/L
                # print("scouring:", scouring)
                # print("scouring_ug:", scouring_ug)
                # 记录失去的污染物
                accumulation_sub = accumulation_sub + scouring_ug * 1000
                accumulation_sub = clamp_value(accumulation_sub, torch.tensor([0.0]), np.inf)
                # print("accumulation_sub,", str(accumulation_sub.item()))
                # print("accumulation,", str(accumulation.item()))
                # print("scouring_ug,", str(scouring_ug.item()))

                # scouring_ug *= wetting
                # print("scouring_ug: ", scouring_ug)
                # 记录数据
                total_pesticide_output_tensor = torch.cat((total_pesticide_output_tensor, scouring), 0)
            # print("accumulation_ug:", str(accumulation_ug.item()))
            # print("accumulation_sub:", str(accumulation_sub.item()))
            time_rainfall_length = (time.timestamp() - time_begin.timestamp()) / 60 / 60 / 24
            # 去除该次降雨时间中降解的污染物
            accumulation_ug = (accumulation_ug - accumulation_sub) * torch.pow(self.accumulate_degrade,
                                                                               time_rainfall_length)
            accumulation = accumulation_ug / greenland
        total_pesticide_output_np = total_pesticide_output_tensor.detach().numpy()
        return total_pesticide_output_np


# PSO - pesticide 读取农药训练数据并输出相对应的污染物浓度时间序列
class LoadPesticideTrainDataSet:
    def __init__(self, area_path: str, pesticide_dir: str, pesticide_usage_path: str, runflow_dir: str,
                 runflow_observe_dir: str, areas_id: list):
        self.area_data = pd.read_csv(area_path, delimiter=",", encoding="utf8", index_col="OBJECTID")
        self.usage_data = pd.read_csv(pesticide_usage_path, delimiter=",", encoding="utf8",
                                      index_col=r"pesticide(ug/m2/d)")
        self.pesticide_dir = pesticide_dir
        self.runflow_dir = runflow_dir
        self.runflow_observe_dir = runflow_observe_dir
        self.areas_id = areas_id

    # 对所有训练区域进行单个污染物的遍历
    def __call__(self, params: list, pesticide_name: str):
        # 提取系数
        accumulate_C1, accumulate_C2, accumulation_init_coef, \
        scour_C1, scour_C2, \
            = params
        # 记录数据
        total_pesticide_output_list = []  # 输出的污染物浓度
        total_runflow_list = []  # 输出的流量
        total_prec_list = []  # 输出的雨量
        total_area_id_list = []  # 区域ID
        total_time_str_list = []  # 输出对应的时间
        compare_pesticide_predict_list = []  # 输出对应的污染物浓度
        compare_pesticide_observe_list = []  # 输出对应的原始污染物浓度
        compare_runflow_predict_list = []  # 输出对应的流量
        compare_runflow_observe_list = []  # 输出对应的原始流量
        compare_prec_list = []  # 输出对应的雨量
        compare_time_str_list = []  # 输出对应的时间
        compare_area_id_list = []  # 输出对应的区域ID
        for area_id in self.areas_id:
            pesticide_path = os.path.join(self.pesticide_dir, "{}_modify.csv".format(str(area_id)))
            pesticide_data = pd.read_csv(pesticide_path, delimiter=",", encoding="utf8", index_col="time")
            pesticide_data = pesticide_data.loc[:, ["times", pesticide_name]]
            # print(pesticide_data)
            runflow_path = os.path.join(self.runflow_dir, "runflow_{}.csv".format(str(area_id)))
            runflow_data = pd.read_csv(runflow_path, delimiter=",", encoding="utf8")
            # print(runflow_data)
            runflow_observe_path = os.path.join(self.runflow_observe_dir, "modify_time_{}.csv".format(area_id))
            runflow_observe_data = pd.read_csv(runflow_observe_path, delimiter=",", encoding="utf8", index_col="time")
            # print(self.area_data)
            # print(self.usage_data)
            # 提取固定的区域参数
            greenland_5m = self.area_data.loc[area_id, "greenland_5m"]
            water = self.area_data.loc[area_id, "water"]
            impervious = self.area_data.loc[area_id, "impervious"]
            area = self.area_data.loc[area_id, "area"]
            slope = self.area_data.loc[area_id, "slope"]
            area_name = self.area_data.loc[area_id, "name"]
            pesticide_usage = self.usage_data.loc[pesticide_name, area_name]

            # step
            accumulation = accumulate_C1 * accumulation_init_coef
            for i in runflow_data.index:
                # 获取参数
                if i != 0:
                    last_time_str = runflow_data.loc[i - 1, "time"]
                else:
                    last_time_str = runflow_data.loc[i, "time"]
                last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%dT%H:%M")
                time_str = runflow_data.loc[i, "time"]
                time = datetime.datetime.strptime(time_str, "%Y-%m-%dT%H:%M")
                time_interval = (time.timestamp() - last_time.timestamp()) / 60 / 60 / 24  # day
                runflow = runflow_data.loc[i, "runflow"]
                prec = runflow_data.loc[i, "prec"]
                # 污染物累积 (ug/m3)  B = C1 * (1 - e^(-C2 * t))
                accumulation += (accumulate_C1 - accumulation) * (1 - np.power(np.e, -accumulate_C2 * time_interval))
                accumulation_ug = accumulation * area
                # print("accumulation: ", accumulation)
                # print("accumulation_add: ", (accumulate_C1 - accumulation) * (1 - np.power(np.e, -accumulate_C2 * time_interval)))
                # 污染物冲刷 (ug/10min)  W = C1 * q^C2 * B
                scouring = scour_C1 * np.power(np.power(runflow, 1 / 3), scour_C2) * accumulation
                # print("scouring: ", scouring)
                # print("np.power(np.power(runflow, 1 / 2), scour_C2): ", np.power(np.power(runflow, 1 / 2), scour_C2))
                # 累积消耗
                accumulation_ug -= scouring * runflow * 10
                accumulation = accumulation_ug / area
                # 记录数据
                total_pesticide_output_list.append(scouring)
                total_runflow_list.append(runflow)
                total_prec_list.append(prec)
                total_area_id_list.append(area_id)
                total_time_str_list.append(time_str)
                if time_str in pesticide_data.index:
                    compare_time_str_list.append(time_str)
                    compare_runflow_predict_list.append(runflow)
                    compare_pesticide_predict_list.append(scouring)
                    compare_prec_list.append(prec)
                    compare_pesticide_observe_list.append(pesticide_data.loc[time_str, pesticide_name])
                    compare_area_id_list.append(area_id)
                    if datetime.datetime.strftime(time, "%Y-%m-%d %H:%M:%S") in runflow_observe_data.index:
                        compare_runflow_observe_list.append(
                            runflow_observe_data.loc[datetime.datetime.strftime(time, "%Y-%m-%d %H:%M:%S"), "flow"])
                    else:
                        compare_runflow_observe_list.append(runflow)
        return [np.array(total_time_str_list),
                np.array(total_pesticide_output_list),
                np.array(total_runflow_list),
                np.array(total_prec_list),
                np.array(total_area_id_list)], \
               [np.array(compare_pesticide_predict_list),
                np.array(compare_pesticide_observe_list),
                np.array(compare_runflow_predict_list),
                np.array(compare_runflow_observe_list),
                np.array(compare_prec_list),
                np.array(compare_time_str_list),
                np.array(compare_area_id_list)]


# 根据区域id的获得对应的区域数据
def get_area_properties(area_id, area_path=".\\data\\3_area\\area.csv"):
    file_data = pd.read_csv(area_path, index_col="OBJECTID")
    greenland = file_data.loc[int(area_id), "greenland"]
    water = file_data.loc[int(area_id), "water"]
    impervious = file_data.loc[int(area_id), "impervious"]
    area = file_data.loc[int(area_id), "Shape_Area"]
    slope = file_data.loc[int(area_id), "slope"]
    object_id = file_data.loc[int(area_id), "blockId"]
    return greenland, water, impervious, area, slope, object_id


# 对缺失降雨数据通过前后数据进行计算
def get_missing_rainfall(time, file_data):
    interval = datetime.timedelta(hours=1)
    # 往前计算
    interval_pre_num = 0
    time_pre = time
    while True:
        interval_pre_num += 1
        time_pre = time_pre - interval
        if str(time_pre).replace(" ", "T") in file_data.index:
            rainfall_pre = file_data.loc[str(time_pre).replace(" ", "T"), "precip"]
            break
        elif interval_pre_num > 3:
            rainfall_pre = 0
            break
    # 往后计算
    interval_next_num = 0
    time_next = time
    while True:
        interval_next_num += 1
        time_next = time_next + interval
        if str(time_next).replace(" ", "T") in file_data.index:
            rainfall_next = file_data.loc[str(time_next).replace(" ", "T"), "precip"]
            break
        elif interval_next_num > 3:
            rainfall_next = 0
            break
    # 按比例输出结果
    return rainfall_pre * interval_next_num / (
            interval_next_num + interval_pre_num) + rainfall_next * interval_pre_num / (
                   interval_next_num + interval_pre_num)


# 异常值处理 - 1
def sigma3(flow, next_flow, last_flow, std, mean):  # 正态分布 3std
    flow_lower, flow_upper = mean - 3 * std, mean + 3 * std
    if flow > flow_upper or flow < flow_lower:
        flow = (next_flow + last_flow) / 2
    return flow


# 异常值处理 - 2
def boxplot(flow, next_flow, last_flow, q1, q3):  # 箱线四分位距
    iqr = q3 - q1
    flow_lower, flow_upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    if flow > flow_upper or flow < flow_lower:
        flow = (next_flow + last_flow) / 2
    return flow


# R2
def r2_score_simulate(observed, simulate):
    observed_mean = np.mean(observed)
    simulate_mean = np.mean(simulate)
    r2_down = np.power(np.sum(np.power(simulate - simulate_mean, 2)), 1 / 2) * np.power(
        np.sum(np.power(observed - observed_mean, 2)), 1 / 2)
    r2_up = np.sum((simulate - simulate_mean) * (observed - observed_mean))
    r2 = np.power(r2_up / r2_down, 2)
    return r2


# TIC
def tic_simulate(observed, simulate):
    n = len(observed)
    tic_up = np.power(np.sum(np.power(observed - simulate, 2)) / n, 1 / 2)
    tic_down = np.power(np.sum(np.power(observed, 2)) / n, 1 / 2) + np.power(np.sum(np.power(simulate, 2)) / n, 1 / 2)
    tic = tic_up / tic_down
    return tic


# LOG
def log_dif(observed, simulate):
    dif_np = np.abs(np.log10(observed + 1e-5) - np.log10(simulate + 1e-5))
    total_num = len(dif_np)
    less_50_num = sum((dif_np < 0.5) == True)
    less_70_num = sum((dif_np < 0.7) == True)
    less_100_num = sum((dif_np < 1) == True)
    return less_50_num / total_num, less_70_num / total_num, less_100_num / total_num
