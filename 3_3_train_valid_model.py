import numpy as np
import torch.utils.data

from _parameters import parameters
from _method import write_statistic_data, write_log
from _algorithm import RunflowTrainDataset, Runflow, r2_score_simulate
import pandas as pd
import os
from sklearn.metrics import r2_score

if __name__ == '__main__':
    train_output_dir = parameters.train_output_dir
    train_output_2_path = os.path.join(train_output_dir, "2_organize_files", "organize_series.csv")
    train_output_3_dir = os.path.join(train_output_dir, "3_train_model_files")
    train_output_3_log_path = os.path.join(train_output_3_dir, "log.txt")
    train_output_3_train_path = os.path.join(train_output_3_dir, "train.csv")
    train_output_3_valid_path = os.path.join(train_output_3_dir, "valid.csv")
    if os.path.exists(train_output_3_train_path):
        os.remove(train_output_3_train_path)

    runflowDataset = RunflowTrainDataset(train_output_2_path)
    # 1- 随机读取
    train_dataset, validation_dataset = torch.utils.data.random_split(runflowDataset, (len(runflowDataset) - 8, 8))
    # 1 - 1
    random_train1 = np.array(
        [44, 39, 20, 22, 19, 15, 34, 9, 8, 5, 26, 37, 30, 42, 17, 16, 10, 18, 25, 33, 35, 27, 24, 4, 1, 29, 13, 43, 45, 12, 3, 11, 28, 38, 41, 21,
         14]) - 1
    random_validate1 = np.array([32, 6, 7, 36, 2, 31, 40, 23]) - 1
    # 2- 序列读取
    train_dataset_size = len(runflowDataset)
    valid_dataset_size = 0
    # 2- 时间读取
    train_range = np.array([i for i in range(len(runflowDataset))])
    valid_range = np.array([4, 10, 12, 41, 42, 43, 44, 45])  # id  每个站点后
    # valid_range = np.array([4, 8, 9, 10, 11, 12, 44, 45])  # id  时间序列后
    valid_range -= 1  # item
    for i in valid_range:  # 1 - 47 % valid
        train_range = np.delete(train_range, np.where(train_range == i))
    print("train_range", train_range)
    print("valid_range", valid_range)

    runflow = Runflow(1, 1, 20)

    # 判断是否开启GPU加速
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # a = runflowDataset[0]

    # 参数
    runflow_learning_rate = parameters.runflow_learning_rate
    runflow_epochs = parameters.runflow_epochs

    # 设定优化器
    optim = torch.optim.SGD(params=runflow.parameters(),
                            lr=runflow_learning_rate)
    # 设定损失函数
    mse_loss_fn = torch.nn.MSELoss()
    mse_loss_fn = mse_loss_fn.to(device=device)

    total_train_step = 0  # 记录训练的次数
    total_loss = 0  # 记录这次训练的总误差
    last_total_loss = 0  # 记录上次的总误差
    for epoch in range(1, runflow_epochs + 1):
        for i in random_train1:  # range(len(train_dataset)):  || range(train_dataset_size) || train_range
            target, label, data_id = runflowDataset[i]  # [n, 10]  [n, 1]   train_dataset[i] || runflowDataset[i] || runflowDataset[i]
            # target = target.to(device)
            # label = label.to(device)
            output = runflow(target, epoch)  # [n, 1, 1]
            label = label.reshape(label.shape[0], 1, 1)
            # print("output", output)
            # print("label", label)
            # 计算误差和训练次数
            loss = mse_loss_fn(label, output)
            total_loss += loss.item()
            total_train_step += 1
            # # 优化
            # for name, parameters in runflow.named_parameters():
            #     print(name, "grad:", parameters.grad, "value:", parameters.item())
            optim.zero_grad()
            loss.backward()
            # # 输出各个参数的值
            # for name, parameters in runflow.named_parameters():
            #     print(name, "grad:", parameters.grad, "value:", parameters.item())
            optim.step()
            # 记录最后一次训练的 train 结果
            if epoch == runflow_epochs:
                label_np = label[:, 0, 0].detach().numpy()
                output_np = output[:, 0, 0].detach().numpy()
                # write_statistic_data(train_output_3_train_path,
                #                      ["training_observe", "training_predict", "th", "prec"])
                for label_id in range(len(label_np)):
                    write_statistic_data(train_output_3_train_path,
                                         [str(label_np[label_id]), str(output_np[label_id]), str(data_id), str(target[label_id, 6].detach().numpy())])
        # write_log(path=train_output_3_log_path,
        #           string="执行第{0}次循环, 第{1}次计算, 现存loss总和为{2}".format(
        #               str(epoch),
        #               str(total_train_step),
        #               str(total_loss - last_total_loss)), )
        last_total_loss = total_loss
        # 保存训练的结果 (参数状态)
        if epoch % 20 == 0:
            torch.save(
                runflow.state_dict(),
                os.path.join(train_output_3_dir,
                             "epoch{0}.pth"
                             .format(str(epoch))
                             )
            )

    # 输出各个参数的值
    for name, parameters in runflow.named_parameters():
        print(name, "grad:", parameters.grad, "value:", parameters.item())

    # 计算 训练期 数据
    # 读取计算 NSE
    result = pd.read_csv(train_output_3_train_path)
    observed = result.iloc[:, 0]
    simulate = result.iloc[:, 1]
    nse_train = r2_score(observed, simulate)
    print("nse_train:", str(nse_train))
    observed_mean = np.mean(result.iloc[:, 0])
    simulate_mean = np.mean(result.iloc[:, 1])
    r2_train_down = np.power(np.sum(np.power(simulate - simulate_mean, 2)), 1 / 2) * np.power(np.sum(np.power(observed - observed_mean, 2)), 1 / 2)
    r2_train_up = np.sum((simulate - simulate_mean) * (observed - observed_mean))
    r2_train = np.power(r2_train_up / r2_train_down, 2)
    print("r2_train:", str(r2_train))

    # 计算 验证期 数据
    if os.path.exists(train_output_3_valid_path):
        os.remove(train_output_3_valid_path)
    for i in random_validate1:  # range(len(validation_dataset)) || valid_range
        target, label, data_id = runflowDataset[i]  # [n, 10]  [n, 1]    validation_dataset[i] || runflowDataset[i]
        output = runflow(target, runflow_epochs)  # [n, 1, 1]
        label = label.reshape(label.shape[0], 1, 1)
        label_np = label[:, 0, 0].detach().numpy()
        output_np = output[:, 0, 0].detach().numpy()
        # write_statistic_data(train_output_3_valid_path,
        #                      ["validation_observe", "validation_predict", "th", "prec"])
        for label_id in range(len(label_np)):
            write_statistic_data(train_output_3_valid_path,
                                 [str(label_np[label_id]), str(output_np[label_id]), str(data_id), str(target[label_id, 6].detach().numpy())])
    print("------------------------")
    print("r2_train:", str(r2_train))
    print("nse_train:", str(nse_train))
    # 读取计算 NSE
    result = pd.read_csv(train_output_3_valid_path)
    observed = result.iloc[:, 0]
    simulate = result.iloc[:, 1]
    r2_valid = r2_score_simulate(observed, simulate)
    nse_valid = r2_score(observed, simulate)
    print("r2_valid", str(r2_valid))
    print("nse_valid:", str(nse_valid))
    # if r2_valid < 0.36 or nse_valid < 0.36 or r2_train < 0.36 or nse_train < 0.36:
    #     os.system("python 3_3_train_valid_model.py")
