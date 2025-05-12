import datetime

import torch
import numpy as np
import os
import pandas as pd

from _parameters import parameters
from _method import write_statistic_data, write_log
from _algorithm import LoadPesticideTrainDataSet, PesticideTrainDataSet, PesticideTrainPredict, r2_score_simulate, \
    tic_simulate, log_dif
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

if __name__ == '__main__':
    # pesticide
    # pesticide_names = parameters.pesticide_names
    pesticide_dir = parameters.pesticide_dir
    pesticide_usage_path = os.path.join(pesticide_dir, "usage.csv")
    pesticide_output_dir = parameters.pesticide_output_dir
    pesticide_output_1_dir = os.path.join(pesticide_output_dir, "1_modify_pesticide")
    pesticide_output_2_dir = os.path.join(pesticide_output_dir, "2_train")
    # runflow - predict
    predicted_runflow_output_dir = parameters.predicted_runflow_output_dir
    predicted_runflow_output_2_dir = os.path.join(predicted_runflow_output_dir, "2_runflow")
    # runflow - observe
    runflow_output_dir = parameters.runflow_output_dir
    runflow_output_3_dir = os.path.join(runflow_output_dir, "2_modify_time_files")
    # area
    area_dir = parameters.area_dir
    area_path = os.path.join(area_dir, "area.csv")
    # area_id
    areas_id = parameters.area_id

    loadPesticideTrainDataSet = LoadPesticideTrainDataSet(area_path=area_path,
                                                          pesticide_dir=pesticide_output_1_dir,
                                                          pesticide_usage_path=pesticide_usage_path,
                                                          runflow_dir=predicted_runflow_output_2_dir,
                                                          runflow_observe_dir=runflow_output_3_dir,
                                                          areas_id=areas_id)

    # 针对不同灭蚊剂进行训练
    # pesticide_names = ["THM", "CLO", "IMI", "ACE", "THA",  # 噻虫嗪 噻虫胺 吡虫啉 啶虫脒 噻虫啉
    #                    "Propoxur", "Chlorpyrifos", "Temephos", "Fenthion", "Fipronil",  # 残杀威 毒死蜱* 双硫磷* 倍硫磷 氟虫腈*
    #                    "Tetramethrin", "Cyhalothrin_Lambda", "Cyfluthrin", "Permethrin", "Cypermethrin"]  # 胺菊酯 高效氯氟氰菊酯0 氟氯氰菊酯 氯菊酯 氯氰菊酯
    pesticide_names = ["THM"]
    for pesticide_name in pesticide_names:
        pesticide_output_2_params_path = os.path.join(pesticide_output_2_dir,
                                                      "{}_params.csv".format(pesticide_name))
        pesticideTrainDataSet = PesticideTrainDataSet(area_path=area_path,
                                                      pesticide_usage_path=pesticide_usage_path,
                                                      pesticide_dir=pesticide_output_1_dir,
                                                      runflow_dir=predicted_runflow_output_2_dir,
                                                      runflow_observe_dir=runflow_output_3_dir,
                                                      pesticide_name=pesticide_name)
        pesticidePredict = PesticideTrainPredict(
            params=[3.050180565, 0.941846794, 0.904118423, 1.084408191, 1.587183106, 2.114010233, 0.472591132,  # 是否自定义初始化参数
                    0.407377747, 0.445718541, 0.006676793])
        write_statistic_data(pesticide_output_2_params_path,
                             ["epoch"] + [name for name, parameters in pesticidePredict.named_parameters()] +
                             ["TIC", "MSLE", "log_50%", "log_70%", "log_100%", "R2", "NSE", "MAE", "MSE"])
        for name, parameters in pesticidePredict.named_parameters():
            print(name, "grad:", parameters.grad, "value:", parameters.item())
        # 判断是否开启GPU加速
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        # 参数
        pesticide_learning_rate = 1e-6
        pesticide_epochs = 1

        # 设定优化器
        optim = torch.optim.SGD(params=pesticidePredict.parameters(),
                                lr=pesticide_learning_rate)
        # 设定损失函数
        mse_loss_fn = torch.nn.L1Loss()
        mse_loss_fn = mse_loss_fn.to(device=device)

        total_train_step = 0  # 记录训练的次数
        total_loss = 0  # 记录这次训练的总误差
        last_total_loss = 0  # 记录上次的总误差
        for epoch in range(1, 1 + pesticide_epochs):
            print("epoch:", str(epoch))
            runflow_data, pesticide_data, runflow_observe_data, usage_data = pesticideTrainDataSet[areas_id]
            # print("runflow_data:", runflow_data)
            # print("pesticide_data:", pesticide_data)
            # print("runflow_observe_data:", runflow_observe_data)
            # print("usage_data:", usage_data)
            [total_time_str_list,
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
             validation_area_id_list] \
                = pesticidePredict(runflow_data, pesticide_data, runflow_observe_data, usage_data, pesticide_name,
                                   areas_id)
            loss = mse_loss_fn(training_pesticide_predict_tensor, training_pesticide_observe_tensor)
            total_loss += loss.item()
            total_train_step += 1

            # 输出各个参数的值
            for name, parameters in pesticidePredict.named_parameters():
                print(name, "grad:", parameters.grad, "value:", parameters.item())
            # 计算统计量 - train validate compare
            nse = r2_score(training_pesticide_observe_tensor.detach().numpy(),
                           training_pesticide_predict_tensor.detach().numpy())  # 评价指标 NSE
            r2 = r2_score_simulate(training_pesticide_observe_tensor.detach().numpy(),
                                   training_pesticide_predict_tensor.detach().numpy())  # 评价指标 R2
            tic = tic_simulate(training_pesticide_observe_tensor.detach().numpy(),
                               training_pesticide_predict_tensor.detach().numpy())  # 评价指标 TIC
            mae = mean_absolute_error(training_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MAE
                                      training_pesticide_predict_tensor.detach().numpy())
            msle = mean_squared_log_error(training_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSLE
                                          training_pesticide_predict_tensor.detach().numpy())
            mse = mean_squared_error(training_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSE
                                     training_pesticide_predict_tensor.detach().numpy())
            log_50, log_70, log_100 = log_dif(training_pesticide_observe_tensor.detach().numpy(),  # 评价指标 log
                                              training_pesticide_predict_tensor.detach().numpy())
            #
            nse_2 = r2_score(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 NSE
                             validation_pesticide_predict_tensor.detach().numpy())
            r2_2 = r2_score_simulate(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 R2
                                     validation_pesticide_predict_tensor.detach().numpy())
            tic_2 = tic_simulate(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 TIC
                                 validation_pesticide_predict_tensor.detach().numpy())
            mae_2 = mean_absolute_error(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MAE
                                        validation_pesticide_predict_tensor.detach().numpy())
            msle_2 = mean_squared_log_error(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSLE
                                            validation_pesticide_predict_tensor.detach().numpy())
            mse_2 = mean_squared_error(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSE
                                       validation_pesticide_predict_tensor.detach().numpy())
            log_50_2, log_70_2, log_100_2 = log_dif(validation_pesticide_observe_tensor.detach().numpy(),  # 评价指标 log
                                                    validation_pesticide_predict_tensor.detach().numpy())
            #
            nse_3 = r2_score(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 NSE
                             compare_pesticide_predict_tensor.detach().numpy())
            r2_3 = r2_score_simulate(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 R2
                                     compare_pesticide_predict_tensor.detach().numpy())
            tic_3 = tic_simulate(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 TIC
                                 compare_pesticide_predict_tensor.detach().numpy())
            mae_3 = mean_absolute_error(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MAE
                                        compare_pesticide_predict_tensor.detach().numpy())
            msle_3 = mean_squared_log_error(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSLE
                                            compare_pesticide_predict_tensor.detach().numpy())
            mse_3 = mean_squared_error(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 MSE
                                       compare_pesticide_predict_tensor.detach().numpy())
            log_50_3, log_70_3, log_100_3 = log_dif(compare_pesticide_observe_tensor.detach().numpy(),  # 评价指标 log
                                                    compare_pesticide_predict_tensor.detach().numpy())
            write_statistic_data(pesticide_output_2_params_path, [str(epoch)] +
                                 [str(parameters.item()) for name, parameters in
                                  pesticidePredict.named_parameters()] +
                                 [str(tic), str(msle), str(log_50), str(log_70), str(log_100), str(r2), str(nse),
                                  str(mae),
                                  str(mse)])
            write_statistic_data(pesticide_output_2_params_path, [str(epoch)] +
                                 ["" for name, parameters in
                                  pesticidePredict.named_parameters()] +
                                 [str(tic_2), str(msle_2), str(log_50_2), str(log_70_2), str(log_100_2), str(r2_2),
                                  str(nse_2), str(mae_2),
                                  str(mse_2)])
            write_statistic_data(pesticide_output_2_params_path, [str(epoch)] +
                                 ["" for name, parameters in
                                  pesticidePredict.named_parameters()] +
                                 [str(tic_3), str(msle_3), str(log_50_3), str(log_70_3), str(log_100_3), str(r2_3),
                                  str(nse_3), str(mae_3),
                                  str(mse_3)])

            print("training nse:", str(nse), "R2:", str(r2), "TIC:", str(tic), "MSLE:", str(msle), "log_50:",
                  str(log_50),
                  "log_70%:", str(log_70), "log_100", str(log_100), "MSE:", str(mse), "MAE:", str(mae))
            print("validate nse:", str(nse_2), "R2:", str(r2_2), "TIC:", str(tic_2), "MSLE:", str(msle_2), "log_50:",
                  str(log_50_2),
                  "log_70:", str(log_70_2), "log_100", str(log_100_2), "MSE:", str(mse_2), "MAE:", str(mae_2))
            print("compare nse:", str(nse_3), "R2:", str(r2_3), "TIC:", str(tic_3), "MSLE:", str(msle_3), "log_50:",
                  str(log_50_3),
                  "log_70:", str(log_70_3), "log_100", str(log_100_3), "MSE:", str(mse_3), "MAE:", str(mae_3))
            print("---------------------------")
            # 优化
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 记录最后一次训练的结果  total - compare - training - validate
            if epoch == pesticide_epochs:
                # 对数差
                total_pesticde_log_diff_tensor = torch.abs(torch.log10(total_pesticide_output_tensor + 1e-5)
                                                           - torch.log10(total_pesticide_observe_tensor + 1e-5))
                compare_pesticde_log_diff_tensor = torch.abs(torch.log10(compare_pesticide_predict_tensor + 1e-5)
                                                             - torch.log10(compare_pesticide_observe_tensor + 1e-5))
                training_pesticde_log_diff_tensor = torch.abs(torch.log10(training_pesticide_predict_tensor + 1e-5)
                                                              - torch.log10(training_pesticide_observe_tensor + 1e-5))
                validation_pesticde_log_diff_tensor = torch.abs(torch.log10(validation_pesticide_predict_tensor + 1e-5)
                                                                - torch.log10(
                    validation_pesticide_observe_tensor + 1e-5))
                # 列名
                total_columns = ["time", "pesticide", "observed", "runflow", "prec", "accumulation", "accumulation_sub",
                                 "log_diff",
                                 "area_id"]
                compare_columns = ["time", "pesticide_pred", "pesticde_observe", "runflow_pred", "runflow_observe",
                                   "prec", "accumulation", "accumulation_sub", "log_diff", "area_id"]
                training_columns = ["time", "pesticide_pred", "pesticde_observe", "runflow_pred", "runflow_observe",
                                    "prec", "accumulation", "accumulation_sub", "log_diff", "area_id"]
                validation_columns = ["time", "pesticide_pred", "pesticde_observe", "runflow_pred", "runflow_observe",
                                      "prec", "accumulation", "accumulation_sub", "log_diff", "area_id"]
                pesticide_output_2_total_path = os.path.join(pesticide_output_2_dir,
                                                             "{}_total.csv".format(pesticide_name))
                pesticide_output_2_compare_path = os.path.join(pesticide_output_2_dir,
                                                               "{}_compare.csv".format(pesticide_name))
                pesticide_output_2_training_path = os.path.join(pesticide_output_2_dir,
                                                                "{}_training.csv".format(pesticide_name))
                pesticide_output_2_validation_path = os.path.join(pesticide_output_2_dir,
                                                                  "{}_validation.csv".format(pesticide_name))
                if os.path.exists(pesticide_output_2_total_path):
                    os.remove(pesticide_output_2_total_path)
                if os.path.exists(pesticide_output_2_compare_path):
                    os.remove(pesticide_output_2_compare_path)
                if os.path.exists(pesticide_output_2_training_path):
                    os.remove(pesticide_output_2_training_path)
                if os.path.exists(pesticide_output_2_validation_path):
                    os.remove(pesticide_output_2_validation_path)
                write_statistic_data(pesticide_output_2_total_path, total_columns)
                write_statistic_data(pesticide_output_2_compare_path, compare_columns)
                write_statistic_data(pesticide_output_2_training_path, training_columns)
                write_statistic_data(pesticide_output_2_validation_path, validation_columns)
                # 写入总体的结果和比较的结果
                for i in range(len(total_time_str_list)):
                    write_statistic_data(pesticide_output_2_total_path,
                                         [str(total_time_str_list[i]),
                                          str(total_pesticide_output_tensor[i].item()),
                                          str(total_pesticide_observe_tensor[i].item()),
                                          str(total_runflow_tensor[i].item()),
                                          str(total_prec_tensor[i].item()),
                                          str(total_accumulation_tensor[i].item()),
                                          str(total_accumulation_sub_tensor[i].item()),
                                          str(total_pesticde_log_diff_tensor[i].item()),
                                          str(total_area_id_list[i])])
                for i in range(len(compare_time_str_list)):
                    write_statistic_data(pesticide_output_2_compare_path,
                                         [str(compare_time_str_list[i]),
                                          str(compare_pesticide_predict_tensor[i].item()),
                                          str(compare_pesticide_observe_tensor[i].item()),
                                          str(compare_runflow_predict_tensor[i].item()),
                                          str(compare_runflow_observe_tensor[i].item()),
                                          str(compare_prec_tensor[i].item()),
                                          str(compare_accumulation_tensor[i].item()),
                                          str(compare_accumulation_sub_tensor[i].item()),
                                          str(compare_pesticde_log_diff_tensor[i].item()),
                                          str(compare_area_id_list[i])])
                for i in range(len(training_time_str_list)):
                    write_statistic_data(pesticide_output_2_training_path,
                                         [str(training_time_str_list[i]),
                                          str(training_pesticide_predict_tensor[i].item()),
                                          str(training_pesticide_observe_tensor[i].item()),
                                          str(training_runflow_predict_tensor[i].item()),
                                          str(training_runflow_observe_tensor[i].item()),
                                          str(training_prec_tensor[i].item()),
                                          str(training_accumulation_tensor[i].item()),
                                          str(training_accumulation_sub_tensor[i].item()),
                                          str(training_pesticde_log_diff_tensor[i].item()),
                                          str(training_area_id_list[i])])
                for i in range(len(validation_time_str_list)):
                    write_statistic_data(pesticide_output_2_validation_path,
                                         [str(validation_time_str_list[i]),
                                          str(validation_pesticide_predict_tensor[i].item()),
                                          str(validation_pesticide_observe_tensor[i].item()),
                                          str(validation_runflow_predict_tensor[i].item()),
                                          str(validation_runflow_observe_tensor[i].item()),
                                          str(validation_prec_tensor[i].item()),
                                          str(validation_accumulation_tensor[i].item()),
                                          str(validation_accumulation_sub_tensor[i].item()),
                                          str(validation_pesticde_log_diff_tensor[i].item()),
                                          str(validation_area_id_list[i])])
            # break
        break
