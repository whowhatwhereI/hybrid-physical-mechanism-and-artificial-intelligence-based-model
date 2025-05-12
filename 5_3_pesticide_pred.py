import datetime

import numpy as np
import os
import pandas as pd

from _parameters import parameters
from _method import write_statistic_data
from _algorithm import PesticideTotalDataSet, PesticideTotalPredict

if __name__ == '__main__':
    # pesticide
    pesticide_dir = parameters.pesticide_dir
    pesticide_usage_path = os.path.join(pesticide_dir, "usage.csv")
    pesticide_output_dir = parameters.pesticide_output_dir
    pesticide_output_3_dir = os.path.join(pesticide_output_dir, "3_pred")
    # runflow - predict
    predicted_runflow_output_dir = parameters.predicted_runflow_output_dir
    predicted_runflow_output_2_dir = os.path.join(predicted_runflow_output_dir, "2_runflow")
    # area
    area_dir = parameters.area_dir
    area_path = os.path.join(area_dir, "area.csv")

    # 针对不同灭蚊剂进行训练
    # pesticide_names = ["THM", "CLO", "IMI", "ACE", "THA",  # 噻虫嗪 噻虫胺 吡虫啉 啶虫脒 噻虫啉
    #                    "Propoxur", "Chlorpyrifos", "Temephos", "Fenthion", "Fipronil",  # 残杀威 毒死蜱* 双硫磷* 倍硫磷 氟虫腈*
    #                    "Tetramethrin", "Cyhalothrin_Lambda", "Cyfluthrin", "Permethrin", "Cypermethrin"]  # 胺菊酯 高效氯氟氰菊酯0 氟氯氰菊酯 氯菊酯 氯氰菊酯
    pesticide_names = ["THM", "CLO", "IMI", "ACE", "THA",  # 噻虫嗪 噻虫胺 吡虫啉 啶虫脒 噻虫啉
                       "Propoxur", "Fenthion",  # 残杀威 倍硫磷
                       "Tetramethrin", "Cyfluthrin", "Permethrin", "Cypermethrin"]  # 胺菊酯 氟氯氰菊酯 氯菊酯 氯氰菊酯
    loadPesticideTotalDataSet = PesticideTotalDataSet(area_path=area_path,
                                                      pesticide_usage_path=pesticide_usage_path,
                                                      runflow_dir=predicted_runflow_output_2_dir)

    pesticide_params = {
        "ACE": [3.042378887, 0.969612828, 0.898951067, 0.602530681, 0.972280942, 1.045949834, 0.007563532, 0.298622065, 0.449314212, 0.410021468],
        "CLO": [3.047816148, 0.911445504, 0.897950773, 1.102155271, 2.107019419, 2.68972258, 0.467461415, 0.342354216, 0.438264908, 0.007375715],
        "Cyfluthrin": [3.042140126, 0.946894456, 0.87119264, 1.0, 0.898978265, 1.49261361, 0.169763065, 0.294921535, 0.418784753, 0.53031636],
        "Cypermethrin": [3.042186653, 0.917727925, 0.893584013, 5.531119882, 9.901672764, 15.1930386, 0.226591965, 0.273573946, 0.398803609,
                         0.550287155],
        "Fenthion": [3.042190558, 0.967087225, 0.895101099, 0.90475652, 0.90181869, 1.093061567, 0.127703707, 0.392760737, 0.448764425, 0.410350327],
        "IMI": [3.042086175, 0.99, 0.934628333, 0.575482949, 1.063201033, 0.961637368, 0.044872836, 0.229753836, 0.447693096, 0.414190324],
        "Permethrin": [3.042178797, 0.881193148, 0.866875762, 1.431168233, 0.897792328, 1.19238791, 0.187422627, 0.260573706, 0.428795692,
                       0.630293075],
        "Propoxur": [3.042221956, 0.972472867, 0.901128932, 1.504812974, 52.17314907, 40.8029497, 0.133082765, 0.389984616, 0.448759394, 0.410353024],
        "Tetramethrin": [3.042190558, 0.96731354, 0.895373542, 2.004757731, 45.09306951, 45.90185839, 0.187983188, 0.382647972, 0.418764382,
                         0.530350368],
        "THA": [3.042186572, 0.985254168, 0.915784153, 1.504898597, 0.802564097, 1.173313333, 0.012364454, 0.284305355, 0.448626146, 0.410021468],
        "THM": [3.050180565, 0.941846794, 0.904118423, 1.084408191, 1.587183106, 2.114010233,0.472591132, 0.407377747, 0.445718541, 0.006676793],
    }
    # 遍历每个区域
    areas_id = range(1, 3722 + 1)
    for area_id in areas_id:
        pesticide_output_3_path = os.path.join(pesticide_output_3_dir, "predicted_C_{}.csv".format(str(area_id)))
        # 模拟每个区域的每个灭蚊剂的浓度
        runflow_data, usage_data = loadPesticideTotalDataSet[area_id]
        print(runflow_data)
        # 记录 时间、流量、雨量
        time_np = runflow_data.loc[:, "time"].values  # [n]
        runflow_np = runflow_data.loc[:, "runflow"].values  # [n]
        prec_np = runflow_data.loc[:, "prec"].values  # [n]
        time_np = time_np.reshape(len(time_np), 1)  # [n, 1]
        runflow_np = runflow_np.reshape(len(runflow_np), 1)  # [n, 1]
        prec_np = prec_np.reshape(len(prec_np), 1)  # [n, 1]
        output_data = np.concatenate((time_np, prec_np, runflow_np), axis=1)
        # print(output_data.shape)
        # 遍历模拟灭蚊剂
        for pesticide_name in pesticide_names:
            # print(runflow_data)

            pesticideTotalPredict = PesticideTotalPredict()
            total_pesticide_output_np = pesticideTotalPredict(runflow_data=runflow_data,
                                                              usage_data=usage_data,
                                                              pesticide_name=pesticide_name,
                                                              params=pesticide_params[pesticide_name])
            total_pesticide_output_np = total_pesticide_output_np.reshape(len(total_pesticide_output_np), 1)
            output_data = np.concatenate((output_data, total_pesticide_output_np), axis=1)
        write_statistic_data(pesticide_output_3_path, ["time", "prec", "runflow"] + [str(pesticide_name) for pesticide_name in pesticide_names])
        for i in range(len(output_data)):
            write_statistic_data(pesticide_output_3_path, [str(output_data_) for output_data_ in output_data[i]])
