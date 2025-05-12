class parameters:
    # 原始数据目录
    rainfall_dir = ".\\data\\1_rainfall"
    rainfall_ex_dir = ".\\data\\1_rainfall_ex"
    runflow_dir = ".\\data\\2_runflow"
    runflow_ex_dir = ".\\data\\2_runflow_ex"
    area_dir = ".\\data\\3_area"
    pesticide_dir = "data\\4_pesticide"

    # 输出结果目录
    rainfall_output_dir = ".\\1_rainfall"
    runflow_output_dir = ".\\2_runflow"
    train_output_dir = ".\\3_runflow_train"
    predicted_runflow_output_dir = ".\\4_predicted_runflow"
    pesticide_output_dir = ".\\5_pesticide"
    analysis_output_dir = ".\\6_analysis"

    # 参数
    area_id = [870, 3718, 3716]  # 学校 工业 住宿
    area_ex_id = [2705, 3723]  # 黄埔领军人才区域 从化区溪流河公园
    xinzhi_id = [214, 215, 216]  # 学校 工业 住宿
    xinzhi_ex_id = [135, 217]  # 黄埔领军人才区域 从化区溪流河公园
    xinzhiId_to_areaId = {
        "214": 870,
        "215": 3718,
        "216": 3716,
        "135": 2705,
        "217": 3723
    }
    areaId_to_xinzhiId = {
        "870": 214,
        "3718": 215,
        "3716": 216,
        "2705": 135,
        "3723": 217
    }

    # 流量数据训练参数
    runflow_learning_rate = 1e-5
    runflow_epochs = 60

    # 杀虫剂
    pesticide_names = ["THM", "CLO", "IMI", "ACE", "THA",  # 噻虫嗪 噻虫胺 吡虫啉 啶虫脒 噻虫啉
                       "Propoxur", "Chlorpyrifos", "Temephos", "Fenthion", "Fipronil",  # 残杀威 毒死蜱 双硫磷 倍硫磷 氟虫腈
                       "Tetramethrin", "Cyhalothrin_Lambda", "Cyfluthrin", "Permethrin", "Cypermethrin"]  # 胺菊酯 高效氯氟氰菊酯 氟氯氰菊酯 氯菊酯 氯氰菊酯

