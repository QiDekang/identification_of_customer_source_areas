###############################################################################
# 部署必备
import os    # 必需的包，勿删除
import sys   # 必需的包，勿删除
import glob  # 必需的包，勿删除

###############################################################################
# 以下为用户自定义依赖包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, OPTICS
import random as rn
import time
import json
from sklearn.feature_extraction.text import TfidfTransformer
import multiprocessing
import math
from math import radians, cos, sin, asin, sqrt

###############################################################################
#######  定义常用函数
def geo_distance(lat1, lng1, lat2, lng2):
    
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    
    return distance

def data_filter_3km(train_data, service_bus_id):

    # 中心点坐标
    center = [[100001,116.4188178,39.919721],
            [100002,116.4825568,39.91598925],
            [100003,116.3804663,39.915559],
            [100004,116.4651895,39.9156885],
            [100005,116.45222,39.9277485],
            [100006,116.4572058,39.93927325],
            [100007,116.4238158,39.90459875],
            [100008,116.4516563,39.915751],
            [100009,116.45442,39.9792865],
            [100010,116.4871475,39.9495125],
            [100011,116.4740715,39.9476425],
            [100012,116.5015485,39.97913675],
            [100013,116.47115,39.99876075],
            [100014,116.4141475,40.002097],
            [100015,116.466728,39.900057],
            [100016,116.488916,39.8488705],
            [100017,116.425078,39.96248275],
            [100018,116.436449,39.9138235],
            [100019,116.439675,39.946997],
            [100020,116.4046815,39.902885],
            [100021,116.4123868,39.88803525],
            [100022,116.3658798,39.91291875],
            [100023,116.3877638,39.8948345],
            [100024,116.382381,39.905321],
            [100025,116.342241,39.941554],
            [100026,116.3220475,39.986847],
            [100027,116.3437205,39.99705025],
            [100028,116.2906773,39.96421675],
            [100029,116.5076143,39.80454575],
            [100030,116.346155,39.49673075],
            [100031,116.6679218,39.8840825],
            [100032,116.4278198,40.076865],
            [100033,116.664185,39.85794575]]
    center_array = np.array(center)
    # 去掉附件3公里数据
    train_data_no_arround = pd.DataFrame(columns=train_data.columns.values.tolist())
    for i in range(0, len(train_data)):
        distance = geo_distance(train_data.loc[i, "lat_bd"], train_data.loc[i, "lon_bd"], center_array[service_bus_id, 2], center_array[service_bus_id, 1])
        if distance > 3.0:
            train_data_no_arround = train_data_no_arround.append(train_data.loc[i, :], ignore_index=True)
    train_data_no_arround = train_data_no_arround.reset_index(drop=True)  #  重设索引
    print('len(train_data_no_arround)', len(train_data_no_arround))

    return train_data_no_arround

def data_filter(train_data, sample_len):

    # 筛选5环数据
    #train_data_5 =  train_data[(train_data["lon_bd"] > 116.217306) & (train_data["lon_bd"] < 116.554782) & (train_data["lat_bd"] > 39.766448) & (train_data["lat_bd"] < 40.029051)]
    # 为包含通州，需要筛选六环数据
    train_data_5 =  train_data[(train_data["lon_bd"] > 116.159786) & (train_data["lon_bd"] < 116.714723) & (train_data["lat_bd"] > 39.706535) & (train_data["lat_bd"] < 40.164284)]
    train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引
    print('len(train_data_5)', len(train_data_5))
    # 数据过多则抽样固定大小数据，避免崩溃
    if len(train_data_5) > sample_len:
        train_data_5 = train_data_5.sample(n=sample_len, axis=0, random_state=123)
        train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引
    
    return train_data_5

### 还原到数据点
def data_recovery(train_data_5, sample_len):

    # 每个格子人数在1-100之间
    train_data_5.loc[:, "indx_avg_pin"] = (100 * train_data_5["indx_avg_pin"]) / max(train_data_5["indx_avg_pin"])
    #  去掉不足一个人的格子
    train_data_5 = train_data_5[(train_data_5["indx_avg_pin"] >= 1)]
    train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引
    #  人数均值, 避免人数过大，恢复出太多数据。
    pin_average = train_data_5["indx_avg_pin"].mean()
    #  抽样35000 / 4000 = 8.75
    if pin_average > 8.75:
        scale = pin_average / 8.75
        train_data_5["indx_avg_pin"] = train_data_5["indx_avg_pin"] / scale
        # 向上取整，保证最小的格子有一个人
        #train_data_5["indx_avg_pin"] = math.ceil(train_data_5["indx_avg_pin"])
        #  去掉不足一个人的格子
        train_data_5 = train_data_5[(train_data_5["indx_avg_pin"] >= 1)]
        train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引
    #  保留0小数
    #train_data_5.loc[:, "indx_avg_pin"] = round(train_data_5.loc[:, "indx_avg_pin"], 0)
    train_data_5["indx_avg_pin"] = train_data_5["indx_avg_pin"].round(0)
    train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引
    ####  恢复到原始点
    list_range = range(-75, 75) # 栅格150*150米，即正负75
    lat_one_meter = 0.00000898 # 一米对应纬度
    lng_one_meter = 0.00000941 # 一米对应经度
    train_data_ori = pd.DataFrame(columns=train_data_5.columns.values.tolist())
    for i in range(0, len(train_data_5["indx_avg_pin"])):
        if i%1000 == 0:
            print('data_recovery:', i)
        for j in range(0, int(train_data_5.iloc[i, :]["indx_avg_pin"])):
            #current = train_data_5.iloc[i, :][["geohash7_bd", "lon_bd", "lat_bd"]]
            current = train_data_5.iloc[i, :]
            current_copy = current.copy()
            #current_copy.loc["lon_bd"] = current_copy.loc["lon_bd"] + j * 0.00000001
            #current_copy.loc["lat_bd"] = current_copy.loc["lat_bd"] + j * 0.00000001
            # random
            random_lng = rn.sample(list_range, 1)[0]
            random_lat = rn.sample(list_range, 1)[0]
            current_copy.loc["lon_bd"] = current_copy.loc["lon_bd"] + random_lng * lng_one_meter
            current_copy.loc["lat_bd"] = current_copy.loc["lat_bd"] + random_lat * lat_one_meter
            #train_data_ori.iloc[i, 0] = train_data_ori.iloc[i, 0] + random_lng * lng_one_meter
            #train_data_ori.iloc[i, 1] = train_data_ori.iloc[i, 1] + random_lat * lat_one_meter
            train_data_ori = train_data_ori.append(current_copy, ignore_index=True)
    #print('train_data_ori', train_data_ori)

    # 数据过多则抽样固定大小数据，避免时间过长
    if len(train_data_ori) > sample_len:
        train_data_ori = train_data_ori.sample(n=sample_len, axis=0, random_state=123)
        train_data_ori = train_data_ori.reset_index(drop=True)  #  重设索引

    # 判读是否有重复数据
    result1=train_data_ori.duplicated()
    np.unique(result1)
    
    return train_data_ori

### 还原到数据点
def data_recovery_in(train_data_5, sample_len):

    ###  商圈内格子太少，如果采用每个格子最多有100人（除以最大值 * 100 之后 再去除小于1的格子），则会去除非常多数据，本来数据就少，容易出问题
    ###  使用除以最小值的，每个格子最少有一个人。
    # 除以最小值
    train_data_5.loc[:, "indx_avg_pin"] = train_data_5["indx_avg_pin"] / min(train_data_5["indx_avg_pin"])
    #  保留0小数
    train_data_5["indx_avg_pin"] = train_data_5["indx_avg_pin"].round(0)
    train_data_5 = train_data_5.reset_index(drop=True)  #  重设索引

    ####  恢复到原始点
    #  只需要重要的列
    list_range = range(-75, 75) # 栅格150*150米，即正负75
    lat_one_meter = 0.00000898 # 一米对应纬度
    lng_one_meter = 0.00000941 # 一米对应经度
    train_data_ori = pd.DataFrame(columns=train_data_5.columns.values.tolist())
    for i in range(0, len(train_data_5["indx_avg_pin"])):
        if i%1000 == 0:
            print('data_recovery:', i)
        for j in range(0, int(train_data_5.iloc[i, :]["indx_avg_pin"])):
            #current = train_data_5.iloc[i, :][["geohash7_bd", "lon_bd", "lat_bd"]]
            current = train_data_5.iloc[i, :]
            current_copy = current.copy()
            #current_copy.loc["lon_bd"] = current_copy.loc["lon_bd"] + j * 0.00000001
            #current_copy.loc["lat_bd"] = current_copy.loc["lat_bd"] + j * 0.00000001
            # random
            random_lng = rn.sample(list_range, 1)[0]
            random_lat = rn.sample(list_range, 1)[0]
            current_copy.loc["lon_bd"] = current_copy.loc["lon_bd"] + random_lng * lng_one_meter
            current_copy.loc["lat_bd"] = current_copy.loc["lat_bd"] + random_lat * lat_one_meter

            train_data_ori = train_data_ori.append(current_copy, ignore_index=True)
    #print('train_data_ori', train_data_ori)

    # 数据过多则抽样固定大小数据，避免崩溃
    if len(train_data_ori) > sample_len:
        train_data_ori = train_data_ori.sample(n=sample_len, axis=0, random_state=123)
        train_data_ori = train_data_ori.reset_index(drop=True)  #  重设索引

    # 判读是否有重复数据
    result1=train_data_ori.duplicated()
    np.unique(result1)
    
    return train_data_ori

def TF_IDF_transform(result_data_top_10_unique_all_fields):

    ##### 计算top-10类画像TF-IDF均值，并降维
    ##### json转换
    base_data = result_data_top_10_unique_all_fields
    useful_data = base_data
    feature_name_list = []
    feature_data = pd.DataFrame(columns=feature_name_list)
    portrayal_data = pd.concat([useful_data, feature_data], axis=1) #  数据拼接
    portrayal_data[feature_name_list] = 0
    for i in range(0, len(base_data)):
        if i%1000 == 0:
            print('TF_IDF_transform', i)
        if pd.isnull(base_data.loc[i, "portrayal_json"]) == False:
            portrayal_dic = json.loads(base_data.loc[i, "portrayal_json"])
            for category_key in portrayal_dic.keys():
                current_category = portrayal_dic[category_key][0]
                current_sum = 0
                for feature_key in current_category.keys():
                    feature_name = category_key + '_' + feature_key
                    if feature_name in feature_name_list:
                      portrayal_data.loc[i, feature_name] = current_category[feature_key]
                      current_sum = current_sum + portrayal_data.loc[i, feature_name]
                    #  字段命名异常值
                    else:
                        print(i)
                        print(feature_name)
                # 百分比确认
                if current_sum != 1:
                    for feature_key in current_category.keys():
                        feature_name = category_key + '_' + feature_key
                        portrayal_data.loc[i, feature_name] = portrayal_data.loc[i, feature_name] / current_sum
    # TF IDF
    #类调用
    transformer = TfidfTransformer()  
    #print('transformer', transformer)
    #将词频矩阵X统计成TF-IDF值  
    portrayal_details = portrayal_data[feature_name_list]
    tfidf = transformer.fit_transform(portrayal_details)  
    #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
    #print('tfidf.toarray()', tfidf.toarray())
    # 数据转换
    tfidf_score = pd.DataFrame(tfidf.toarray(), columns=feature_name_list)
    portrayal_data_tfidf = pd.concat([useful_data, tfidf_score], axis=1) #  数据拼接

    return portrayal_data_tfidf, feature_name_list

def TF_IDF_transform_novisit(train_data_5_novisit):

    # 有用的字段
    base_data = train_data_5_novisit
    #useful_data = base_data[["geohash7_bd", "lon_bd", "lat_bd", "indx_avg_pin"]]
    useful_data = base_data
    feature_name_list = []
    feature_data = pd.DataFrame(columns=feature_name_list)
    portrayal_data = pd.concat([useful_data, feature_data], axis=1) #  数据拼接
    portrayal_data[feature_name_list] = 0
    for i in range(0, len(base_data)):
        if i%1000 == 0:
            print('TF_IDF_transform_novisit', i)
        if pd.isnull(base_data.loc[i, "portrayal_json"]) == False:
            portrayal_dic = json.loads(base_data.loc[i, "portrayal_json"])
            for category_key in portrayal_dic.keys():
                current_category = portrayal_dic[category_key][0]
                current_sum = 0
                for feature_key in current_category.keys():
                    # credit_level/credi没有统一
                    if category_key == "credit":
                        category_key = "credit_level"
                    if category_key == "education":
                        category_key = "education_feat02"
                    if category_key == "marriage":
                        category_key = "marriage_feat02"
                    feature_name = category_key + '_' + feature_key
                    if feature_name in feature_name_list:
                      portrayal_data.loc[i, feature_name] = current_category[feature_key]
                      current_sum = current_sum + portrayal_data.loc[i, feature_name]
                    #  字段命名异常值
                    else:
                        print(i)
                        print(feature_name)
                # 百分比确认
                if current_sum != 1:
                    for feature_key in current_category.keys():
                        feature_name = category_key + '_' + feature_key
                        portrayal_data.loc[i, feature_name] = portrayal_data.loc[i, feature_name] / current_sum

    # TF IDF
    from sklearn.feature_extraction.text import TfidfTransformer
    #类调用  
    transformer = TfidfTransformer()  
    #print('transformer', transformer)
    #将词频矩阵X统计成TF-IDF值  
    portrayal_details = portrayal_data[feature_name_list]
    tfidf = transformer.fit_transform(portrayal_details)  
    #查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
    #print('tfidf.toarray()', tfidf.toarray())
    # 数据转换
    tfidf_score = pd.DataFrame(tfidf.toarray(), columns=feature_name_list)
    # 保留前20个特征在main中自行处理
    #tfidf_score_20_novisit = tfidf_score[tfidf_top_20_feature_name]
    #portrayal_data_tfidf_novisit = pd.concat([useful_data, tfidf_score_20_novisit], axis=1) #  数据拼接
    portrayal_data_tfidf_novisit = pd.concat([useful_data, tfidf_score], axis=1) #  数据拼接

    return portrayal_data_tfidf_novisit, feature_name_list

######  计算相似度
def cos_sim(vector_a, vector_b):

    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    sim = num / denom

    return sim

### 聚类
def OPTICS_core(ori_people_data, lower_boundary, upper_boundary, lower_r, upper_r, gap_r, lower_n, upper_n, gap_n):

    ## 数据准备
    sample_data = ori_people_data
    print('len(sample_data)',len(sample_data))
    sample_data = sample_data.reset_index(drop=True)  #  重设索引
    # OPTICS
    dbsacn_data = sample_data[["lon_bd", "lat_bd"]]
    kms_per_radian = 6371.0088
    for r_10 in range(lower_r, upper_r, gap_r):
        for n in range(lower_n, upper_n, gap_n): # 100, 1500, 50
            #r = 0.3
            #n = 300
            r = r_10 / 10
            epsilon = r / kms_per_radian
            starttime = time.time()
            y_pred = OPTICS(max_eps=epsilon, min_samples=n, algorithm='ball_tree', metric='haversine').fit_predict(np.radians(dbsacn_data))
            endtime = time.time()
            dtime = endtime - starttime
            print("程序运行时间：%.8s s" % dtime)  #显示到微秒
            # 聚类数量
            class_number = len(np.unique(y_pred))-1
            print('r:', r)
            print('n:', n)
            print('class_number:', class_number)
            if class_number >= lower_boundary and class_number <= upper_boundary:
                #  作为最终聚类结果
                result_data = sample_data
                class_data = pd.DataFrame(y_pred, columns=["class"])
                result_data = pd.concat([result_data, class_data], axis=1)
                return class_data, result_data, class_number
    #return class_data, result_data, class_number

### 聚类 筛选前十
def cluster_top_10(result_data, class_number):

    ### 筛选前十
    density = np.zeros((class_number, 2), dtype=np.float)
    for i in range(0, class_number):
        current_class = result_data[(result_data["class"] == i)]
        density[i, 0] = i
        density[i, 1] = len(current_class["geohash7_bd"]) / len(np.unique(current_class["geohash7_bd"])) #密度
    # 排序
    density_data = pd.DataFrame(density, columns=["class", "density"])
    sorted_density_data = density_data.sort_values(by='density', ascending=False)
    sorted_density_data = sorted_density_data.reset_index(drop=True)  #  重设索引

    # 保留前十
    # 此时不是左闭右开的
    if class_number < 10:
        class_top_10 = sorted_density_data.loc[:, "class"].values
    else:
        class_top_10 = sorted_density_data.loc[0:9, "class"].values
        if len(class_top_10) < 10:
            class_top_10 = sorted_density_data.loc[0:10, "class"].values

    # 前十
    result_data_top_10 = pd.DataFrame(columns=result_data.columns.values.tolist())
    for i in range(0, len(result_data)):
        if result_data.loc[i, "class"] in class_top_10:
            class_top_10_list = class_top_10.tolist()
            rank = class_top_10_list.index(result_data.loc[i, "class"]) # 密度排名
            current_row = result_data.loc[i, :]
            current_row_copy = current_row.copy()
            current_row_copy.loc['class'] = rank
            #result_data_top_10 = result_data_top_10.append(result_data.loc[i, :], ignore_index=True)
            result_data_top_10 = result_data_top_10.append(current_row_copy, ignore_index=True)
    #plt.scatter(result_data_top_10["lon_bd"], result_data_top_10["lat_bd"], c=result_data_top_10["class"])
    #plt.show()

    # 保留前十
    #class_top_10 = sorted_density_data.loc[0:10, "class"].values
    #class_top_N = sorted_density_data.loc[10:class_number, "class"].values

    # 画图
    #result_data_drop = result_data[(result_data["class"] > -1)]
    #result_data_out = result_data[(result_data["class"] < 0)]
    #plt.scatter(result_data_out["lon_bd"], result_data_out["lat_bd"], c='darkgray')
    #plt.scatter(result_data_drop["lon_bd"], result_data_drop["lat_bd"], c=result_data_drop["class"])
    #plt.show()

    # 前十画图N
    #print('客流洞察')
    #print('聚类 top 10')
    #result_data_top_10 = pd.DataFrame(columns=result_data.columns.values.tolist())
    #result_data_top_N = pd.DataFrame(columns=result_data.columns.values.tolist())
    #for i in range(0, len(result_data)):
    #    if result_data.loc[i, "class"] in class_top_10:
    #        result_data_top_10 = result_data_top_10.append(result_data.loc[i, :], ignore_index=True)
    #    if result_data.loc[i, "class"] in class_top_N:
    #        result_data_top_N = result_data_top_N.append(result_data.loc[i, :], ignore_index=True)
    #plt.scatter(result_data_top_N["lon_bd"], result_data_top_N["lat_bd"], c='darkgray')
    #plt.scatter(result_data_top_10["lon_bd"], result_data_top_10["lat_bd"], c=result_data_top_10["class"])
    #plt.show()

    # 前十，格子形式，geohash7_bd去重
    result_data_top_10_unique = result_data_top_10.drop_duplicates(subset=["geohash7_bd"], keep='first', inplace=False)
    result_data_top_10_unique = result_data_top_10_unique.reset_index(drop=True)  #  重设索引

    return result_data_top_10_unique

###############################################################################

#  循环体
def all_business_district(service_bus_id, addr_id, is_workday_id, dt, feature_list_to_file):

    try:

        print('service_bus_id, addr_id, is_workday_id', service_bus_id, addr_id, is_workday_id)

        #dt = '2020-07-18'
        inout = '1' # 商圈外
        service_bus_code = str(100001 + service_bus_id)
        if addr_id == 0:
            addr_type="到访"
        if addr_id == 1:
            addr_type="居住"
        if addr_id == 2:
            addr_type="工作"
        if is_workday_id == 0:
            is_workday="weekday"
        if is_workday_id == 1:
            is_workday="weekend"

        # 文件只写一次，否则会覆盖掉
        #feature_list_to_file = ["geohash7_bd", "service_bus_code", "is_visit", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "is_center", "class", "dt"]
        save_file = pd.DataFrame(columns=feature_list_to_file)

        ###############################################################################
        # 创建输入数据文件夹 【无需修改】
        input_dir = 'inputs_customer' + '_' + str(service_bus_id) + '_' + str(addr_id) + '_' + str(is_workday_id)
        input_dir = 'data/' + input_dir  # 拼上data文件夹，作为子目录
        print("Create directory begin:")
        if not os.path.exists(input_dir):  # 判断是否存在文件夹如果不存在则创建文件夹
            os.makedirs(input_dir)
        else:
            cmd = 'rm -rf ' + input_dir
            os.system(cmd)
            os.makedirs(input_dir)
        print("Create directory end!")
        # 读取数据 【需要修改：用户自定义sql部分，只需要修改select至""""之间的部分】

        #####  从数据库读数据
        input_file_path = input_dir+'/input.txt'  #  数据库文件存储
        read_data_sql = """hive -e "set hive.cli.print.header=true;
        select
        *
        from dmu_uc.dmuuc_sl_visit_thermal_portrayal_distribution_i_d where service_bus_code='{service_bus_code}' and inout='{inout}' and addr_type='{addr_type}' and is_workday='{is_workday}' and dt='{dt}'
        " > {path}
        """.format(service_bus_code = service_bus_code, inout = inout, addr_type = addr_type, is_workday = is_workday, dt = dt, path=input_file_path)
        res_sql = os.system(read_data_sql)

        if res_sql == 0:
            print("Read data done!")
        else:
            raise ValueError("输入数据读取失败!")

        print("Read data done!")

        ###############################################################################
        ##### 现有客流
        ##### 数据处理
        #train_df = pd.read_csv('data/inputs_customer/input.txt', sep='\t')
        train_df = pd.read_csv(input_file_path, sep='\t')
        #print('train_df', train_df)
        #  筛选5环数据、抽样(避免数据过多，时间过长)
        train_data_5 = data_filter(train_df, 20000)
        #  去掉附件3公里数据
        train_data_3km = data_filter_3km(train_data_5, service_bus_id)
        # 还原到数据点
        train_data_ori = data_recovery(train_data_3km, 35000)
        ##### 聚类
        cluster_data = train_data_ori
        #  超参数，聚类大于10，小于30时，则可停止探索
        lower_boundary = 10
        upper_boundary = 30
        #  最大的min_n
        cluster_data_len = len(cluster_data)
        if cluster_data_len < 50:
            #  to do 记录异常值
            exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
            exception_file.loc[0, "is_center"] = 2
            exception_file.loc[0, "service_bus_code"] = service_bus_code
            exception_file.loc[0, "is_visit"] = 1
            exception_file.loc[0, "is_workday"] = is_workday
            exception_file.loc[0, "inout"] = inout
            exception_file.loc[0, "addr_type"] = addr_type
            save_file = save_file.append(exception_file, ignore_index=True)
            #continue #跳出本次循环，执行下一次
            return save_file  #直接返回结果。跳出本次循环，执行下一次
        else:
            if cluster_data_len > 400:
                upper_n = 400
            else:
                upper_n = int(cluster_data_len / 50) * 50
            #  处理没有返回的异常情况。判断满足限制条件返回，还是循环结束返回
            optics_return = OPTICS_core(cluster_data, lower_boundary, upper_boundary, 1, 10, 1, 50, upper_n, 50)
            if optics_return != None:
                class_data, result_data, class_number = optics_return
            else:
                #  to do 记录异常值
                exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
                exception_file.loc[0, "is_center"] = 2
                exception_file.loc[0, "service_bus_code"] = service_bus_code
                exception_file.loc[0, "is_visit"] = 1
                exception_file.loc[0, "is_workday"] = is_workday
                exception_file.loc[0, "inout"] = inout
                exception_file.loc[0, "addr_type"] = addr_type
                save_file = save_file.append(exception_file, ignore_index=True)
                #continue #跳出本次循环，执行下一次
                return save_file  #直接返回结果。跳出本次循环，执行下一次

            # 筛选前十, 其中包含画像数据
            result_data_top_10_unique = cluster_top_10(result_data, class_number)

            ####  存储格式
            ###  和表结构对应
            result_data_top_10_unique_to_file = result_data_top_10_unique[["geohash7_bd", "service_bus_code", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "class", "dt"]]
            result_data_top_10_unique_to_file.loc[:, "is_center"] = 0
            result_data_top_10_unique_to_file.loc[:, "is_visit"] = 1
            #result_data_top_10_unique_to_file.loc[:, "dt"] = '2020-07-23'
            result_data_top_10_unique_to_file = result_data_top_10_unique_to_file[feature_list_to_file]
            save_file = save_file.append(result_data_top_10_unique_to_file, ignore_index=True)

            #######  平均画像
            ##### 计算top-10类画像TF-IDF均值，并降维
            # 计算TF_IDF矩阵
            portrayal_data_tfidf, feature_name_list = TF_IDF_transform(result_data_top_10_unique)
            # 计算TFIDF均值
            # 人数作为权重
            for i in range(0, len(portrayal_data_tfidf)):
                for j in range(0, len(feature_name_list)):
                    portrayal_data_tfidf.loc[i, feature_name_list[j]] = portrayal_data_tfidf.loc[i, feature_name_list[j]] * portrayal_data_tfidf.loc[i, "indx_avg_pin"]
            #  求平均
            ### 筛选前20维度
            pin_sum = portrayal_data_tfidf["indx_avg_pin"].sum()
            tfidf_average = pd.DataFrame(columns=["feature_name", "tfidf_average"])
            for i in range(0, len(feature_name_list)):
                tfidf_average.loc[i, "feature_name"] = feature_name_list[i]
                tfidf_average.loc[i, "tfidf_average"] = portrayal_data_tfidf[feature_name_list[i]].sum() / pin_sum

            # 排序
            sorted_tfidf_average = tfidf_average.sort_values(by='tfidf_average', ascending=False)
            sorted_tfidf_average = sorted_tfidf_average.reset_index(drop=True)  #  重设索引
            # 保留前三十维度特征，之前取前20维度
            tfidf_top_20_feature_name = sorted_tfidf_average.loc[0:30, "feature_name"].values
            tfidf_average_top_20 = sorted_tfidf_average.loc[0:30, :]

            ####################################################
            #####  目标新客挖掘
            #####  计算新客TF-IDF
            #####  从数据库读数据
            input_file_path = input_dir+'/input_novisit.txt'  #  数据库文件存储
            read_data_sql = """hive -e "set hive.cli.print.header=true;
            select
            *
            from dmu_uc.dmuuc_sl_notvisist_thermal_portrayal_distribution_i_d where service_bus_code='{service_bus_code}' and addr_type='{addr_type}' and is_workday='{is_workday}' and dt='{dt}'
            " > {path}
            """.format(service_bus_code = service_bus_code, addr_type = addr_type, is_workday = is_workday, dt = dt, path=input_file_path)
            res_sql = os.system(read_data_sql)

            if res_sql == 0:
                print("Read data done!")
            else:
                raise ValueError("输入数据读取失败!")

            print("Read data done!")
            ##### 数据处理
            #train_df_novisit = pd.read_csv('data/inputs_customer/input_novisit.txt', sep='\t')
            train_df_novisit = pd.read_csv(input_file_path, sep='\t')
            print('len(train_df_novisit)', len(train_df_novisit))

            train_data_5_novisit = data_filter(train_df_novisit, 20000)
            print(len(train_data_5_novisit))
            # 计算未到用户TFIDF
            print('TF_IDF_transform_novisit')
            portrayal_data_tfidf_novisit, feature_name_list = TF_IDF_transform_novisit(train_data_5_novisit)
            ######  计算相似度
            array_top_10_class_tfidf = np.array(tfidf_average_top_20["tfidf_average"].values)
            portrayal_data_tfidf_novisit["similarity"] = 0
            for i in range(0, len(portrayal_data_tfidf_novisit)):
                array_novisit_tfidf = np.array(portrayal_data_tfidf_novisit.loc[i, tfidf_top_20_feature_name].values)
                current_similarity = cos_sim(array_top_10_class_tfidf, array_novisit_tfidf)
                portrayal_data_tfidf_novisit.loc[i, "similarity"] = current_similarity
            print('计算相似度')
            ####  保留相似度高的前20%数据
            # 去除详细画像数据，只保留相似度即可
            useful_data = train_data_5_novisit
            #print('useful_data', useful_data)
            similarity_data_cols = useful_data.columns.values.tolist()
            #print('similarity_data_cols', similarity_data_cols)
            #similarity_data_cols = similarity_data_cols.append("similarity")
            similarity_data_cols.append("similarity") #  list.append()是没有返回值的
            #print('similarity_data_cols_2', similarity_data_cols)
            similarity_data = portrayal_data_tfidf_novisit[similarity_data_cols]
            #print('similarity_data', similarity_data)
            #  筛选出相似度TOP 20%
            save_len = int(0.2 * len(similarity_data))
            print('save_len', save_len)
            # 排序
            sorted_similarity_data = similarity_data.sort_values(by='similarity', ascending=False)
            sorted_similarity_data = sorted_similarity_data.reset_index(drop=True)  #  重设索引
            # 保留前20%
            similarity_data_top_20_percent = sorted_similarity_data.loc[0:save_len, :]


            ####  还原到数据点再聚类
            # 相似度最大值是最小值的1.003倍，基本可以认为是相同的，聚类时可以忽略相似度这个维度，不用三维聚类
            ##聚类
            # 还原到数据点
            train_data_ori_novisit = data_recovery(similarity_data_top_20_percent, 35000)
            ##### 聚类
            cluster_data_novisit = train_data_ori_novisit
            #  超参数，聚类大于10，小于30时，则可停止探索
            lower_boundary = 10
            upper_boundary = 30
            #  最大的min_n
            cluster_data_novisit_len = len(cluster_data_novisit)
            if cluster_data_novisit_len < 50:
                #  to do 记录异常值
                exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
                exception_file.loc[0, "is_center"] = 2
                exception_file.loc[0, "service_bus_code"] = service_bus_code
                exception_file.loc[0, "is_visit"] = 0
                exception_file.loc[0, "is_workday"] = is_workday
                exception_file.loc[0, "inout"] = inout # 新客不区分inout，将inout置1，避免NAN
                exception_file.loc[0, "addr_type"] = addr_type
                save_file = save_file.append(exception_file, ignore_index=True)
                #continue #跳出本次循环，执行下一次
                return save_file  #直接返回结果。跳出本次循环，执行下一次
            else:
                if cluster_data_novisit_len > 400:
                    upper_n = 400
                else:
                    upper_n = int(cluster_data_novisit_len / 50) * 50
                #  处理没有返回的异常情况。判断满足限制条件返回，还是循环结束返回
                optics_return = OPTICS_core(cluster_data_novisit, lower_boundary, upper_boundary, 3, 10, 1, 50, upper_n, 50)
                if optics_return != None:
                    class_data, result_data, class_number = optics_return
                else:
                    #  to do 记录异常值
                    exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
                    exception_file.loc[0, "is_center"] = 2
                    exception_file.loc[0, "service_bus_code"] = service_bus_code
                    exception_file.loc[0, "is_visit"] = 0
                    exception_file.loc[0, "is_workday"] = is_workday
                    exception_file.loc[0, "inout"] = inout # 新客不区分inout，将inout置1，避免NAN
                    exception_file.loc[0, "addr_type"] = addr_type
                    save_file = save_file.append(exception_file, ignore_index=True)
                    #continue #跳出本次循环，执行下一次
                    return save_file  #直接返回结果。跳出本次循环，执行下一次

                # 筛选前十, 其中包含画像数据
                result_data_top_10_unique_novisit = cluster_top_10(result_data, class_number)

                ###  和表结构对应
                novisit_result_data_top_10_unique_to_file = result_data_top_10_unique_novisit[["geohash7_bd", "service_bus_code", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "class", "dt"]]
                novisit_result_data_top_10_unique_to_file.loc[:, "is_center"] = 0
                novisit_result_data_top_10_unique_to_file.loc[:, "is_visit"] = 0
                novisit_result_data_top_10_unique_to_file.loc[:, "inout"] = inout
                #novisit_result_data_top_10_unique_to_file.loc[:, "dt"] = '2020-07-23'
                novisit_result_data_top_10_unique_to_file = novisit_result_data_top_10_unique_to_file[feature_list_to_file]

                ##### append
                save_file = save_file.append(novisit_result_data_top_10_unique_to_file, ignore_index=True)


                ###############################################################################
                #####  商圈内客流只关注到访。现有客流和潜在客流关注居住和工作
                #  商圈内客流
                if addr_id == 1: # 外部循环只有1,2
                    addr_type="到访"
                    inout = '0' # 商圈内

                    # 读取数据 【需要修改：用户自定义sql部分，只需要修改select至""""之间的部分】
                    #####  从数据库读数据
                    input_file_path = input_dir+'/input_in.txt'  #  数据库文件存储
                    read_data_sql = """hive -e "set hive.cli.print.header=true;
                    select
                    *
                    from dmu_uc.dmuuc_sl_visit_thermal_portrayal_distribution_i_d where service_bus_code='{service_bus_code}' and inout='{inout}' and addr_type='{addr_type}' and is_workday='{is_workday}' and dt='{dt}'
                    " > {path}
                    """.format(service_bus_code = service_bus_code, inout = inout, addr_type = addr_type, is_workday = is_workday, dt = dt, path=input_file_path)
                    res_sql = os.system(read_data_sql)

                    if res_sql == 0:
                        print("Read data done!")
                    else:
                        raise ValueError("输入数据读取失败!")

                    print("Read data done!")

                    ###############################################################################
                    ##### 数据处理
                    train_df_in = pd.read_csv(input_file_path, sep='\t')
                    print('train_df_in_len', len(train_df_in))
                    #  筛选5环数据
                    train_data_5_in = data_filter(train_df_in, 20000)
                    # 还原到数据点
                    train_data_ori_in = data_recovery_in(train_data_5_in, 35000)
                    ##### 聚类
                    cluster_data_in = train_data_ori_in
                    #  超参数，聚类大于10，小于30时，则可停止探索
                    lower_boundary = 3
                    upper_boundary = 6
                    cluster_data_in_len = len(cluster_data_in)
                    if cluster_data_in_len < 20:
                        #  to do 记录异常值
                        exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
                        exception_file.loc[0, "is_center"] = 2
                        exception_file.loc[0, "service_bus_code"] = service_bus_code
                        exception_file.loc[0, "is_visit"] = 1
                        exception_file.loc[0, "is_workday"] = is_workday
                        exception_file.loc[0, "inout"] = inout
                        exception_file.loc[0, "addr_type"] = addr_type
                        save_file = save_file.append(exception_file, ignore_index=True)
                        #continue #跳出本次循环，执行下一次
                        return save_file  #直接返回结果。跳出本次循环，执行下一次
                    else:
                        if cluster_data_in_len > 150:
                            upper_n = 80
                        else:
                            upper_n = int(cluster_data_in_len / 20) * 10
                        #  处理没有返回的异常情况。判断满足限制条件返回，还是循环结束返回
                        optics_return = OPTICS_core(cluster_data_in, lower_boundary, upper_boundary, 1, 10, 1, 5, upper_n, 10)
                        if optics_return != None:
                            class_data, result_data_in, class_number = optics_return
                        else:
                            upper_boundary = 12 # 放宽限制
                            optics_return = OPTICS_core(cluster_data_in, lower_boundary, upper_boundary, 1, 10, 1, 5, upper_n, 10)
                            if optics_return != None:
                                class_data, result_data_in, class_number = optics_return
                            else:
                                #  to do 记录异常值
                                exception_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
                                exception_file.loc[0, "is_center"] = 2
                                exception_file.loc[0, "service_bus_code"] = service_bus_code
                                exception_file.loc[0, "is_visit"] = 1
                                exception_file.loc[0, "is_workday"] = is_workday
                                exception_file.loc[0, "inout"] = inout
                                exception_file.loc[0, "addr_type"] = addr_type
                                save_file = save_file.append(exception_file, ignore_index=True)
                                #continue #跳出本次循环，执行下一次
                                return save_file  #直接返回结果。跳出本次循环，执行下一次

                        # 筛选前十, 其中包含画像数据
                        result_data_top_10_unique_in = cluster_top_10(result_data_in, class_number)

                        ####  存储格式
                        ###  和表结构对应
                        result_data_top_10_unique_to_file_in = result_data_top_10_unique_in[["geohash7_bd", "service_bus_code", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "class", "dt"]]
                        result_data_top_10_unique_to_file_in.loc[:, "is_center"] = 0
                        result_data_top_10_unique_to_file_in.loc[:, "is_visit"] = 1
                        #result_data_top_10_unique_to_file_in.loc[:, "dt"] = '2020-07-23'
                        result_data_top_10_unique_to_file_in = result_data_top_10_unique_to_file_in[feature_list_to_file]
                        save_file = save_file.append(result_data_top_10_unique_to_file_in, ignore_index=True)
                
        return save_file

    except:
        print('error', service_bus_id, addr_id, is_workday_id)
        index_id = str(service_bus_id) + '_' + str(addr_id) + '_' + str(is_workday_id)
        # 文件只写一次，否则会覆盖掉
        #feature_list_to_file = ["geohash7_bd", "service_bus_code", "is_visit", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "is_center", "class", "dt"]
        save_file = pd.DataFrame(columns=feature_list_to_file, index=[0])
        #save_file.loc[0, "is_center"] = index_id
        save_file.loc[0, "is_center"] = 2
        return save_file

    return save_file


###############################################################################
from multiprocessing import Pool
import time

#  循环

#返回值只有进程池才有,父子进程没有返回值
if __name__ == '__main__':

    p = Pool(44)
    #从异步提交任务获取结果
    #  所有结果汇总
    # 文件只写一次，否则会覆盖掉
    feature_list_to_file = ["geohash7_bd", "service_bus_code", "is_visit", "is_workday", "inout", "addr_type", "indx_avg_pin", "lat_bd", "lon_bd", "is_center", "class", "dt"]
    save_file_all = pd.DataFrame(columns=feature_list_to_file)
    #  返回值
    res_l = []

    dt = '2020-08-03'
    business_district_len = 33
    for i in range(0, business_district_len):
        #for i in range(0, 1):
        #for j in range(1, 3): # addr_type
        for j in range(1, 2): # addr_type
            for k in range(0, 2):
                #for k in range(0, 1):
                #all_business_district(i, j, k)
                res = p.apply_async(all_business_district, args=(i, j, k, dt, feature_list_to_file,))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
                res_l.append(res)
    #  结果
    for res in res_l:
        print(res.get())
        save_file_one = res.get()
        save_file_all = save_file_all.append(save_file_one, ignore_index=True)

    print(save_file_all)
    ###############################################################################
    ###  保存数据
    # 创建结果文件夹【无需修改,输出数据在data/outputs文件夹下】
    output_dir = 'outputs'
    output_dir = 'data/' + output_dir  # 拼上data文件夹,对应离线模型部署模型结果文件路径
    print("Create output directory begin")
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)
    else:
        cmd = 'rm -rf ' + output_dir
        os.system(cmd)
        os.makedirs(output_dir)
    print("Create output directory end!")

    # write result into a file with the same name
    #file_name = file.split('/')[-1]
    file_name = 'dmuuc_sl_top_10_class_i_d.txt'
    print(file_name)
    file_path = os.path.join(output_dir, file_name)
    #novisit_result_data_top_10_unique_to_file.to_csv(file_path, sep = '\t', header = False, index = False)
    save_file_all.to_csv(file_path, sep = '\t', header = False, index = False)


    ###############################################################################
    #######  代码文件格式转换，KuAI仅支持.py部署
    try:
        !jupyter nbconvert --to python multiprocessing_all_5km_0730.ipynb
        # python即转化为.py，script即转化为.html
        # file_name.ipynb即当前module的文件名
    except:
        pass