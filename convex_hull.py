###############################################################################
## 部署必备
import os    # 必需的包，勿删除
import sys   # 必需的包，勿删除
import glob  # 必需的包，勿删除

## 自定义依赖包
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from geojson import Polygon, Feature, FeatureCollection

###############################################################################

#  数据库读取数据
def get_cluster_data(service_bus_code, addr_type, is_workday, is_visit, inout, dt):

    ####  读取数据
    ## 创建输入数据文件夹
    input_dir = 'inputs_convex_hull'
    input_dir = 'data/' + input_dir  # 拼上data文件夹，作为子目录
    print("Create directory begin:")
    if not os.path.exists(input_dir):  # 判断是否存在文件夹如果不存在则创建文件夹
        os.makedirs(input_dir)
    else:
        cmd = 'rm -rf ' + input_dir
        os.system(cmd)
        os.makedirs(input_dir)
    print("Create directory end!")
    ## SQL读取数据到文件
    input_file_path = input_dir+'/input.txt'  #  数据库文件存储
    ##########  银河库，应改为mysql库
    if is_visit == 0:
        #  潜在客流，数据库中inout为NAN，语句中不要出现inout
        read_data_sql = """hive -e "set hive.cli.print.header=true;
        select geohash7_bd, lat_bd, lon_bd, indx_avg_pin, class
        from dmu_uc.dmuuc_sl_top_10_class_i_d where service_bus_code='{service_bus_code}' and addr_type='{addr_type}' and is_workday='{is_workday}' and is_visit = '{is_visit}' and dt='{dt}'
        " > {path}
        """.format(service_bus_code = service_bus_code, addr_type = addr_type, is_workday = is_workday, is_visit = is_visit, dt = dt, path=input_file_path)
        res_sql = os.system(read_data_sql)
    else:
        read_data_sql = """hive -e "set hive.cli.print.header=true;
        select geohash7_bd, lat_bd, lon_bd, indx_avg_pin, class
        from dmu_uc.dmuuc_sl_top_10_class_i_d where service_bus_code='{service_bus_code}' and inout='{inout}' and addr_type='{addr_type}' and is_workday='{is_workday}' and is_visit = '{is_visit}' and dt='{dt}'
        " > {path}
        """.format(service_bus_code = service_bus_code, addr_type = addr_type, is_workday = is_workday, is_visit = is_visit, inout = inout, dt = dt, path=input_file_path)
        res_sql = os.system(read_data_sql)

    if res_sql == 0:
        print("Read data done!")
    else:
        raise ValueError("输入数据读取失败!")

    ##########  mysql库相应sql语句表明变量名做对应修改
    '''
    if is_visit == 0:
        SELECT geohash7_bd, lat_bd, lon_bd, indx_avg_pin, class FROM btgdb.sl_top_10_class where sl_top_10_class.service_bus_code = '100001' and sl_top_10_class.addr_type = '居住' and sl_top_10_class.is_workday = 'weekday' and sl_top_10_class.is_visit = '0' and sl_top_10_class.day = '2020-07-20'
    else:
        SELECT geohash7_bd, lat_bd, lon_bd, indx_avg_pin, class FROM btgdb.sl_top_10_class where sl_top_10_class.service_bus_code = '100001' and sl_top_10_class.addr_type = '居住' and sl_top_10_class.is_workday = 'weekday' and sl_top_10_class.is_visit = '1' and sl_top_10_class.inout = '1' and sl_top_10_class.day = '2020-07-20'
    '''
    ## 从文件中读取数据
    cluster_data = pd.read_csv(input_file_path, sep='\t')
    print('cluster_data', cluster_data)

    return cluster_data

## 获取重心坐标
def get_center_point(lis):
    area = 0.0
    x, y = 0.0, 0.0
 
    a = len(lis)
    for i in range(a):
        lat = lis[i][0] #weidu
        lng = lis[i][1] #jingdu
 
        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]
 
        else:
            lat1 = lis[i-1][0]
            lng1 = lis[i-1][1]
 
        fg = (lat*lng1 - lng*lat1)/2.0
 
        area += fg
        x += fg*(lat+lat1)/3.0
        y += fg*(lng+lng1)/3.0
 
    x = x/area
    y = y/area
 
    return x, y

####  商圈内数据太少，一条直线无法构成平面，增加周边东南西北数据
def data_around(one_cluster_data):

    #  50米，经度约0.0005，纬度0.0005
    one_cluster_data_around = pd.DataFrame(columns=one_cluster_data.columns.values.tolist())
    for i in range(0, len(one_cluster_data["indx_avg_pin"])):
        current = one_cluster_data.iloc[i, :]
        current_east = current.copy()
        current_east.loc["lon_bd"] = current_east.loc["lon_bd"] + 0.0005
        one_cluster_data_around = one_cluster_data_around.append(current_east, ignore_index=True)
        current_south = current.copy()
        current_south.loc["lat_bd"] = current_south.loc["lat_bd"] - 0.0005
        one_cluster_data_around = one_cluster_data_around.append(current_south, ignore_index=True)
        current_west = current.copy()
        current_west.loc["lon_bd"] = current_west.loc["lon_bd"] - 0.0005
        one_cluster_data_around = one_cluster_data_around.append(current_west, ignore_index=True)
        current_north = current.copy()
        current_north.loc["lat_bd"] = current_north.loc["lat_bd"] + 0.0005
        one_cluster_data_around = one_cluster_data_around.append(current_north, ignore_index=True)
    
    return one_cluster_data_around

def get_convex_hull(cluster_data, is_visit, inout):
    
    #### 数据处理
    feature_list = []
    if len(cluster_data) > 0:
        class_number = len(np.unique(cluster_data["class"]))
        for i in range(0, class_number):
            one_cluster_data = cluster_data[(cluster_data["class"] == str(i))]
            one_cluster_data = one_cluster_data.reset_index(drop=True) # 重设索引
            one_cluster_data['indx_avg_pin'] = pd.to_numeric(one_cluster_data["indx_avg_pin"])
            one_cluster_data['lat_bd'] = pd.to_numeric(one_cluster_data["lat_bd"])
            one_cluster_data['lon_bd'] = pd.to_numeric(one_cluster_data["lon_bd"])

            if len(one_cluster_data["indx_avg_pin"]) > 0:

                #  商圈内数据太少，一条直线无法构成平面，增加周边东南西北数据
                #if is_visit == '1' and inout == '0':
                if len(one_cluster_data["indx_avg_pin"]) < 10:
                    one_cluster_data = data_around(one_cluster_data)
                 ##  计算人数密度
                density = one_cluster_data["indx_avg_pin"].sum() / len(one_cluster_data["indx_avg_pin"])
                ##  计算凸包
                convex_hull = ConvexHull(one_cluster_data[["lat_bd", "lon_bd"]]) # 计算凸包
                convex_hull_index = list(convex_hull.vertices)
                convex_hull_index.append(convex_hull_index[0])  # 多边形要闭合必须再回到起点，list append没有返回值，不要赋值
                convex_hull_data = one_cluster_data.loc[convex_hull_index, :] # 多边形对应数据
                convex_hull_data = convex_hull_data.reset_index(drop=True)  #  重设索引
                ##  计算重心
                lis = convex_hull_data[["lat_bd", "lon_bd"]].values.tolist()
                lat_center, lon_center = get_center_point(lis)
                ##   转成geojson
                lon_lat = [lon_lat for lon_lat in zip(convex_hull_data.lon_bd, convex_hull_data.lat_bd)]
                lon_lat_new = [lon_lat]
                polygon_current = Polygon(lon_lat_new)
                feature_current = Feature(geometry=polygon_current)
                properties = dict(cluster_class = i, rank = i, density = density, lat_center = lat_center, lon_center = lon_center)
                feature_current.properties = properties
                feature_list.append(feature_current)
    
    else:
        print('no data')

    #  排序
    density = np.zeros((class_number), dtype=np.float)
    for i in range(0, class_number):
        density[i] = feature_list[i].properties["density"]
    rank_list = np.argsort(-density) # 逆序输出索引，从大到小
    print(rank_list)
    for i in range(0, class_number):
        class_id = rank_list[i]
        feature_list[class_id].properties["rank"] = i
    # feature_collection
    feature_collection = FeatureCollection(feature_list)

    return feature_collection

if __name__ == '__main__':
    
    #### 设置参数，初始值
    service_bus_code = '100001'
    addr_type="居住"
    is_workday ="weekday"
    is_visit = '1'
    inout = '1' # 商圈外
    dt = ''

    #### 获取聚类区域数据
    cluster_data = get_cluster_data(service_bus_code, addr_type, is_workday, is_visit, inout, dt)
    
    #### 获取凸包、重心、密度，并转换成geojson
    feature_collection = get_convex_hull(cluster_data, is_visit, inout)
    print(feature_collection)

    
