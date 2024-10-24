
import pandas as pd
import geopandas as gpd
import ast
import transbigdata as tbd  # 假设tbd库有必要的GPS_to_grid和area_to_grid函数
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

def site_genetic(station_info_path, taz_path, car_infos_path, poi_path, price_path, gridfile_path, gridgejson_path,max_sites):
# 定义处理充电站数据的函数
    def process_charging_orders(station_info, step_length, params):
        station_info_table = station_info[
            ['station_id', 'lon', 'lat', 'max_capacity', 'charge_speed_station']].drop_duplicates().copy()

        station_info = station_info[station_info['num_current_car'] > 0]
        station_info['time'] = pd.to_datetime(station_info['time'])
        station_info['current_car'] = station_info['current_car'].apply(lambda a: ast.literal_eval(a))
        station_info['waiting_car'] = station_info['waiting_car'].apply(lambda a: ast.literal_eval(a))
        station_info.sort_values(by=['station_id', 'time'], inplace=True)

        current_car_infos = station_info[['station_id', 'time', 'current_car']].explode('current_car')
        current_car_infos = current_car_infos[~current_car_infos['current_car'].isnull()]
        current_car_infos = current_car_infos.sort_values(by=['current_car', 'time'])[['current_car', 'time', 'station_id']]

        waiting_car_infos = station_info[['station_id', 'time', 'waiting_car']].explode('waiting_car')
        waiting_car_infos = waiting_car_infos[~waiting_car_infos['waiting_car'].isnull()]
        waiting_car_infos = waiting_car_infos.sort_values(by=['waiting_car', 'time'])[['waiting_car', 'time', 'station_id']]

        return station_info_table, current_car_infos, waiting_car_infos


    # 计算充电订单
    def get_charging_order(station_info_table, current_car_infos, step_length):
        current_car_infos['timegap'] = current_car_infos['time'].diff().dt.total_seconds().fillna(1000000).astype(int)
        current_car_infos['order_id'] = (current_car_infos['timegap'] > step_length).cumsum()
        charge_info_s = current_car_infos.groupby(['current_car', 'order_id']).first().reset_index()
        charge_info_e = current_car_infos.groupby(['current_car', 'order_id']).last().reset_index()
        charging_order = pd.merge(charge_info_s, charge_info_e, on=['current_car', 'order_id', 'station_id'])
        charging_order = charging_order[['current_car', 'order_id', 'time_x', 'time_y', 'station_id']]
        charging_order.columns = ['carid', 'order_id', 'stime', 'etime', 'station_id']
        charging_order['duration'] = (charging_order['etime'] - charging_order['stime']).dt.total_seconds()
        charging_order = charging_order[charging_order['duration'] > 0]
        charging_orders = pd.merge(charging_order, station_info_table)
        # 计算充电时长
        charging_orders['duration'] = (charging_orders['etime'] - charging_orders['stime']).dt.total_seconds()
        return charging_orders


    def get_order(charging_orders, params, grid):
        charging_order = charging_orders
        charging_order['LONCOL'], charging_order['LATCOL'] = tbd.GPS_to_grid(charging_order['lon'], charging_order['lat'],
                                                                             params=params)
        charging_order_agg = charging_order.groupby(['LONCOL', 'LATCOL'])['carid'].count().reset_index()
        charging_order_agg = pd.merge(grid[['LONCOL', 'LATCOL']], charging_order_agg, on=['LONCOL', 'LATCOL'], how="left")[
            ['LONCOL', 'LATCOL', 'carid']]

        charging_order_agg.fillna(0, inplace=True)
        return charging_order_agg


    # 处理潜在充电需求
    def process_potential_demand(car_infos, step_length, params):
        car_infos['time'] = pd.to_datetime(car_infos['time'])
        stay_infos = car_infos.sort_values(by=['carid', 'time'])
        stay_infos['timegap'] = (-stay_infos['time'].diff(-1).dt.total_seconds()).fillna(1000000).astype(int)
        stay_infos['etime'] = stay_infos['time'].shift(-1)
        stay_order = stay_infos[stay_infos['timegap'] > step_length][['carid', 'time', 'etime', 'soc', 'lon', 'lat']]
        stay_order.columns = ['carid', 'stime', 'etime', 'soc', 'lon', 'lat']
        stay_order['duration'] = (stay_order['etime'] - stay_order['stime']).dt.total_seconds()
        stay_order = stay_order[(stay_order["soc"] <= 50) & (stay_order["duration"] >= 60 * 5)]
        stay_order['LONCOL'], stay_order['LATCOL'] = tbd.GPS_to_grid(stay_order['lon'], stay_order['lat'], params=params)
        stay_order_agg = stay_order.groupby(['LONCOL', 'LATCOL']).size().reset_index()
        stay_order_agg.columns = ['LONCOL', 'LATCOL', "pdemand"]
        return stay_order_agg


    # 处理 POI 数据
    def process_poi(poi_data, grid, params, poi_types=['停车场', '加油站']):
        poi_data = poi_data[poi_data["pname"] == "上海市"]
        poi_data['ttype'] = poi_data['type'].apply(lambda x: x.split(';')[0])
        poi_data["lon"] = poi_data['location'].apply(lambda x: x.split(',')[0])
        poi_data["lat"] = poi_data['location'].apply(lambda x: x.split(',')[1])
        poi_data = poi_data[poi_data["type"].str.contains('|'.join(poi_types))]
        poi_data["lon"] = poi_data["lon"].astype("float")
        poi_data["lat"] = poi_data["lat"].astype("float")
        poi_data['LONCOL'], poi_data['LATCOL'] = tbd.GPS_to_grid(poi_data['lon'], poi_data['lat'], params=params)
        poi_agg = poi_data.groupby(['LONCOL', 'LATCOL']).size().reset_index()
        poi_agg = pd.merge(poi_agg, grid[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how="right")
        poi_agg.columns = ['LONCOL', 'LATCOL', "park"]
        poi_agg.fillna(0, inplace=True)
        return poi_agg


    # 处理充电站的利用率
    def calculate_utilization(charging_orders, grid, params):
        """
        计算充电站的充电需求满足度（利用率）。

        参数:
        - charging_orders: 包含充电订单的DataFrame，至少应包含 ['stime', 'station_id', 'lon', 'lat', 'max_capacity', 'duration']
        - grid: 栅格数据，用于进行GPS坐标转换和聚合。
        - params: 栅格化参数。

        返回:
        - station_chargetime_agg: DataFrame, 包含聚合后的利用率数据，按栅格坐标（LONCOL, LATCOL）。
        """
        # 确定充电订单中的最早和最晚时间
        start_time = charging_orders["stime"].min()
        end_time = charging_orders["stime"].max()
        duration = end_time - start_time

        # 计算每个充电站的充电时长（duration）
        station_chargetime = charging_orders.groupby(["station_id", "lon", "lat", "max_capacity"])[
            "duration"].sum().reset_index()

        # 转换为timedelta格式
        station_chargetime["duration"] = pd.to_timedelta(station_chargetime["duration"], unit='s')

        # 计算充电站的利用率
        station_chargetime["uti"] = station_chargetime["duration"] / (duration * station_chargetime["max_capacity"])

        # 将站点的经纬度转换为栅格坐标
        station_chargetime['LONCOL'], station_chargetime['LATCOL'] = tbd.GPS_to_grid(station_chargetime['lon'],
                                                                                     station_chargetime['lat'],
                                                                                     params=params)

        # 按照栅格（LONCOL, LATCOL）聚合充电站利用率
        station_chargetime_agg = station_chargetime.groupby(['LONCOL', 'LATCOL']).mean().reset_index()

        # 将聚合后的利用率数据与栅格进行合并
        station_chargetime_agg = \
        pd.merge(station_chargetime_agg, grid[['LONCOL', 'LATCOL']], on=['LONCOL', 'LATCOL'], how="right")[
            ['LONCOL', 'LATCOL', 'uti']]

        # 处理空值情况
        station_chargetime_agg.fillna(0, inplace=True)

        return station_chargetime_agg


    # 计算建站成本
    def process_station_cost(price_data, grid, params):
        price_data["geometry"] = gpd.points_from_xy(price_data["lon"], price_data["lat"])
        price_data = gpd.GeoDataFrame(price_data, geometry=price_data["geometry"])
        price_data.crs = "EPSG:4326"
        price_data = price_data.to_crs("EPSG:32651")
        buffer = price_data.buffer(1000)
        price_data = gpd.GeoDataFrame(price_data, geometry=buffer)
        pricegrid = grid.to_crs("EPSG:32651")
        pricegrid = gpd.sjoin(pricegrid, price_data)
        pricegrid = pricegrid.groupby(["LONCOL", "LATCOL"])["price"].mean().reset_index()

        pricegrid["price"] = pricegrid["price"] * 100 * 0.02 * 0.1 * 20 + 200000 + 400000 + 0.4 * (
                    pricegrid["price"] * 100 * 0.02 * 0.1 * 20 + 200000 + 400000)
        pricegrid = pd.merge(pricegrid, grid, how="right", on=['LONCOL', 'LATCOL'])
        pricegrid.fillna(pricegrid["price"].min(), inplace=True)
        pricegrid = pricegrid[['LONCOL', 'LATCOL', "price"]]
        return pricegrid


    # 整合栅格
    def merge_grid(station_info_path, taz_path, car_infos_path, poi_path, price_path, gridfile_path, gridgejson_path):
        # 加载数据
        station_info = pd.read_csv(station_info_path)
        taz = gpd.read_file(taz_path)
        car_infos = pd.read_csv(car_infos_path)
        poi = pd.read_excel(poi_path)
        price = pd.read_csv(price_path)
        gridfile = gridfile_path
        gridgejson = gridgejson_path

        # 设置栅格化参数
        paramssh = {'slon': 120.88125, 'slat': 30.7125, 'deltalon': 0.0125, 'deltalat': 0.008333, 'theta': 0,
                    'method': 'rect', 'gridsize': 1000}
        grid, paramssh = tbd.area_to_grid(taz, params=paramssh)

        # 处理订单数据
        step_length = 5 * 60
        station_info_table, current_car_infos, waiting_car_infos = process_charging_orders(station_info, step_length,
                                                                                           paramssh)
        charging_orders = get_charging_order(station_info_table, current_car_infos, step_length)
        # 处理充电需求数据
        charging_order_agg = get_order(charging_orders, paramssh, grid)
        # 处理潜在充电需求
        stay_order_agg = process_potential_demand(car_infos, step_length, paramssh)
        # 处理POI
        poi_agg = process_poi(poi, grid, paramssh)

        # 处理充电站利用率
        station_chargetime_agg = calculate_utilization(charging_orders, grid, paramssh)

        # 计算建站成本
        pricegrid = process_station_cost(price, grid, paramssh)

        gridsum = pd.merge(charging_order_agg, stay_order_agg, on=['LONCOL', 'LATCOL'])
        gridsum = pd.merge(gridsum, poi_agg, on=['LONCOL', 'LATCOL'])
        gridsum = pd.merge(gridsum, station_chargetime_agg, on=['LONCOL', 'LATCOL'])
        gridsum = pd.merge(gridsum, pricegrid, on=['LONCOL', 'LATCOL'])

        gridsum.columns = ['LONCOL', 'LATCOL', "demand", "pdemand", "park", "uti", "price"]

        gridsum.to_csv(gridfile, index=False)

        grid.to_file(gridgejson)

        # 数据汇总（如果需要后续处理，可以在这里继续处理或保存）
        return gridsum


    # 数据预处理函数
    def preprocess_data(gridsum):
        df = gridsum.copy()
        df = df[df["uti"] <= 0.5]  # 保证利用率 <= 0.5
        df = df[df["park"] >= 1]  # 保证停车位数 >= 1
        df.reset_index(drop=True, inplace=True)
        df["demand"] = df["demand"].astype(float)
        df["pdemand"] = df["pdemand"].astype(float)
        return df


    # 目标函数：只考虑充电需求和潜在充电需求
    def objective_function(individual, df):
        selected_indices = [i for i in range(len(individual)) if individual[i] == 1]
        total_score = 0
        for i in selected_indices:
            D_it = df.at[i, 'demand']
            P_it = df.at[i, 'pdemand']
            total_score += (D_it + P_it)
        return total_score


    # 约束条件检查函数
    def satisfies_constraints(individual, df, max_cost, max_sites):
        selected_indices = [i for i in range(len(individual)) if individual[i] == 1]
        if not selected_indices or len(selected_indices) != max_sites:
            return False  # 如果没有选中的栅格或选中的栅格数不等于max_sites，则返回不满足约束

        total_cost = df.iloc[selected_indices]['price'].sum()

        # 条件1：用地约束
        land_availability = all(df.iloc[selected_indices]['park'] > 0)

        # 条件2：充电需求满足度约束
        demand_satisfaction = all(df.iloc[selected_indices]['uti'] <= 0.5)

        # 条件3：建站成本约束
        cost_constraint = total_cost <= max_cost

        return land_availability and demand_satisfaction and cost_constraint


    # 评价函数
    def evaluate(individual, df, max_cost, max_sites):
        if satisfies_constraints(individual, df, max_cost, max_sites):
            return objective_function(individual, df),
        else:
            return 0.0,  # 不满足约束条件的个体适应度设为0


    # 初始化个体的函数
    def init_individual(icls, df, size, num_ones, target_cost):
        individual = [0] * size
        df_sorted = df.copy()
        df_sorted['demand_pdemand_sum'] = df['demand'] + df['pdemand']

        df_high_cost = df_sorted[df_sorted['price'] > target_cost].sort_values(by=['demand_pdemand_sum', 'price'],
                                                                               ascending=[False, True])
        df_low_cost = df_sorted[df_sorted['price'] <= target_cost].sort_values(by=['demand_pdemand_sum', 'price'],
                                                                               ascending=[False, False])

        selected_indices = []

        high_cost_count = min(len(df_high_cost), num_ones // 2)
        selected_indices.extend(df_high_cost.index[:high_cost_count])

        remaining_count = num_ones - len(selected_indices)
        selected_indices.extend(df_low_cost.index[:remaining_count])

        for idx in selected_indices:
            individual[idx] = 1

        return icls(individual)


    # 自定义变异函数，确保变异后仍然有固定数量的1
    def mut_shuffle_indexes(individual, indpb):
        if np.random.random() < indpb:
            ones_indices = [i for i, bit in enumerate(individual) if bit == 1]
            zeros_indices = [i for i, bit in enumerate(individual) if bit == 0]
            if ones_indices and zeros_indices:
                swap_out = np.random.choice(ones_indices)
                swap_in = np.random.choice(zeros_indices)
                individual[swap_out], individual[swap_in] = individual[swap_in], individual[swap_out]
        return individual,


    # 初始化遗传算法工具
    def init_toolbox(df, cost, max_sites):
        # 注册工具
        toolbox = base.Toolbox()
        toolbox.register("individual", init_individual, creator.Individual, df=df, size=len(df), num_ones=max_sites,
                         target_cost=cost)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", mut_shuffle_indexes, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate, df=df, max_cost=cost, max_sites=max_sites)
        return toolbox


    # 遗传算法主函数
    def genetic_algorithm(toolbox, population_size, generations, cxpb, mutpb):
        pop = toolbox.population(n=population_size)
        algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, verbose=True)

        best_individual = tools.selBest(pop, k=1)[0]
        selected_indices = [i for i in range(len(best_individual)) if best_individual[i] == 1]
        return selected_indices


    # 主调用函数
    def run_site_selection(gridsum, cost=120 * 10000, population_size=900, generations=200, cxpb=0.5, mutpb=0.2,
                           max_sites=100):
        # 数据预处理

        df = preprocess_data(gridsum)

        # 初始化工具
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = init_toolbox(df, cost, max_sites)

        # 执行遗传算法
        optimal_sites = genetic_algorithm(toolbox, population_size, generations, cxpb, mutpb)

        # 筛选出最优站点数据
        optimal_grid_data = df.iloc[optimal_sites]
        return optimal_grid_data


    gridsum = merge_grid(station_info_path, taz_path, car_infos_path, poi_path, price_path, gridfile_path, gridgejson_path)
    result = run_site_selection(gridsum, max_sites=max_sites)
    grid=gpd.read_file(gridgejson_path)
    result=pd.merge(grid,result,on=["LONCOL","LATCOL"])
    result.columns=["LONCOL","LATCOL","geometry","demand","potential demand","park","utilization rate","cost"]
    return result

import pandas as pd
import geopandas as gpd
import transbigdata as tbd
def area_classfication(filename, select_city, bounds,bins):
    # 读取数据文件，选择城市
    def read_data(filename, select_city):
        """
        输入:
            filename (str): 文件路径
            select_city (str): 选择的城市名称
        输出:
            pd.DataFrame: 过滤后的数据集，仅包含选定城市的数据
        """
        df = pd.read_excel(filename)
        return df[df["市"] == select_city]

    # 对每个列进行MinMax标准化，并计算得分
    def calculate_scores(df, col, quantiles):
        """
        输入:
            df (pd.DataFrame): 输入的数据集
            col (pd.Index): 需要进行归一化的列
            quantiles (dict): 包含每列的0.96分位数的字典
        输出:
            pd.DataFrame: 返回包含标准化得分的新数据集，基于传入的0.96分位数作为最大值进行归一化
        """
        for c in col:
            # 获取对应列的0.96分位数作为max值
            max_val = quantiles[c]

            # 定义归一化公式：根据0.96分位数作为最大值
            new_col = c + "_score"
            df[new_col] = (df[c] / max_val) * 100

            # 如果大于100，归一化得分上限为100
            df[new_col] = df[new_col].clip(upper=100)

        # 计算总得分，汇总所有含有'score'字样的列
        df["total_score"] = df[[c for c in df.columns if "score" in c]].sum(axis=1)

        return df

    # 提取经纬度坐标信息
    def extract_coordinates(df):
        """
        输入:
            df (pd.DataFrame): 输入的数据集，包含栅格编号
        输出:
            pd.DataFrame: 返回提取了经纬度坐标的更新数据集
        """
        df["lon"] = df["栅格编号"].str.split("-", expand=True)[1].astype("float")
        df["lat"] = df["栅格编号"].str.split("-", expand=True)[2].astype("float")
        return df

    # 将数据转换为GeoDataFrame
    def create_geodataframe(df):
        """
        输入:
            df (pd.DataFrame): 包含经纬度的普通DataFrame
        输出:
            gpd.GeoDataFrame: 包含几何信息的GeoDataFrame
        """
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]))

    # 对栅格评分，并生成格网
    def score_grid(df, bounds):
        """
        输入:
            df (pd.DataFrame): 包含经纬度及评分的DataFrame
            bounds (tuple): 定义区域边界的坐标
        输出:
            pd.DataFrame: 包含总评分的格网数据
        """
        p = tbd.area_to_params(bounds, accuracy=5000)
        grid, p = tbd.area_to_grid(bounds, params=p)
        df["LONCOL"], df["LATCOL"] = tbd.GPS_to_grid(df["lon"], df["lat"], params=p)
        score = df.groupby(["LONCOL", "LATCOL"])["total_score"].sum().reset_index()
        score.columns = ["LONCOL", "LATCOL", "total_score"]
        grid = pd.merge(grid, score, on=["LONCOL", "LATCOL"], how="left")
        return grid[grid["total_score"].notna()]

    # 根据评分将格网进行分类
    def classify_grid(grid,bins):
        """
        输入:
            grid (pd.DataFrame): 包含格网和评分的数据
        输出:
            pd.DataFrame: 排序后的格网数据，并按评分分类
        """
        grid.sort_values(by="total_score", ascending=False, inplace=True)
        grid["class"] = pd.cut(grid["total_score"], bins=bins, labels=range(1,bins+1))
        return grid

    # 选择最佳站点
    def select_best_sites(df, grid):
        """
        输入:
            df (pd.DataFrame): 原始数据集，包含坐标和评分
            grid (pd.DataFrame): 已经分类的格网数据
        输出:
            pd.DataFrame: 每个栅格中的最佳站点选择
        """
        select = pd.merge(df[["geometry", "LONCOL", "LATCOL", "total_score"]],
                          grid[["LONCOL", "LATCOL", "class"]])
        select["class"] = select["class"].astype(int)
        select = select.sort_values(by=["LONCOL", "LATCOL", "total_score"], ascending=[True, True, False])
        f = select.groupby(["LONCOL", "LATCOL"]).apply(lambda x: x.head(x["class"].iloc[0])).reset_index(drop=True)
        return f

    # 主函数，用于调用所有步骤
    def main(filename, select_city, bounds,bins):
        """
        输入:
            filename (str): 文件路径
            select_city (str): 选择的城市
            bounds (tuple): 定义区域边界的坐标
        输出:
            pd.DataFrame: 最终的站点选择结果
        """
        df_gz = read_data(filename, select_city)
        cols = df_gz.columns[4:]
        df = pd.read_excel(filename)
        quantiles = {}
        for col in cols:
            quantiles[col] = df[col].quantile(0.96)
        df_gz = calculate_scores(df_gz, cols, quantiles)
        df_gz = extract_coordinates(df_gz)
        df_gz = create_geodataframe(df_gz)
        grid = score_grid(df_gz, bounds)
        grid = classify_grid(grid,bins)
        # best_sites = select_best_sites(df_gz, grid)
        return grid

    # 调用主函数
    area = main(filename, select_city, bounds,bins)
    return area



import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import transbigdata as tbd  # 假设用来处理GPS与网格转换的模块
import ast
import numpy as np


def grid_score(area, station_info_path, car_info_path, poi_file_path, price_file, station_file_path):
    def process_station_data(station_info_path, step_length=300):
        # Load station info
        station_info = pd.read_csv(station_info_path)
        station_info = station_info[station_info['num_current_car'] > 0]

        # Extract relevant columns and clean data
        station_info_table = station_info[
            ['station_id', 'lon', 'lat', 'max_capacity', 'charge_speed_station']].drop_duplicates().copy()
        station_info = station_info[['station_id', 'time', 'current_car', 'waiting_car']]

        # Convert time and handle current/waiting car columns
        station_info['time'] = pd.to_datetime(station_info['time'])
        station_info['current_car'] = station_info['current_car'].apply(lambda a: ast.literal_eval(a))
        station_info['waiting_car'] = station_info['waiting_car'].apply(lambda a: ast.literal_eval(a))
        station_info.sort_values(by=['station_id', 'time'], inplace=True)

        # Process current car infos
        current_car_infos = station_info[['station_id', 'time', 'current_car']].explode('current_car')
        current_car_infos = current_car_infos[~current_car_infos['current_car'].isnull()]
        current_car_infos = current_car_infos.sort_values(by=['current_car', 'time'])[
            ['current_car', 'time', 'station_id']]

        # Process waiting car infos
        waiting_car_infos = station_info[['station_id', 'time', 'waiting_car']].explode('waiting_car')
        waiting_car_infos = waiting_car_infos[~waiting_car_infos['waiting_car'].isnull()]
        waiting_car_infos = waiting_car_infos.sort_values(by=['waiting_car', 'time'])[
            ['waiting_car', 'time', 'station_id']]

        # Helper function to generate charging or waiting orders
        def get_charging_order(car_infos, step_length):
            car_infos['timegap'] = car_infos['time'].diff().dt.total_seconds().fillna(1000000).astype(int)
            car_infos['order_id'] = (car_infos['timegap'] > step_length).cumsum()
            charge_info_s = car_infos.groupby(['current_car', 'order_id']).first().reset_index()
            charge_info_e = car_infos.groupby(['current_car', 'order_id']).last().reset_index()
            charging_order = pd.merge(charge_info_s, charge_info_e, on=['current_car', 'order_id', 'station_id'])
            charging_order = charging_order[['current_car', 'order_id', 'time_x', 'time_y', 'station_id']]
            charging_order.columns = ['carid', 'order_id', 'stime', 'etime', 'station_id']
            charging_order['duration'] = (charging_order['etime'] - charging_order['stime']).dt.total_seconds()
            charging_order = charging_order[charging_order['duration'] > 0]
            return charging_order

        # Generate charging and waiting orders
        charging_orders = get_charging_order(current_car_infos, step_length)
        waiting_orders = get_charging_order(waiting_car_infos.rename(columns={'waiting_car': 'current_car'}),
                                            step_length)

        # Merge station info details into orders
        charging_orders = pd.merge(charging_orders, station_info_table)
        waiting_orders = pd.merge(waiting_orders, station_info_table)

        # Save charging and waiting orders to CSV
        # charging_orders.to_csv('output/charging_orders_from_station.csv', index=False)
        # waiting_orders.to_csv('output/waiting_orders_from_station.csv', index=False)

        return charging_orders, waiting_orders

    def process_waiting_score(waiting, area, params):

        # Group waiting data by station and location, and count the number of waiting cars
        waiting = waiting.groupby(["station_id", "lon", "lat"]).size().reset_index()
        waiting.columns = ["station_id", "lon", "lat", "waiting"]

        # Convert the area to a grid using the provided function
        grid_rec, params_a = tbd.area_to_grid(area, params=params)

        # Convert the GPS coordinates to LONCOL and LATCOL using the GPS to grid function
        waiting["LONCOL"], waiting["LATCOL"] = tbd.GPS_to_grid(waiting["lon"], waiting["lat"], params)

        # Group the waiting data by LONCOL and LATCOL, summing the waiting counts
        waitinggrid = waiting.groupby(["LONCOL", "LATCOL"])["waiting"].sum().reset_index()

        # Merge the grid records with the waiting grid data
        grid_rec2 = pd.merge(grid_rec, waitinggrid, on=["LONCOL", "LATCOL"], how="left")

        # Fill NaN values in the merged data with zeros
        grid_rec2.fillna(0, inplace=True)

        # Extract the final waiting result with LONCOL, LATCOL, and waiting count
        waiting_result = grid_rec2[["LONCOL", "LATCOL", "waiting"]]

        # Normalize the waiting counts using MinMaxScaler and compute waiting score
        scaler = MinMaxScaler()
        waiting_result["w_score"] = scaler.fit_transform(waiting_result[["waiting"]])

        # Scale the waiting score to a 0-100 range
        waiting_result["w_score"] = waiting_result["w_score"] * 100

        # Return the waiting score with LONCOL, LATCOL, and w_score
        waiting_score = waiting_result[["LONCOL", "LATCOL", "w_score"]]

        return waiting_score

    def process_parking_score(parking, area, params):
        # Load the parking orders data

        # Filter parking data for vehicles with SOC <= 60 and duration >= 300 seconds
        parking = parking[parking["soc"] <= 60]
        parking = parking[parking["duration"] >= 300]

        # Group parking data by longitude and latitude, and count the number of parking events
        parking = parking.groupby(["lon", "lat"]).size().reset_index()
        parking.columns = ["lon", "lat", "parking"]

        # Convert the GPS coordinates to LONCOL and LATCOL using the GPS to grid function
        parking["LONCOL"], parking["LATCOL"] = tbd.GPS_to_grid(parking["lon"], parking["lat"], params)

        # Group the parking data by LONCOL and LATCOL, summing the parking counts
        parkinggrid = parking.groupby(["LONCOL", "LATCOL"])["parking"].sum().reset_index()

        # Convert the area to a grid using the provided function
        grid_rec, params_a = tbd.area_to_grid(area, params=params)

        # Merge the grid records with the parking grid data
        grid_rec3 = pd.merge(grid_rec, parkinggrid, on=["LONCOL", "LATCOL"], how="left")

        # Fill NaN values in the merged data with zeros
        grid_rec3.fillna(0, inplace=True)

        # Extract the final parking result with LONCOL, LATCOL, and parking count
        parking_result = grid_rec3[["LONCOL", "LATCOL", "parking"]]

        # Normalize the parking counts using MinMaxScaler and compute parking score
        scaler = MinMaxScaler()
        parking_result["p_score"] = scaler.fit_transform(parking_result[["parking"]])

        # Scale the parking score to a 0-100 range
        parking_result["p_score"] = parking_result["p_score"] * 100

        # Return the parking score with LONCOL, LATCOL, and p_score
        parking_score = parking_result[["LONCOL", "LATCOL", "p_score"]]

        return parking_score, parkinggrid

    # Function to process car information and generate stay orders
    def process_car_data(car_info_path, step_length=300):
        car_infos = pd.read_csv(car_info_path)
        car_infos['time'] = pd.to_datetime(car_infos['time'])

        stay_infos = car_infos.sort_values(by=['carid', 'time'])
        stay_infos['timegap'] = (-stay_infos['time'].diff(-1).dt.total_seconds()).fillna(1000000).astype(int)
        stay_infos['etime'] = stay_infos['time'].shift(-1)

        stay_order = stay_infos[stay_infos['timegap'] > step_length][['carid', 'time', 'etime', 'soc', 'lon', 'lat']]
        stay_order.columns = ['carid', 'stime', 'etime', 'soc', 'lon', 'lat']
        stay_order['duration'] = (stay_order['etime'] - stay_order['stime']).dt.total_seconds()
        stay_order = stay_order[stay_order['duration'] > 0]

        # Save stay orders to CSV
        # stay_order.to_csv('output/stay_orders_from_car.csv', index=False)

        return stay_order

    # Main function to process both station and car data
    def process_all_data(station_info_path, car_info_path, step_length=300):
        # Process station data to get charging and waiting orders
        charging_orders, waiting_orders = process_station_data(station_info_path, step_length)

        # Process car data to get stay orders
        stay_orders = process_car_data(car_info_path, step_length)

        return charging_orders, waiting_orders, stay_orders

    def load_and_filter_orders(order):
        """加载订单数据并按时间进行过滤"""
        order = order.groupby(["station_id", "lon", "lat"]).size().reset_index()
        order.columns = ["station_id", "lon", "lat", "order"]
        return order

    def map_orders_to_grid(order, area, params):
        """将订单映射到网格"""
        grid_rec, params_a = tbd.area_to_grid(area, params=params)
        order["LONCOL"], order["LATCOL"] = tbd.GPS_to_grid(order["lon"], order["lat"], params)
        order_grid = order.groupby(["LONCOL", "LATCOL"])["order"].sum().reset_index()
        grid_rec1 = pd.merge(grid_rec, order_grid, on=["LONCOL", "LATCOL"], how="left")
        grid_rec1.fillna(0, inplace=True)
        return grid_rec1[["LONCOL", "LATCOL", "order"]]

    def calculate_sum_order(order_result):
        """计算每个网格周围的订单总数"""
        sumorder = pd.DataFrame(columns=["LONCOL", "LATCOL", "order", "sum_order"])
        for index, row in order_result.iterrows():
            lon = row["LONCOL"]
            lat = row["LATCOL"]
            order1 = row["order"]
            sum_order = order_result[((order_result["LONCOL"] == lon - 1) |
                                      (order_result["LONCOL"] == lon) |
                                      (order_result["LONCOL"] == lon + 1)) &
                                     ((order_result["LATCOL"] == lat - 1) |
                                      (order_result["LATCOL"] == lat) |
                                      (order_result["LATCOL"] == lat + 1))]["order"].sum()
            sumorder = pd.concat([sumorder, pd.DataFrame([[lon, lat, order1, sum_order]],
                                                         columns=["LONCOL", "LATCOL", "order", "sum_order"])],
                                 ignore_index=True)
        return sumorder

    def normalize_scores(data, column, new_column_name):
        """归一化得分"""
        scaler = MinMaxScaler()
        data[new_column_name] = scaler.fit_transform(data[[column]])
        data[new_column_name] = data[new_column_name] * 100
        return data

    def process_poi_score(poi_data, area, city_name="上海市", poi_type="公司企业", score_column_name="poi_c_score"):
        # Load the POI data

        # Generate grid parameters based on the area
        params = tbd.area_to_params(area, accuracy=1000)

        # Filter POI data for the specified city
        poi_data = poi_data[poi_data["pname"] == city_name]

        # Extract the first type from the 'type' column
        poi_data['ttype'] = poi_data['type'].apply(lambda x: x.split(';')[0])

        # Split the 'location' column into 'lon' and 'lat'
        poi_data["lon"] = poi_data['location'].apply(lambda x: x.split(',')[0]).astype(float)
        poi_data["lat"] = poi_data['location'].apply(lambda x: x.split(',')[1]).astype(float)

        # Filter POI data by the specified poi_type (e.g., "公司企业")
        poi_data = poi_data[poi_data["ttype"] == poi_type]

        # Convert GPS to grid coordinates (LONCOL, LATCOL)
        poi_data["LONCOL"], poi_data["LATCOL"] = tbd.GPS_to_grid(poi_data["lon"], poi_data["lat"], params)

        # Group by grid coordinates and count the POIs in each grid
        poigrid = poi_data.groupby(["LONCOL", "LATCOL"]).size().reset_index()
        poigrid.columns = ["LONCOL", "LATCOL", "poi"]

        # Create the grid from the area and merge with the POI grid data
        grid_rec, _ = tbd.area_to_grid(area, params=params)
        grid_rec_poi = pd.merge(grid_rec, poigrid, on=["LONCOL", "LATCOL"], how="left")

        # Fill missing POI counts with 0
        grid_rec_poi.fillna(0, inplace=True)

        # Extract the final POI result with LONCOL, LATCOL, and POI count
        poicommercial_result = grid_rec_poi[["LONCOL", "LATCOL", "poi"]]

        # Normalize the POI counts using MinMaxScaler
        scaler = MinMaxScaler()
        poicommercial_result[score_column_name] = scaler.fit_transform(poicommercial_result[["poi"]])

        # Scale the POI score to a 0-100 range
        poicommercial_result[score_column_name] = poicommercial_result[score_column_name] * 100

        # Prepare the final result DataFrame with renamed columns
        poi_com_score = poicommercial_result[["LONCOL", "LATCOL", score_column_name]]

        # Return the final POI commercial score DataFrame
        return poi_com_score

    def process_price_data(price_file, area, params):
        """处理价格数据并计算价格得分"""
        price = pd.read_csv(price_file)
        price["geometry"] = gpd.points_from_xy(price["lon"], price["lat"])
        price = gpd.GeoDataFrame(price, geometry=price["geometry"])
        price.crs = "EPSG:4326"
        price = price.to_crs("EPSG:32651")
        buffer = price.buffer(1000)
        price = gpd.GeoDataFrame(price, geometry=buffer)
        price = price.to_crs("EPSG:4326")

        grid_rec, _ = tbd.area_to_grid(area, params=params)
        pricegrid = gpd.sjoin(grid_rec, price)
        pricegrid = pricegrid.groupby(["LONCOL", "LATCOL"])["price"].mean().reset_index()

        # Normalize price
        scaler = MinMaxScaler()
        pricegrid["price_score"] = scaler.fit_transform(pricegrid[["price"]])
        pricegrid["price_score"] = 1 - pricegrid["price_score"]  # 价格越高，得分越低
        pricegrid["price_score"] = pricegrid["price_score"] * 100
        return pricegrid[["LONCOL", "LATCOL", "price_score"]]

    def compute_satisfaction(order_data, parking_data):
        """计算满意度得分"""
        order_data = order_data[["LONCOL", "LATCOL", "order"]]
        parking_data = parking_data[["LONCOL", "LATCOL", "parking"]]
        satisfaction = pd.merge(parking_data, order_data, on=["LONCOL", "LATCOL"])
        satisfaction["total"] = satisfaction["parking"] + satisfaction["order"]
        satisfaction["rate"] = satisfaction["order"] / satisfaction["total"]

        scaler = MinMaxScaler()
        satisfaction["satisfaction_score"] = scaler.fit_transform(satisfaction[["rate"]])
        satisfaction["satisfaction_score"] = 1 - satisfaction["satisfaction_score"]  # 满意度得分计算
        satisfaction["satisfaction_score"] = satisfaction["satisfaction_score"] * 100
        return satisfaction[["LONCOL", "LATCOL", "satisfaction_score"]]

    def process_station_score(station_file_path, grid_area, params, score_column_name="station_score"):
        # Load the charging station data
        charge_station = pd.read_csv(station_file_path)

        # Convert GPS to grid coordinates (LONCOL, LATCOL)
        charge_station["LONCOL"], charge_station["LATCOL"] = tbd.GPS_to_grid(charge_station["lon"],
                                                                             charge_station["lat"], params)

        # Group by grid coordinates and count the charging stations in each grid
        stationgrid = charge_station.groupby(["LONCOL", "LATCOL"]).size().reset_index()
        stationgrid.columns = ["LONCOL", "LATCOL", "charge_station"]

        # Create the grid from the area and merge with the station grid data
        grid_rec, _ = tbd.area_to_grid(grid_area, params=params)
        grid_rec5 = pd.merge(grid_rec, stationgrid, on=["LONCOL", "LATCOL"], how="left")

        # Fill missing charge station counts with 0
        grid_rec5.fillna(0, inplace=True)

        # Extract the final station result with LONCOL, LATCOL, and station count
        station_result = grid_rec5[["LONCOL", "LATCOL", "charge_station"]]

        # Normalize the station counts using MinMaxScaler
        scaler = MinMaxScaler()
        station_result[score_column_name] = scaler.fit_transform(station_result[["charge_station"]])

        # Reverse the score (1 - normalized value)
        station_result[score_column_name] = 1 - station_result[score_column_name]

        # Scale the station score to a 0-100 range
        station_result[score_column_name] = station_result[score_column_name] * 100

        # Prepare the final result DataFrame with LONCOL, LATCOL, and station score
        station_score = station_result[["LONCOL", "LATCOL", score_column_name]]

        # Return the final station score DataFrame
        return station_score

    def calculate_total_score(order_score, waiting_score, parking_score, satisfaction_score,
                              poi_com_score, poi_e_score, poi_l_score, poi_edu_score,
                              station_score, price_score, order_score3, order_score5):
        # Merge all the dataframes on 'LONCOL' and 'LATCOL'
        total_score = pd.merge(order_score, waiting_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, parking_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, satisfaction_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, poi_com_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, poi_e_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, poi_l_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, poi_edu_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, station_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, price_score, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, order_score3, on=["LONCOL", "LATCOL"], how="outer")
        total_score = pd.merge(total_score, order_score5, on=["LONCOL", "LATCOL"], how="outer")

        return total_score

    def entropy_weighted_score(df, columns):
        """
        Calculate the total score using the entropy weight method and scale it to a 100-point system.

        Args:
            df (pd.DataFrame): The dataframe containing the relevant columns.
            columns (list): The list of columns to calculate the total score.

        Returns:
            pd.DataFrame: The dataframe with an additional 'total_score' column, scaled to 100 points.
        """

        # Step 1: Calculate entropy for each column
        def calculate_entropy(col):
            col_sum = col.sum()
            if col_sum == 0:
                return 0
            p = col / col_sum
            p = p.replace(0, np.nan)  # Replace 0s to avoid log2(0)
            return -np.nansum(p * np.log2(p))  # np.nansum ignores NaNs

        # Apply entropy calculation to each column
        entropy_values = df[columns].apply(calculate_entropy)

        # Step 2: Calculate weights based on entropy
        weights = (1 - entropy_values) / (1 - entropy_values).sum()

        # Step 3: Calculate the weighted total score for each row
        df['total_score_raw'] = df[columns].dot(weights)

        # Step 4: Scale total_score to a 100-point system (min-max normalization)
        min_score = df['total_score_raw'].min()
        max_score = df['total_score_raw'].max()
        df['total_score'] = 100 * (df['total_score_raw'] - min_score) / (max_score - min_score)

        # Optionally, drop the raw score column if you don't need it
        df = df.drop(columns=['total_score_raw'])

        return df, weights

    # 示例流程：调用这些函数得到结果
    def main(area, station_info_path, car_info_path, poi_file_path, price_file, station_file_path):
        params = tbd.area_to_params(area, accuracy=1000)
        charging_orders, waiting_orders, stay_orders = process_all_data(station_info_path, car_info_path)
        # 订单数据处理
        order = load_and_filter_orders(charging_orders)
        order_grid = map_orders_to_grid(order, area, params)
        sum_order = calculate_sum_order(order_grid)
        order_score = normalize_scores(sum_order, "sum_order", "o_score")
        order_score = order_score[["LONCOL", "LATCOL", "o_score"]]
        # 3year
        order3 = load_and_filter_orders(charging_orders)
        order_grid3 = map_orders_to_grid(order3, area, params)
        sum_order3 = calculate_sum_order(order_grid3)
        order_score3 = normalize_scores(sum_order3, "sum_order", "o_score(3year)")
        order_score3 = order_score3[["LONCOL", "LATCOL", "o_score(3year)"]]
        # 5year
        order5 = load_and_filter_orders(charging_orders)
        order_grid5 = map_orders_to_grid(order5, area, params)
        sum_order5 = calculate_sum_order(order_grid5)
        order_score5 = normalize_scores(sum_order5, "sum_order", "o_score(5year)")
        order_score5 = order_score5[["LONCOL", "LATCOL", "o_score(5year)"]]
        # waiting_score
        waiting_score = process_waiting_score(waiting_orders, area, params)
        # parking_score
        parking_score, parkinggrid = process_parking_score(stay_orders, area, params)
        poi_data = pd.read_excel(poi_file_path)
        # commercial
        poi_com_score = process_poi_score(poi_data, area=area, poi_type="公司企业", score_column_name="poi_c_score")
        # entertainment
        poi_e_score = process_poi_score(poi_data, area=area, poi_type="购物服务", score_column_name="poi_e_score")
        # life service
        poi_l_score = process_poi_score(poi_data, area=area, poi_type="生活服务", score_column_name="poi_l_score")
        # education
        poi_edu_score = process_poi_score(poi_data, area=area, poi_type="科教文化服务",
                                          score_column_name="poi_edu_score")
        # 处理价格数据

        price_score = process_price_data(price_file, area, params)

        # 计算满意度得分
        satisfaction_score = compute_satisfaction(order_grid, parkinggrid)
        # 充电站得分

        # Process and get station score
        station_score = process_station_score(station_file_path, grid_area=area, params=params,
                                              score_column_name="station_score")
        # 返回或保存结果
        total_score = calculate_total_score(order_score, waiting_score, parking_score, satisfaction_score,
                                            poi_com_score, poi_e_score, poi_l_score, poi_edu_score,
                                            station_score, price_score, order_score3, order_score5)
        total_score.fillna(0, inplace=True)
        columns = total_score.columns[2::]
        total_score_df, weights = entropy_weighted_score(total_score, columns)
        grid_rec, params_a = tbd.area_to_grid(area, params=params)
        grid = pd.merge(grid_rec, total_score, on=["LONCOL", "LATCOL"], how="left")
        grid.fillna(0, inplace=True)

        return grid

    def scale_to_percentage(df, column):

        min_val = df[column].min()
        max_val = df[column].max()

        # Step 2: Apply the min-max scaling formula to scale between 0 and 100
        df[column] = 100 * (df[column] - min_val) / (max_val - min_val)

        return df

    results = main(area, station_info_path, car_info_path, poi_file_path, price_file, station_file_path)
    results.columns = ['LONCOL', 'LATCOL', 'geometry', 'Charge_demand_score', 'Waiting_score', 'Potential_demand_score',
                    'Demand_satisfaction_score', 'poi_c_score', 'poi_e_score', 'poi_l_score',
                    'poi_edu_score', 'Station_score', 'price_score', 'Future_demand_score(3years)',
                    'Future_demand_score(5years)', 'total_score_raw', 'total_score']
    results["Poi_score"] = results["poi_c_score"] + results["poi_e_score"] + results["poi_l_score"] + results[
        "poi_edu_score"]
    results = results[["geometry", 'Charge_demand_score', 'Waiting_score', 'Potential_demand_score',
                    'Demand_satisfaction_score', 'Station_score', 'Future_demand_score(3years)', "Poi_score",
                    'Future_demand_score(5years)', "total_score"]]

    results = scale_to_percentage(results, "Poi_score")
    return results

def area_classification_v2(grid,bins):
    def transgird(grid):
        bounds=[113.00558,22.510,114.0100,23.90000]
        grid_rec,params=tbd.area_to_grid(bounds,accuracy=5000)
        grid['centroid'] = grid.geometry.centroid
        grid['lon'] = grid['centroid'].apply(lambda point: point.x)
        grid['lat'] = grid['centroid'].apply(lambda point: point.y)
        grid["LONCOL"], grid["LATCOL"] = tbd.GPS_to_grid(grid["lon"], grid["lat"], params=params)
        grid=grid.groupby(["LONCOL","LATCOL"])["total_score"].mean().reset_index()
        polygon=tbd.grid_to_polygon([grid["LONCOL"],grid["LATCOL"]], params)
        grid=gpd.GeoDataFrame(grid,geometry=polygon)
        return grid
    
    def classify_grid(grid,bins):
        """
        输入:
            grid (pd.DataFrame): 包含格网和评分的数据
        输出:
            pd.DataFrame: 排序后的格网数据，并按评分分类
        """
        grid.sort_values(by="total_score", ascending=False, inplace=True)
        grid["class"] = pd.cut(grid["total_score"], bins=bins, labels=range(bins))
        return grid
    grid=transgird(grid)
    grid=classify_grid(grid,bins)
    return grid

import pandas as pd
import geopandas as gpd
import transbigdata as tbd  # 假设用来处理GPS与网格转换的模块
import ast
import numpy as np
def urban_capacity(station_info_path,stations_num=5000):
    def process_station_data(station_info_path, step_length=300):
            # Load station info
            station_info = pd.read_csv(station_info_path)
            station_info = station_info[station_info['num_current_car'] > 0]

            # Extract relevant columns and clean data
            station_info_table = station_info[
                ['station_id', 'lon', 'lat', 'max_capacity', 'charge_speed_station']].drop_duplicates().copy()
            station_info = station_info[['station_id', 'time', 'current_car', 'waiting_car']]

            # Convert time and handle current/waiting car columns
            station_info['time'] = pd.to_datetime(station_info['time'])
            station_info['current_car'] = station_info['current_car'].apply(lambda a: ast.literal_eval(a))
            station_info['waiting_car'] = station_info['waiting_car'].apply(lambda a: ast.literal_eval(a))
            station_info.sort_values(by=['station_id', 'time'], inplace=True)

            # Process current car infos
            current_car_infos = station_info[['station_id', 'time', 'current_car']].explode('current_car')
            current_car_infos = current_car_infos[~current_car_infos['current_car'].isnull()]
            current_car_infos = current_car_infos.sort_values(by=['current_car', 'time'])[
                ['current_car', 'time', 'station_id']]

            # Process waiting car infos
            waiting_car_infos = station_info[['station_id', 'time', 'waiting_car']].explode('waiting_car')
            waiting_car_infos = waiting_car_infos[~waiting_car_infos['waiting_car'].isnull()]
            waiting_car_infos = waiting_car_infos.sort_values(by=['waiting_car', 'time'])[
                ['waiting_car', 'time', 'station_id']]

            # Helper function to generate charging or waiting orders
            def get_charging_order(car_infos, step_length):
                car_infos['timegap'] = car_infos['time'].diff().dt.total_seconds().fillna(1000000).astype(int)
                car_infos['order_id'] = (car_infos['timegap'] > step_length).cumsum()
                charge_info_s = car_infos.groupby(['current_car', 'order_id']).first().reset_index()
                charge_info_e = car_infos.groupby(['current_car', 'order_id']).last().reset_index()
                charging_order = pd.merge(charge_info_s, charge_info_e, on=['current_car', 'order_id', 'station_id'])
                charging_order = charging_order[['current_car', 'order_id', 'time_x', 'time_y', 'station_id']]
                charging_order.columns = ['carid', 'order_id', 'stime', 'etime', 'station_id']
                charging_order['duration'] = (charging_order['etime'] - charging_order['stime']).dt.total_seconds()
                charging_order = charging_order[charging_order['duration'] > 0]
                return charging_order

            # Generate charging and waiting orders
            charging_orders = get_charging_order(current_car_infos, step_length)
            waiting_orders = get_charging_order(waiting_car_infos.rename(columns={'waiting_car': 'current_car'}),
                                                step_length)

            # Merge station info details into orders
            charging_orders = pd.merge(charging_orders, station_info_table)
            waiting_orders = pd.merge(waiting_orders, station_info_table)

            # Save charging and waiting orders to CSV
            # charging_orders.to_csv('output/charging_orders_from_station.csv', index=False)
            # waiting_orders.to_csv('output/waiting_orders_from_station.csv', index=False)

            return charging_orders, waiting_orders

    def load_and_filter_orders(order):
            """加载订单数据并按时间进行过滤"""
            order = order.groupby(["station_id","max_capacity", "lon", "lat"]).size().reset_index()
            order.columns = ["station_id","max_capacity", "lon", "lat", "order"]
            return order
    import pandas as pd
    def charge_num(order,stations_num = 5000):
        # Sort the DataFrame by 'total_score' and reset index
        df = order.sort_values(by='order', ascending=False).reset_index(drop=True)

        # Divide sites into 10 levels using pd.qcut
        df['level'] = pd.qcut(df['order'], 10, labels=False, duplicates='drop') + 1

        # Calculate the sum of total_score for each level
        level_score_sum = df.groupby('level')['order'].sum()

        # Calculate the proportion of total_score for each level
        total_score_sum = level_score_sum.sum()
        level_score_ratio = level_score_sum / total_score_sum

        # Total number of charging stations

        # Calculate the number of stations to allocate for each level (based on total_score ratio)
        level_station_allocation = (level_score_ratio * stations_num).astype(int)

        # Initialize allocated stations to 0
        df['allocated_stations'] = 0

        # Allocate stations for each level, excluding level 1
        for level, num_stations in level_station_allocation.items():
            if level != 1:  # Exclude level 1 from allocation
                level_stations = df[df['level'] == level].index
                df.loc[level_stations, 'allocated_stations'] = num_stations // len(level_stations)
        df["num"]=df["max_capacity"]+df["allocated_stations"]
        df=df[["station_id","max_capacity","lon","lat","num"]]
        df=gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df["lon"],df["lat"]))
        return df

    charging_orders, waiting_orders=process_station_data(station_info_path, step_length=300)
    order = load_and_filter_orders(charging_orders)
    station=charge_num(order,stations_num = stations_num)
    return station