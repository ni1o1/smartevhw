from flask import Flask, request
import json
import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
from flask import send_from_directory

from flask_cors import CORS

app = Flask(__name__)

# 启用 CORS，允许跨域请求
CORS(app)


@app.route('/DownloadRoad', methods=['GET'])
def DownloadRoad():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/SimUrbanRoad', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadUrbanStation', methods=['GET'])
def DownloadUrbanStation():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/UrbanStation', request.args.get('filename'), as_attachment=True)
    
@app.route('/DownloadAreaClassOutput', methods=['GET'])
def DownloadAreaClassOutput():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/AreaClassOutput', request.args.get('filename'), as_attachment=True)
    
@app.route('/DownloadAreaClassInput', methods=['GET'])
def DownloadAreaClassInput():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/AreaClassInput', request.args.get('filename'), as_attachment=True)
    
@app.route('/DownloadGridScoreInput', methods=['GET'])
def DownloadGridScoreInput():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/GridScoreInput', request.args.get('filename'), as_attachment=True)

@app.route('/DownloadGridScoreOutput', methods=['GET'])
def DownloadGridScoreOutput():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/GridScoreOutput', request.args.get('filename'), as_attachment=True)

@app.route('/DownloadSimUrbanConfig', methods=['GET'])
def DownloadSimUrbanConfig():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/SimUrbanConfig', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadSimHighwayEVinfo', methods=['GET'])
def DownloadSimHighwayEVinfo():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/SimHighwayEVinfo', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadTraj', methods=['GET'])
def DownloadTraj():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/Trajectory', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadLifepat', methods=['GET'])
def DownloadLifepat():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/Lifepattern', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadKeyloc', methods=['GET'])
def DownloadKeyloc():
    if request.method == 'GET':
        # 从后端获取文件到前端
        # 压缩文件夹为zip，然后下载
        import zipfile
        with zipfile.ZipFile('./Files/Keylocation/'+request.args.get('filename')+'.zip', 'w') as z:
            for foldername, subfolders, filenames in os.walk('./Files/Keylocation/'+request.args.get('filename')):
                for filename in filenames:
                    z.write(os.path.join(foldername, filename), filename)
        return send_from_directory('./Files/Keylocation', request.args.get('filename')+'.zip', as_attachment=True)


@app.route('/DownloadSimUrbanRes', methods=['GET'])
def DownloadSimUrbanRes():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/SimUrbanResult', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadSimHighwayRes', methods=['GET'])
def DownloadSimHighwayRes():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/SimHighwayResult', request.args.get('filename'), as_attachment=True)


@app.route('/DownloadSelectRes', methods=['GET'])
def DownloadSelectRes():
    if request.method == 'GET':
        # 从后端获取文件到前端
        return send_from_directory('./Files/UrbanStationSelection', request.args.get('filename'), as_attachment=True)

def upload_file(file, path):
    file.save(os.path.join(path, file.filename))
    return {
        'status': 'success'
    }


@app.route('/UploadRoad', methods=['POST'])
def UploadRoad():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/SimUrbanRoad')
        return result


@app.route('/UploadUrbanStation', methods=['POST'])
def UploadUrbanStation():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/UrbanStation')
        return result


@app.route('/UploadSimUrbanConfig', methods=['POST'])
def UploadSimUrbanConfig():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/SimUrbanConfig')
        return result


@app.route('/UploadSimHighwayEVinfo', methods=['POST'])
def UploadSimHighwayEVinfo():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/SimHighwayEVinfo')
        return result

@app.route('/UploadAreaClassInput', methods=['POST'])
def UploadAreaClassInput():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/AreaClassInput')
        return result

@app.route('/UploadGridScoreInput', methods=['POST'])
def UploadGridScoreInput():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/GridScoreInput')
        return result
    
@app.route('/UploadPOI', methods=['POST'])
def UploadPOI():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/POI')
        return result
    
@app.route('/UploadTraj', methods=['POST'])
def UploadTraj():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/Trajectory')
        return result


@app.route('/UploadLifepat', methods=['POST'])
def UploadLifepat():
    if request.method == 'POST':
        result = upload_file(request.files['file'], './Files/Lifepattern')
        return result

@app.route('/UploadLifepatList', methods=['POST'])
def UploadLifepatList():
    if request.method == 'POST':
        # Get the name of the uploaded file
        file = request.files['file']
        # Save the file to ./uploads
        file.save(os.path.join('./Files/Lifepattern', file.filename))
        # Return 201 CREATED
        return '', 201
    
@app.route('/UploadKeyloc', methods=['POST'])
def UploadKeyloc():
    if request.method == 'POST':
        # Get the name of the uploaded file
        file = request.files['file']
        # 后缀为zip的文件
        # 检查其中的文件是否包含三个文件夹
        import zipfile
        with zipfile.ZipFile(file, 'r') as z:
            filelist = z.namelist()
            if 'h0/' not in filelist:
                result = {
                    'status': 'error',
                    'message': '缺少h0文件'
                }
            elif 'h2w/' not in filelist:
                result = {
                    'status': 'error',
                    'message': '缺少h2w文件'
                }
            elif 'Others/' not in filelist:
                result = {
                    'status': 'error',
                    'message': '缺少Others文件'
                }
            else:
                z.extractall('./Files/Keylocation/' +
                             file.filename.split('.')[0])
                result = {
                    'status': 'success',
                    'message': '文件上传成功'
                }
        return result


def get_file_info(path, endswith='json'):
    KeylocList = os.listdir(path)
    # Return the date of the files
    if endswith == 'isdir':
        KeylocList = [file for file in KeylocList if os.path.isdir(
            os.path.join(path, file))]
    elif endswith == 'all':
        KeylocList
    else:
        KeylocList = [file for file in KeylocList if file.endswith(
            endswith) or file.endswith('.generating')]

    def get_file_date(file_path, file_name):
        # 获取文件的最后修改时间（返回的是时间戳）
        modification_time = Path(file_path).stat().st_mtime

        # 将时间戳转换为可读的日期格式
        modification_date = datetime.datetime.fromtimestamp(
            modification_time).strftime('%Y-%m-%d %H:%M:%S')
        if file_name.endswith('.generating'):
            generated = False
        else:
            generated = True
        result = {
            'name': file_name,
            'date': modification_date,
            'generated': generated
        }
        return result
    KeylocList = [get_file_date(os.path.join(path, file), file)
                  for file in KeylocList]
    KeylocList = sorted(
        KeylocList, key=lambda x: x['date'], reverse=True)
    returndata = {
        'data': KeylocList
    }

    return returndata


@app.route('/GetRoadList', methods=['GET'])
def GetRoadList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/SimUrbanRoad', 'graphml')
        return json.dumps(returndata)

@app.route('/GetPOIList', methods=['GET'])
def GetPOIList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/POI', 'csv')
        return json.dumps(returndata)

@app.route('/GetKeylocList', methods=['GET'])
def GetKeylocList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/Keylocation', 'isdir')
        return json.dumps(returndata)


@app.route('/GetSimHighwayEVinfoList', methods=['GET'])
def GetSimHighwayEVinfoList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/SimHighwayEVinfo', 'json')
        return json.dumps(returndata)


@app.route('/GetSimHighwayResList', methods=['GET'])
def GetSimHighwayResult():
    if request.method == 'GET':
        returndata = get_file_info('./Files/SimHighwayResult', 'csv')
        return json.dumps(returndata)


@app.route('/GetProvinceList', methods=['GET'])
def GetProvinceList():
    if request.method == 'GET':
        returndata = {
            'data': [
                {'name': '北京市'},
                {'name': '天津市'},
                {'name': '河北省'},
            ]
        }
        return json.dumps(returndata)

@app.route('/GetGridScoreInputList', methods=['GET'])
def GetGridScoreInputList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/GridScoreInput', 'all')
        return json.dumps(returndata)

@app.route('/GetGridScoreOutputList', methods=['GET'])
def GetGridScoreOutputList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/GridScoreOutput', 'json')
        return json.dumps(returndata)

@app.route('/GetAreaClassInputList', methods=['GET'])
def GetAreaClassInputList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/AreaClassInput', 'xlsx')
        return json.dumps(returndata)
    
@app.route('/GetAreaClassOutputList', methods=['GET'])
def GetAreaClassOutputList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/AreaClassOutput', 'json')
        return json.dumps(returndata)
    
@app.route('/GetAreaClassCityList', methods=['GET'])
def GetAreaClassCityList():
    if request.method == 'GET':
        filename = request.args.get('filename')
        cities = pd.read_excel(f'./Files/AreaClassInput/{filename}')['市'].drop_duplicates().to_list()
        returndata = {
            'data': cities
        }
        return json.dumps(returndata)

@app.route('/GetLifepatList', methods=['GET'])
def GetLifepatList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/Lifepattern', 'pkl')
        return json.dumps(returndata)


@app.route('/GetSimUrbanConfigList', methods=['GET'])
def GetSimUrbanConfigList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/SimUrbanConfig', 'json')
        return json.dumps(returndata)


@app.route('/GetSelectResList', methods=['GET'])
def GetSelectResList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/UrbanStationSelection', 'json')
        return json.dumps(returndata)


@app.route('/GetSimUrbanResList', methods=['GET'])
def GetSimUrbanResList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/SimUrbanResult', 'csv')
        return json.dumps(returndata)


@app.route('/GetUrbanStationList', methods=['GET'])
def GetUrbanStationList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/UrbanStation', 'csv')
        return json.dumps(returndata)


@app.route('/GetTrajList', methods=['GET'])
def GetTrajList():
    if request.method == 'GET':
        returndata = get_file_info('./Files/Trajectory', 'csv')
        return json.dumps(returndata)


@app.route('/DeleteRoad', methods=['GET'])
def DeleteRoad():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/SimUrbanRoad/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteTraj', methods=['GET'])
def DeleteTraj():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/Trajectory/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteConfig', methods=['GET'])
def DeleteConfig():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/SimUrbanConfig/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteUrbanStation', methods=['GET'])
def DeleteUrbanStation():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/UrbanStation/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)

@app.route('/DeleteAreaClassInput', methods=['GET'])
def DeleteAreaClassInput():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/AreaClassInput/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)
    
@app.route('/DeleteGridScoreInput', methods=['GET'])
def DeleteGridScoreInput():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/GridScoreInput/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)

@app.route('/DeleteGridScoreOutput', methods=['GET'])
def DeleteGridScoreOutput():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/GridScoreOutput/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)
    
@app.route('/DeleteAreaClassOutput', methods=['GET'])
def DeleteAreaClassOutput():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/AreaClassOutput/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteSimUrbanRes', methods=['GET'])
def DeleteSimUrbanRes():
    if request.method == 'GET':
        filename = request.args.get('filename')

        filename = filename.replace('car_infos', 'station_infos')
        os.remove('./Files/SimUrbanResult/'+filename)
        filename = filename.replace('station_infos', 'car_infos')
        os.remove('./Files/SimUrbanResult/'+filename)

        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteSimHighwayRes', methods=['GET'])
def DeleteSimHighwayRes():
    if request.method == 'GET':
        filename = request.args.get('filename')
        if 'cars_vis' in filename:
            os.remove('./Files/SimHighwayResult/'+filename)
            os.remove('./Files/SimHighwayResult/'+filename.replace('cars_vis', 'stations_vis'))
            os.remove('./Files/SimHighwayResult/'+filename.replace('cars_vis', 'charge_orders'))
        elif 'stations_vis' in filename:
            os.remove('./Files/SimHighwayResult/'+filename)
            os.remove('./Files/SimHighwayResult/'+filename.replace('stations_vis', 'cars_vis'))
            os.remove('./Files/SimHighwayResult/'+filename.replace('stations_vis', 'charge_orders'))
        elif 'charge_orders' in filename:
            os.remove('./Files/SimHighwayResult/'+filename)
            os.remove('./Files/SimHighwayResult/'+filename.replace('charge_orders', 'cars_vis'))
            os.remove('./Files/SimHighwayResult/'+filename.replace('charge_orders', 'stations_vis'))

        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)


@app.route('/DeleteKeyloc', methods=['GET'])
def DeleteKeyloc():
    if request.method == 'GET':
        filename = request.args.get('filename')

        # 检查是否为文件夹
        if os.path.isdir('./Files/Keylocation/'+filename):

            import shutil
            shutil.rmtree('./Files/Keylocation/'+filename)
            returndata = {
                'status': 'success'
            }
        elif os.path.isfile('./Files/Keylocation/'+filename):
            os.remove('./Files/Keylocation/'+filename)
            returndata = {
                'status': 'success'
            }
        else:
            returndata = {
                'status': 'error'
            }

        return json.dumps(returndata)


@app.route('/DeleteLifepat', methods=['GET'])
def DeleteLifepat():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/Lifepattern/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)

@app.route('/DeleteSelectRes', methods=['GET'])
def DeleteSelectRes():
    if request.method == 'GET':
        filename = request.args.get('filename')
        os.remove('./Files/UrbanStationSelection/'+filename)
        returndata = {
            'status': 'success'
        }
        return json.dumps(returndata)

@app.route('/PreviewTraj', methods=['GET'])
def PreviewTraj():
    if request.method == 'GET':
        filename = request.args.get('filename')
        filetype = request.args.get('filetype')

        if filetype == 'traj':
            traj = pd.read_csv('./Files/Trajectory/'+filename)
            traj = traj[(traj['lon'] > -180) & (traj['lon'] < 180)
                        & (traj['lat'] > -90) & (traj['lat'] < 90)]
            traj_count = traj.groupby(
                ['lon', 'lat']).size().reset_index(name='count')
            params = tbd.area_to_params([traj_count['lon'].iloc[0], traj_count['lat'].iloc[0],
                                        traj_count['lon'].iloc[0], traj_count['lat'].iloc[0]], 1000)
            traj_count['LONCOL'], traj_count['LATCOL'] = tbd.GPS_to_grid(
                traj_count['lon'], traj_count['lat'], params)
            traj_count = traj_count.groupby(
                ['LONCOL', 'LATCOL']).size().reset_index(name='count')
            traj_count['geometry'] = tbd.grid_to_polygon(
                [traj_count['LONCOL'], traj_count['LATCOL']], params)
            traj_count = gpd.GeoDataFrame(traj_count, geometry='geometry')
            traj_count['maxcount'] = traj_count['count'].quantile(0.9995)
            return traj_count.to_json()
        if filetype == 'highway':
            if 'cars_vis' in filename:
                traj = pd.read_csv('./Files/SimHighwayResult/'+filename)
                traj = traj[(traj['lon'] > -180) & (traj['lon'] < 180)
                            & (traj['lat'] > -90) & (traj['lat'] < 90)]
                traj_count = traj.groupby(
                    ['lon', 'lat']).size().reset_index(name='count')

                params = tbd.area_to_params([traj_count['lon'].iloc[0], traj_count['lat'].iloc[0],
                                            traj_count['lon'].iloc[0], traj_count['lat'].iloc[0]], 1000)
                traj_count['LONCOL'], traj_count['LATCOL'] = tbd.GPS_to_grid(
                    traj_count['lon'], traj_count['lat'], params)
                traj_count = traj_count.groupby(
                    ['LONCOL', 'LATCOL']).size().reset_index(name='count')
                traj_count['geometry'] = tbd.grid_to_polygon(
                    [traj_count['LONCOL'], traj_count['LATCOL']], params)
                traj_count = gpd.GeoDataFrame(traj_count, geometry='geometry')
                traj_count['maxcount'] = traj_count['count'].quantile(0.9995)

                filename = filename.replace('cars_vis','stations_vis')
                station_vis = pd.read_csv('./Files/SimHighwayResult/'+filename)
                station_vis = station_vis[(station_vis['lon'] > -180) & (
                    station_vis['lon'] < 180) & (station_vis['lat'] > -90) & (station_vis['lat'] < 90)]
                station_vis_origin = station_vis.copy()
                charge_orders = pd.read_csv(
                    './Files/SimHighwayResult/'+filename.replace('stations_vis', 'charge_orders'))

                station_vis = pd.merge(charge_orders.groupby(['charge_node_id','场站名称']).size().rename('count').reset_index().sort_values(
                    'count', ascending=False), station_vis[['charge_node_id', 'lon', 'lat']].drop_duplicates(), on='charge_node_id')

                station_vis['geometry'] = gpd.points_from_xy(
                    station_vis['lon'], station_vis['lat'])
                station_vis = gpd.GeoDataFrame(
                    station_vis, geometry='geometry')
                station_vis.drop(['lon', 'lat'], axis=1, inplace=True)
                station_vis['maxcount'] = station_vis['count'].quantile(0.9995)
                station_vis.crs = 'EPSG:4326'
                station_vis['geometry'] = station_vis.to_crs('EPSG:3857').buffer(1000).buffer(
                    5000*station_vis['count']/station_vis['maxcount']).to_crs('EPSG:4326')

                '''统计充电功率'''
                station_vis_origin['charged_power'] = station_vis_origin.groupby(
                    'charge_node_id')['charging_demand'].diff().fillna(0)
                station_vis_origin['weekday'] = pd.to_datetime(
                    station_vis_origin['time']).dt.weekday
                station_vis_origin['hour'] = pd.to_datetime(
                    station_vis_origin['time']).dt.hour
                station_vis_origin['order_num'] = station_vis_origin['current_car_count']
                charged_power = station_vis_origin[[
                    'weekday', 'hour', 'lon', 'lat', 'charged_power']]
                charged_power = charged_power[charged_power['charged_power'] > 0]
                '''统计充电订单'''
                order_num = station_vis_origin[[
                    'lon', 'lat', 'weekday', 'hour', 'time', 'order_num']]

                total={}
                a = station_vis.to_json()
                b = charged_power.to_dict('records')
                c = order_num.to_dict('records')
                d = traj_count.to_json()
                import json
                total['station_vis'] = json.loads(a)
                total['charged_power'] = b
                total['order_num'] = c
                total['traj_count'] = json.loads(d)
                total = json.dumps(total)

                return total
        if filetype == 'simresult':
            if 'car_infos' in filename:
                traj = pd.read_csv('./Files/SimUrbanResult/'+filename)
                traj = traj[(traj['lon'] > -180) & (traj['lon'] < 180)
                            & (traj['lat'] > -90) & (traj['lat'] < 90)]
                traj_count = traj.groupby(
                    ['lon', 'lat']).size().reset_index(name='count')


                params = tbd.area_to_params([traj_count['lon'].iloc[0], traj_count['lat'].iloc[0],
                                            traj_count['lon'].iloc[0], traj_count['lat'].iloc[0]], 1000)
                traj_count['LONCOL'], traj_count['LATCOL'] = tbd.GPS_to_grid(
                    traj_count['lon'], traj_count['lat'], params)
                traj_count = traj_count.groupby(
                    ['LONCOL', 'LATCOL']).size().reset_index(name='count')
                traj_count['geometry'] = tbd.grid_to_polygon(
                    [traj_count['LONCOL'], traj_count['LATCOL']], params)
                traj_count = gpd.GeoDataFrame(traj_count, geometry='geometry')
                traj_count['maxcount'] = traj_count['count'].quantile(0.9995)

                filename = filename.replace('car_infos','station_infos')

                station_vis = pd.read_csv(f'./Files/SimUrbanResult/{filename}')
                station_vis = station_vis[(station_vis['lon'] > -180) & (
                    station_vis['lon'] < 180) & (station_vis['lat'] > -90) & (station_vis['lat'] < 90)]

                station_vis['weekday'] = pd.to_datetime(
                    station_vis['time']).dt.weekday
                station_vis['hour'] = pd.to_datetime(
                    station_vis['time']).dt.hour

                '''统计充电功率'''
                charged_power = station_vis.copy()

                charged_power['charged_power'] = charged_power['current_charge_speed']
                charged_power = charged_power[[
                    'weekday', 'hour', 'lon', 'lat', 'charged_power']]

                '''统计充电订单'''
                order_num = station_vis[[
                    'lon', 'lat', 'weekday', 'hour', 'time', 'current_car']]
                order_num['current_car'] = order_num['current_car'].str.replace(
                    '[', '').str.replace(']', '').str.split(',')
                order_num = order_num.explode('current_car')
                order_num = order_num.drop_duplicates(
                    subset=['lon', 'lat', 'current_car'], keep='first')
                order_num = order_num.groupby(['lon', 'lat', 'weekday', 'time', 'hour']).count(
                ).reset_index().drop(columns=['time']).rename(columns={'current_car': 'order_num'})

                '''转换json'''
                station_vis = station_vis.groupby(['station_id', 'lon', 'lat'])['current_car'].apply(lambda x: len(set(
                    x.sum().replace('[', ',').replace(']', ',').replace(',,', ',').split(',')[1:-1]))).rename('count').reset_index()
                station_vis['geometry'] = gpd.points_from_xy(
                    station_vis['lon'], station_vis['lat'])
                station_vis = gpd.GeoDataFrame(
                    station_vis, geometry='geometry')

                station_vis.drop(['lon', 'lat'], axis=1, inplace=True)
                station_vis['maxcount'] = station_vis['count'].quantile(0.9995)
                station_vis.crs = 'EPSG:4326'

                station_vis['geometry'] = station_vis.to_crs('EPSG:3857').buffer(
                    1500*station_vis['count']/station_vis['maxcount']).to_crs('EPSG:4326')
                station_vis['场站名称']=station_vis['station_id']
                print(station_vis)
                total={}
                a = station_vis.to_json()
                b = charged_power.to_dict('records')
                c = order_num.to_dict('records')
                d = traj_count.to_json()
                import json
                total['station_vis'] = json.loads(a)
                total['charged_power'] = b
                total['order_num'] = c
                total['traj_count'] = json.loads(d)
                total = json.dumps(total)

                return total
        if filetype == 'selectstation':
            selectstation = gpd.read_file('./Files/UrbanStationSelection/'+filename)
            selectstation['count'] = selectstation['demand']
            selectstation['maxcount'] = selectstation['count'].quantile(0.9995)
            returnresult = selectstation.to_json()

            return returnresult
        if filetype == 'areaclass':
            areaclass = gpd.read_file('./Files/AreaClassOutput/'+filename)
            areaclass['count'] = areaclass['class']
            areaclass['maxcount'] = areaclass['class'].max()
            returnresult = areaclass.to_json()

            return returnresult
        if filetype == 'gridscore':
            areaclass = gpd.read_file('./Files/GridScoreOutput/'+filename)
            returnresult = areaclass.to_json()

            return returnresult

@app.route('/GenerateTraj', methods=['GET'])
def GenerateTraj():
    if request.method == 'GET':

        import sys
        sys.path.append('Models')
        import Models.trajgene_sh.generate_traj as generate_traj

        keylocpath = request.args.get('keyloc')
        lifepatpath = request.args.get('lifepat')
        GENE_NUM = request.args.get('numtraj')
        GENE_NUM = int(GENE_NUM)
        GENE_DAYS = request.args.get('numdays')
        GENE_DAYS = int(GENE_DAYS)
        taskname = request.args.get('taskname')
        taskname = f'{taskname}_{keylocpath}_{GENE_NUM}Traj_{GENE_DAYS}Days'

        with open(f'./Files/Trajectory/{taskname}.generating', 'w') as f:
            f.write('This is a temp file')

        # -------------此处为生成轨迹的代码---------------

        import sys
        sys.path.append('Models')
        import Models.trajgene_sh.generate_traj as generate_traj
        generate_traj.generate_traj(
            Trained_Model_Path='./Models/trajgene_sh/Life_pattern_format_and_GAN_model/wgan-gp-netG_epoch_49999_tuned_sh_202311.pth',
            Trained_Model_Parameter=[100, 256, 3034],
            Life_Pattern_Format_Path='./Models/trajgene_sh/Life_pattern_format_and_GAN_model/5000_lp_format.pkl',
            Index2node_dict_Path='./Models/trajgene_sh/Life_pattern_format_and_GAN_model/index2node.pkl',
            Node2index_dict_Path='./Models/trajgene_sh/Life_pattern_format_and_GAN_model/node2index.pkl',
            KeyPoint_PobTable_PATH=f'./Files/Keylocation/{keylocpath}',
            GENE_NUM=GENE_NUM,
            GENE_DAYS=GENE_DAYS,
            taskname=taskname,
            SAVE_PATH='./Files/Trajectory/')

        # --------------------------------------------
        # 保存文件

        returndata = {
            'status': 'success',
            'message': '轨迹生成成功'
        }

        # 删除生成中的文件
        os.remove(
            f'./Files/Trajectory/{taskname}.generating')

        return json.dumps(returndata)


@app.route('/SimUrban', methods=['GET'])
def SimUrban():
    if request.method == 'GET':

        traj_path = request.args.get('traj')
        charge_station_path = request.args.get('station')
        simulated_vehicle_nums = request.args.get('numtraj')
        useroad = request.args.get('useroad')
        road_path = request.args.get('road')
        numdays = request.args.get('numdays')
        numdays = int(numdays)
        config = request.args.get('config')
        taskname = request.args.get('taskname')

        if useroad == 'false':
            useroad = False
        else:
            useroad = True
        road_path = f'./Files/SimUrbanRoad/{road_path}'
        traj_path = f'./Files/Trajectory/{traj_path}'
        charge_station_path = f'./Files/UrbanStation/{charge_station_path}'
        config_path = f'./Files/SimUrbanConfig/{config}'
        simulated_vehicle_nums = int(simulated_vehicle_nums)

        # 生成文件名
        with open(f'./Files/SimUrbanResult/{taskname}_car_infos.generating', 'w') as f:
            f.write('This is a temp file')

        # 生成文件名
        with open(f'./Files/SimUrbanResult/{taskname}_station_infos.generating', 'w') as f:
            f.write('This is a temp file')

        # -------------此处为仿真的代码---------------
        from Models.urban_abm.main import simulation_urban
        try:
            simulation_urban(traj_path=traj_path,
                             charge_station_path=charge_station_path,
                             config_path=config_path,
                             useroad=useroad,
                             road_path=road_path,
                             simulated_vehicle_nums=simulated_vehicle_nums,
                             numdays=numdays,
                             version=taskname)
            returndata = {
                'status': 'success'
            }
        except:
            returndata = {
                'status': 'error'
            }
        # --------------------------------------------

        # 删除生成中的文件
        os.remove(f'./Files/SimUrbanResult/{taskname}_car_infos.generating')
        os.remove(
            f'./Files/SimUrbanResult/{taskname}_station_infos.generating')

        return json.dumps(returndata)


@app.route('/SimHighway', methods=['GET'])
def SimHighway():
    if request.method == 'GET':
        province = request.args.get('province')
        car_config = request.args.get('SimHighwayEVinfo')
        od_cnt = request.args.get('numtraj')
        od_cnt = int(od_cnt)
        taskname = request.args.get('taskname')

        print(province, car_config, od_cnt, taskname)
        # 生成文件名
        with open(f'./Files/SimHighwayResult/car_infos_{taskname}.generating', 'w') as f:
            f.write('This is a temp file')

        # -------------此处为仿真的代码---------------

        '''
        from Models.highway_sim.simulate import od_simulate
        od_simulate(
            province = province,
            od_cnt = od_cnt,
            car_config = car_config,
            output_path='./Files/SimHighwayResult/',
            taskname=taskname)
        '''
        # --------------------------------------------

        returndata = {
            'status': 'success',
            'message': '仿真结束'
        }

        # 删除生成中的文件
        os.remove(f'./Files/SimHighwayResult/car_infos_{taskname}.generating')

        return json.dumps(returndata)


@app.route('/SiteGenetic', methods=['GET'])
def SiteGenetic():
    if request.method == 'GET':
        max_sites = int(request.args.get('numsite'))
        selectedSim = request.args.get('selectedSim')
        taskname = request.args.get('taskname')
        car_infos_path = r'./Files/SimUrbanResult/'+selectedSim
        station_info_path = car_infos_path.replace('car_infos', 'station_infos')

        from Models.station_selection.main import site_genetic


        taz_path = r"./Models/station_selection/taz.geojson"
        poi_path = r"./Models/station_selection/gd_310000_poi.xlsx"
        price_path = r"./Models/station_selection/上海房价.csv"
        gridfile_path = r"./Models/station_selection/gridsum.csv"
        gridgejson_path = r"./Models/station_selection/grid.geojson"


        # 生成文件名
        with open(f'./Files/UrbanStationSelection/{taskname}.generating', 'w') as f:
            f.write('This is a temp file')
        try:
            result = site_genetic(station_info_path, taz_path, car_infos_path, poi_path, price_path, gridfile_path, gridgejson_path,max_sites=max_sites)

            result.to_file(f'./Files/UrbanStationSelection/{taskname}.json', driver='GeoJSON')
            returndata = {
                'status': 'success',
                'message': '站点选择成功'
            }
        except:
            returndata = {
                'status': 'error',
                'message': '站点选择失败'
            }

        # 删除生成中的文件
        os.remove(f'./Files/UrbanStationSelection/{taskname}.generating')

        return json.dumps(returndata)


@app.route('/AreaClass', methods=['GET'])
def AreaClass():
    if request.method == 'GET':
        bins = int(request.args.get('bins'))
        filename = request.args.get('filename')
        select_city = request.args.get('selectCity')
        bounds = [113.00558,22.510,114.0100,23.90000]

        taskname = request.args.get('taskname')
        taskname = taskname+'_'+select_city+'_'+str(bins)

        filename = r'./Files/AreaClassInput/'+filename

        from Models.station_selection.main import area_classfication
        # 生成文件名
        with open(f'./Files/AreaClassOutput/{taskname}.generating', 'w') as f:
            f.write('This is a temp file')
        try:
            result = area_classfication(filename, select_city, bounds, bins)
            result['class'] = result['class'].astype(int)
            result.to_file(f'./Files/AreaClassOutput/{taskname}.json', driver='GeoJSON')
            returndata = {
                'status': 'success',
                'message': '分级成功'
            }
        except:
            returndata = {
                'status': 'error',
                'message': '分级失败'
            }

        # 删除生成中的文件
        os.remove(f'./Files/AreaClassOutput/{taskname}.generating')

        return json.dumps(returndata)

@app.route('/GridScore', methods=['GET'])
def GridScore():
    if request.method == 'GET':
        selectedAreaTAZ = request.args.get('selectedAreaTAZ')
        selectedStationInfo = request.args.get('selectedStationInfo')
        selectedCarInfo = request.args.get('selectedCarInfo')
        selectedPOI = request.args.get('selectedPOI')
        selectedHousePrice = request.args.get('selectedHousePrice')
        selectedStationFile = request.args.get('selectedStationFile')
        taskname = request.args.get('taskname')

        area = gpd.read_file(r"./Files/GridScoreInput/"+selectedAreaTAZ)
        station_info_path = r'./Files/SimUrbanResult/'+selectedStationInfo
        car_info_path = r'./Files/SimUrbanResult/'+selectedCarInfo
        poi_file_path = r"./Files/GridScoreInput/"+selectedPOI
        price_file=r"./Files/GridScoreInput/"+selectedHousePrice
        station_file_path = r"./Files/GridScoreInput/"+selectedStationFile


        from Models.station_selection.main import grid_score
        # 生成文件名
        with open(f'./Files/GridScoreOutput/{taskname}.generating', 'w') as f:
            f.write('This is a temp file')
        try:
            result=grid_score(area,station_info_path,car_info_path,poi_file_path,price_file,station_file_path)
            result.to_file(f'./Files/GridScoreOutput/{taskname}.json', driver='GeoJSON')
            returndata = {
                'status': 'success',
                'message': '评分成功'
            }
        except:
            returndata = {
                'status': 'error',
                'message': '评分失败'
            }

        # 删除生成中的文件
        os.remove(f'./Files/GridScoreOutput/{taskname}.generating')

        return json.dumps(returndata)
    

@app.route('/KeylocFromPOI', methods=['GET'])
def KeylocFromPOI():
    if request.method == 'GET':
        poipath = request.args.get('poi')
        gridsize = int(request.args.get('gridsize'))
        taskname = request.args.get('taskname')

        import os
        os.mkdir(f'./Files/Keylocation/{taskname}.generating')

        poi = pd.read_csv(f'./Files/POI/{poipath}')
        if ('tag' not in poi.columns)|('lon' not in poi.columns)|('lat' not in poi.columns):
            returndata = {
                'status': 'error',
                'message': '错误，POI数据需要有lon,lat,tag列'
            }
        else:
            poi['tag'] = poi['tag'].str.split(';').str.get(0)

            poitype = pd.read_csv('./Files/POI/POI分类')
            poi = pd.merge(poi.drop('type',axis=1),poitype)

            from Models.hwo_predict.main import keyloc_from_poi
            h0_pob_dict,h2w_pob_dict,hw2o_pob_dict,params = keyloc_from_poi(poi,gridsize)

            if len(h0_pob_dict) ==0 :
                returndata = {
                    'status': 'error',
                    'message': '错误，POI数据规模过小'
                }
            else:
                import pickle
                import os
                if not os.path.exists(f'./Files/Keylocation/{taskname}'):
                    os.mkdir(f'./Files/Keylocation/{taskname}')
                if not os.path.exists(f'./Files/Keylocation/{taskname}/h0'):
                    os.mkdir(f'./Files/Keylocation/{taskname}/h0')
                if not os.path.exists(f'./Files/Keylocation/{taskname}/h2w'):
                    os.mkdir(f'./Files/Keylocation/{taskname}/h2w')
                if not os.path.exists(f'./Files/Keylocation/{taskname}/Others'):
                    os.mkdir(f'./Files/Keylocation/{taskname}/Others')

                with open(f'./Files/Keylocation/{taskname}/h0/h0_pob_dict.pkl','wb') as f:
                    pickle.dump(h0_pob_dict,f)
                with open(f'./Files/Keylocation/{taskname}/h2w/h2w_pob_dict.pkl','wb') as f:
                    pickle.dump(h2w_pob_dict,f)
                with open(f'./Files/Keylocation/{taskname}/Others/hw2o_pob_dict.pkl','wb') as f:
                    pickle.dump(hw2o_pob_dict,f)
                import json
                json.dump(params,open(f'./Files/Keylocation/{taskname}/tbdParams.json','w'))

                returndata = {
                    'status': 'success',
                    'message': '生成成功'
                }


        # 删除生成中的文件
        os.rmdir(f'./Files/Keylocation/{taskname}.generating')

        return json.dumps(returndata)
    
if __name__ == '__main__':
    app.run(debug=True, port=8001)
