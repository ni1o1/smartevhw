import pandas as pd
import transbigdata as tbd

def keyloc_from_heatmap(heatmap,gridsize):
    '''
    生成核心地点的概率分布，其中输入的heatmap需要包含以下字段：
    lon: 经度
    lat: 纬度
    hour: 小时
    day: 日期 (可选)，如果提供则会根据日期判断是否为工作日
    weekday: 星期几 (可选)，如果没有，且 day 也没有提供，则默认为工作日
    count: 计数 (可选)，如果没有，则默认为1

    '''
    if 'weekday' in heatmap.columns:
        heatmap['workday']=heatmap['weekday'].apply(lambda x:1 if x<5 else 0)
    else:
        if 'day' in heatmap.columns:
            heatmap['workday']=pd.to_datetime(heatmap['day']).dt.weekday.apply(lambda x:1 if x<5 else 0)
        else:
            heatmap['workday']=1
    if 'count' not in heatmap.columns:
        heatmap['count'] = 1
    heatmap = heatmap.groupby(['lon','lat','hour','workday'])['count'].sum().reset_index()
    params = tbd.area_to_params([
        heatmap['lon'].mean(),
        heatmap['lat'].mean(),
        heatmap['lon'].mean(),
        heatmap['lat'].mean()], accuracy=gridsize)
    heatmap['LONCOL'], heatmap['LATCOL'] = tbd.GPS_to_grid(heatmap['lon'], heatmap['lat'], params)
    # 核心居住时间
    H_prob = heatmap[heatmap['hour']<6].groupby(['LONCOL','LATCOL'])['count'].sum().rename('count').reset_index()
    H_prob['grid'] = H_prob['LONCOL'].astype(str)+','+H_prob['LATCOL'].astype(str)
    H_prob = H_prob[['grid','count']]

    # 核心工作时间
    W_prob = heatmap[(((heatmap['hour']>=9)&(heatmap['hour']<11))|((heatmap['hour']>=14)&(heatmap['hour']<16)))&(heatmap['workday']==1)].groupby(['LONCOL','LATCOL'])['count'].sum().rename('count').reset_index()
    W_prob['grid'] = W_prob['LONCOL'].astype(str)+','+W_prob['LATCOL'].astype(str)
    W_prob = W_prob[['grid','count']]

    # 其他时间
    O_prob = heatmap[((heatmap['hour']>=18)&(heatmap['hour']<20)&(heatmap['workday']==1))|((heatmap['hour']>=10)&(heatmap['hour']<20)&(heatmap['workday']==0))].groupby(['LONCOL','LATCOL'])['count'].sum().rename('count').reset_index()
    O_prob['grid'] = O_prob['LONCOL'].astype(str)+','+O_prob['LATCOL'].astype(str)
    O_prob = O_prob[['grid','count']]

    o = heatmap[['LONCOL','LATCOL']].drop_duplicates()
    o['flag'] = 1
    d = heatmap[['LONCOL','LATCOL']].drop_duplicates()
    d['flag'] = 1
    od = pd.merge(o, d, on='flag').drop('flag', axis=1)
    od['reachtime'] = ((((od['LONCOL_x']-od['LONCOL_y'])**2+(od['LATCOL_x']-od['LATCOL_y'])**2)**0.5*1.5*gridsize*60)/20000+10).astype(int)
    od['o'] = od['LATCOL_x'].astype(str)+','+od['LONCOL_x'].astype(str)
    od['d'] = od['LATCOL_y'].astype(str)+','+od['LONCOL_y'].astype(str)
    rij = od[['o','d','reachtime']].copy()


    # 计共同分布
    from Models.hwo_predict.keyloc_generate import heatmap_to_jointprob

    #通过辐射模型估算Home Work Other
    W_prob,H_prob,O_prob,HW_prob,HO_prob,WO_prob = heatmap_to_jointprob(W_prob,H_prob,O_prob,rij)


    #整理home的概率分布
    H_prob['prob'] = H_prob['count']/H_prob['count'].sum()
    H_prob['grid'] = H_prob['grid'].str.replace(',','|')
    h0_pob_dict = H_prob[['grid','prob']].set_index('grid').to_dict()['prob']


    #整理home到work的概率分布
    HW_prob['hgrid'] = HW_prob['hgrid'].str.replace(',','|')
    HW_prob['wgrid'] = HW_prob['wgrid'].str.replace(',','|')
    def getwprob(x):
        x['prob'] = x['T_ij']/x['T_ij'].sum()
        return list(x['wgrid']),list(x['prob'])

    # 有些地方没有工作地点，所以要补充
    H_grid = H_prob[['grid']].copy()
    H_grid.columns = ['hgrid']
    HW_prob = pd.merge(HW_prob,H_grid,on='hgrid',how='right')
    HW_prob.loc[HW_prob['wgrid'].isnull(),'T_ij'] = 1
    HW_prob.loc[HW_prob['wgrid'].isnull(),'wgrid'] = HW_prob.loc[HW_prob['wgrid'].isnull(),'hgrid'] 

    h2w_pob_dict = HW_prob.groupby(['hgrid']).apply(lambda x:getwprob(x)).to_dict()

    #整理home到other的概率分布
    HO_prob['hgrid'] = HO_prob['hgrid'].str.replace(',','|')
    HO_prob['ogrid'] = HO_prob['ogrid'].str.replace(',','|')
    def getopro(x):
        x['prob'] = x['T_ij']/x['T_ij'].sum()
        return list(x['ogrid']),list(x['prob'])

    # 有些地方没有工作地点，所以要补充
    HO_prob = pd.merge(HO_prob,H_grid,on='hgrid',how='right')
    HO_prob.loc[HO_prob['ogrid'].isnull(),'T_ij'] = 1
    HO_prob.loc[HO_prob['ogrid'].isnull(),'wgrid'] = HO_prob.loc[HO_prob['ogrid'].isnull(),'hgrid'] 
    hw2o_pob_dict = HO_prob.groupby(['hgrid']).apply(lambda x:getopro(x)).to_dict()

    return h0_pob_dict,h2w_pob_dict,hw2o_pob_dict,params

def keyloc_from_poi(poi,gridsize):

    params = tbd.area_to_params([
        poi['lon'].mean(),
        poi['lat'].mean(),
        poi['lon'].mean(),
        poi['lat'].mean()], accuracy=gridsize)
    poi['LONCOL'], poi['LATCOL'] = tbd.GPS_to_grid(poi['lon'], poi['lat'], params)


    H_prob = poi[poi['type']=='Residential'].groupby(['LONCOL','LATCOL'])['type'].count().rename('count').reset_index()
    H_prob['grid'] = H_prob['LONCOL'].astype(str)+','+H_prob['LATCOL'].astype(str)
    H_prob = H_prob[['grid','count']]

    W_prob = poi[poi['type'].isin(['Workplace','Government','Education','Hospital'])].groupby(['LONCOL','LATCOL'])['type'].count().rename('count').reset_index()
    W_prob['grid'] = W_prob['LONCOL'].astype(str)+','+W_prob['LATCOL'].astype(str)
    W_prob = W_prob[['grid','count']]

    O_prob = poi[poi['type'].isin(['Others','Shopping','Tourist','Sport'])].groupby(['LONCOL','LATCOL'])['type'].count().rename('count').reset_index()
    O_prob['grid'] = O_prob['LONCOL'].astype(str)+','+O_prob['LATCOL'].astype(str)
    O_prob = O_prob[['grid','count']]


    o = poi[['LONCOL','LATCOL']].drop_duplicates()
    o['flag'] = 1
    d = poi[['LONCOL','LATCOL']].drop_duplicates()
    d['flag'] = 1
    od = pd.merge(o, d, on='flag').drop('flag', axis=1)
    od['reachtime'] = ((((od['LONCOL_x']-od['LONCOL_y'])**2+(od['LATCOL_x']-od['LATCOL_y'])**2)**0.5*1.5*gridsize*60)/20000+10).astype(int)
    od['o'] = od['LATCOL_x'].astype(str)+','+od['LONCOL_x'].astype(str)
    od['d'] = od['LATCOL_y'].astype(str)+','+od['LONCOL_y'].astype(str)
    rij = od[['o','d','reachtime']].copy()


    # 计共同分布
    from Models.hwo_predict.keyloc_generate import heatmap_to_jointprob

    #通过辐射模型估算Home Work Other
    W_prob,H_prob,O_prob,HW_prob,HO_prob,WO_prob = heatmap_to_jointprob(W_prob,H_prob,O_prob,rij)


    #整理home的概率分布
    H_prob['prob'] = H_prob['count']/H_prob['count'].sum()
    H_prob['grid'] = H_prob['grid'].str.replace(',','|')
    h0_pob_dict = H_prob[['grid','prob']].set_index('grid').to_dict()['prob']


    #整理home到work的概率分布
    HW_prob['hgrid'] = HW_prob['hgrid'].str.replace(',','|')
    HW_prob['wgrid'] = HW_prob['wgrid'].str.replace(',','|')
    def getwprob(x):
        x['prob'] = x['T_ij']/x['T_ij'].sum()
        return list(x['wgrid']),list(x['prob'])

    # 有些地方没有工作地点，所以要补充
    H_grid = H_prob[['grid']].copy()
    H_grid.columns = ['hgrid']
    HW_prob = pd.merge(HW_prob,H_grid,on='hgrid',how='right')
    HW_prob.loc[HW_prob['wgrid'].isnull(),'T_ij'] = 1
    HW_prob.loc[HW_prob['wgrid'].isnull(),'wgrid'] = HW_prob.loc[HW_prob['wgrid'].isnull(),'hgrid'] 

    h2w_pob_dict = HW_prob.groupby(['hgrid']).apply(lambda x:getwprob(x)).to_dict()

    #整理home到other的概率分布
    HO_prob['hgrid'] = HO_prob['hgrid'].str.replace(',','|')
    HO_prob['ogrid'] = HO_prob['ogrid'].str.replace(',','|')
    def getopro(x):
        x['prob'] = x['T_ij']/x['T_ij'].sum()
        return list(x['ogrid']),list(x['prob'])

    # 有些地方没有工作地点，所以要补充
    HO_prob = pd.merge(HO_prob,H_grid,on='hgrid',how='right')
    HO_prob.loc[HO_prob['ogrid'].isnull(),'T_ij'] = 1
    HO_prob.loc[HO_prob['ogrid'].isnull(),'wgrid'] = HO_prob.loc[HO_prob['ogrid'].isnull(),'hgrid'] 
    hw2o_pob_dict = HO_prob.groupby(['hgrid']).apply(lambda x:getopro(x)).to_dict()

    return h0_pob_dict,h2w_pob_dict,hw2o_pob_dict,params