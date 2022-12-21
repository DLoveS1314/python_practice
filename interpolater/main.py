# %% [markdown]
# 导入需要的环境

# %%

from pathlib import Path
import os
import sys
os.environ['PROJ_LIB'] = '/home/dls/anaconda3/envs/pytorch/share/proj'
# ERROR 1: PROJ: proj_create_from_database: Open of /home/dls/anaconda3/envs/pytorch/share/proj failed 不然会报错
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dggrid4py import DGGRIDv7, Dggs, dgselect, dggs_types
##测试正则表达式写法 
import re 
import xarray as xr
 
def process_nei_chi(gdf):
    '''由于生成的是字符串数组 需要拆开 处理生成的child和nei s'''

    neighbors_name = ['u','l','d','r']#up left down right
    children_name = ['0','1','2','3']#up left down right
    gd_nei =gdf['neighbors'].str.replace('[\[\]\"]','').str.split(',',expand=True )#如果正则中没有\" 那么生成的数字是 ' "2048" '这种 就是 字符串中包含的字符串 所以需要吧" 也给消除
    gd_nei.rename(columns={x:neighbors_name[x] for x in gd_nei.columns},inplace=True) 
    # print(gd_nei)


    gd_child =gdf['children'].str.replace('[\[\]\"]','').str.split(',',expand=True ) 
    gd_child.rename(columns={x:children_name[x] for x in gd_child.columns},inplace=True) 
    # print(gd_child)
    gdf =gdf.drop(columns={'neighbors','children'}).rename(columns={'name':'seqnum'})
    # gdf.seqnum=gdf.seqnum.astype(int)
    gdf =pd.concat([gdf,gd_nei,gd_child],axis=1)
    
    return gdf.astype(int)
 
def example_read_geojson(dggs_type='ISEA4T',resolution = 6):
    """
    GDALcollection生成输出 包含了边临近编码 子格网编码 自身编码 中心点经纬度 cell边界经纬度
    Args:
        path (_type_):geojson路径
    """    
    dggridPath= '/home/dls/data/openmmlab/DGGRID/build/src/apps/dggrid/dggrid'
    working_dir = '/home/dls/data/openmmlab/python_practice/interpolater/tmp/grids'
    dggrid = DGGRIDv7(executable= dggridPath, working_dir=working_dir, capture_logs=True, silent=False)
    dggs_type='ISEA4D'
    resolution = 6
    gdf_cell_point= dggrid.gen_cell_point( dggs_type=dggs_type,resolution = resolution,if_drop=False)
    # print(gdf_cell_point['seqnum'])
    df_nei_chi= dggrid.gen_nei_chi( dggs_type=dggs_type,resolution = resolution)
    df_nei_chi=process_nei_chi(df_nei_chi)
    # print(df_nei_chi['seqnum'])
    gdf = gdf_cell_point.merge(df_nei_chi,on='seqnum',how ='inner')
    gdf.attrs = {'level':resolution,'dggs_type':dggs_type}
    # gdf_cell = dggrid.gen_cell( dggs_type=dggs_type,resolution = resolution)
    # gdf = gdf_cell_point.join(df_nei_chi,on='seqnum',how ='inner')
    # gdf = gdf_point
    return gdf
def cal_angle_nei(gdf):
    neighbors_name = ['u','lu','l','ld','d','rd','r','ru']#up left down right
    children_name = ['0','1','2','3']#up left down right
    # 根据边临近计算角临近
    lftList=(gdf.l-1).tolist()
    rigList=(gdf.r-1).tolist()

    gdf['lu'] = gdf.u[lftList].reset_index(drop=True)  #对应的行索引不减1代表的是seqnum lu
    gdf['ld'] = gdf.d[lftList].reset_index(drop=True) #对应的行索引不减1代表的是seqnum lu
    gdf['ru'] = gdf.u[rigList].reset_index(drop=True) #对应的行索引不减1代表的是seqnum lu
    gdf['rd'] = gdf.d[rigList].reset_index(drop=True) #对应的行索引不减1代表的是seqnum lu
    # col=list(gdf.columns)
    # col.reverse()
    gdf=gdf[['seqnum']+neighbors_name+children_name+['cell','point']]
    return gdf
gdf = example_read_geojson()
gdf = cal_angle_nei(gdf)
# ##保存数据
# compression ='bz2' #这个压缩率和读取率 keep blance https://zhuanlan.zhihu.com/p/115642111
# savpath='dggsType_{}_level_{}_com_{}.pkl'.format(gdf.attrs['dggs_type'],gdf.attrs['level'],compression)
# gdf.to_pickle(savpath,compression)

# # %%
# ##读取数据
# pd.read_pickle(savpath,compression=compression)

# # %%
# #加载数据
ncvar = '/home/dls/data/climatenet/train/data-1996-09-26-01-1_1.nc'
data=xr.load_dataset(ncvar).squeeze(dim='time',drop=True)
labels= data.LABELS 

data=data.drop_vars('LABELS')
lat = gdf.point.y
lon = gdf.point.x
print('begin interdata')
# interdata = data.interp(lat=lat,lon=lon,method='linear' )
for var in data:
    # print(data[var])
    # interdata = data[var].interp(lat=lat,lon=lon,method='linear' )
    print(var)

# interlabel = labels.interp(lat=lat,lon=lon,method='nearest')