import os
import arcpy
from arcpy.sa import *

import numpy as np
import pandas as pd
from simpledbf import Dbf5
import simpledbf
import matplotlib.pyplot as plt
import seaborn as sns

def set_arcpy_env(xsectshp):
    # Define projection
    arcpy.env.overwriteOutput = True
    dsc = arcpy.Describe(xsectshp)
    coord_sys = dsc.spatialReference
    # arcpy.DefineProjection_management(terrain, coord_sys)

def set_params(case_name):

    unit_BM = 'meter'  # meter
    mFactor_BM = 1

    if case_name == 'SFE_Leggett':
        reach_length_m = 1250  # in meter
        thal_interval = '10m'
    elif case_name == 'SFE_Miranda':
        reach_length_m = 1500  # in meter
        thal_interval = '10m'
    elif case_name == 'SFE_Scotia':
        reach_length_m = 5150  # in meter
        thal_interval = '15m'

    return unit_BM, mFactor_BM, reach_length_m, thal_interval
def set_path(case_name, res):

    drive = os.path.abspath('./')[:2] # e.g., D:, E:

    path_main = drive + '/CIROH/HAND-FIM_Assessment_public/codes/' + case_name + '_hand_param_calc/'
    path_BM = drive + '/CIROH/HAND-FIM_Assessment_public/codes/' + case_name + '_hand_param_calc/HAND_BM/'
    path_HAND = drive + '/CIROH/HAND-FIM_Assessment_public/codes/' + case_name + '_hand_param_calc/HAND_' + res + '/'
    path_tuflow = path_BM + 'tuflow/'

    if res == '0':
        path_HAND = ''

    return path_main, path_BM, path_HAND, path_tuflow

def set_path_HAND(case_name):

    path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, '0') ## path_BM

    path_GIS = path_BM + 'gis/'

    os.chdir(path_BM)

    path_HAND_1m = '../HAND_1m/'
    path_HAND_10m = '../HAND_10m/'

    return path_GIS, path_HAND_1m, path_HAND_10m

def set_path_shp(case_name, thal_interval):

    path_GIS, path_HAND_1m, path_HAND_10m = set_path_HAND(case_name)

    # path to station lines shp file
    xsectshp = path_GIS + 'Thalweg_' + thal_interval + '_adjusted.shp'
    # path to station points shp file
    pointshp = path_GIS + 'Thalweg_' + thal_interval + '_points.shp'
    # Load DEM (Benchmark)
    terrain = path_GIS + case_name + '.asc'

    return xsectshp, pointshp, terrain

def define_mFactor(res):

    if res == '1m': # for LiDAR, unit is in feet
        unit = 'meter'
        mFactor = 1
    else:
        unit = 'meter'
        mFactor = 1

    return unit, mFactor

def read_channel_params(case_name):

    drive = os.path.abspath('./')[:2]  # e.g., D:, E:
    if case_name == 'SFE_Leggett':
        channel_param = pd.read_excel(drive + '/tuflow-SFE-mainstem/parameters_sfe-mainstem.xlsx')
    elif case_name == 'SFE_Miranda':
        channel_param = pd.read_excel(drive + '/tuflow-SFE-Miranda/parameters_sfe-Miranda.xlsx')
    elif case_name == 'SFE_Scotia':
        channel_param = pd.read_excel(drive + '/tuflow-SFE-Scotia/parameters_sfe-Scotia.xlsx')

    return channel_param

def read_slope_n(case_name):
    drive = os.path.abspath('./')[:2]  # e.g., D:, E:
    if case_name == 'SFE_Leggett':
        slope_NHD_1m = 0.0027  ##1. Slope 1m DEM 0.0025 Regression Line, 2. 0.0027 Rise/Run slope 1m DEM, 3. 0.00275 Rise/Run 10m DEM
        slope_NHD_10m = 0.00275
        # ManningN_NHD = 0.027 #0.06 NHD Line #0.05 Literature
        ManningN_NHD = 0.060
    elif case_name == 'SFE_Miranda':
        slope_NHD_1m = 0.0012  ###rRise/Run 1m DEM
        slope_NHD_10m = 0.0015  #####Rise/Run 10m DEM
        # Slope Regression Line 1m DEM 0.0013
        # ManningN_NHD = 0.020 #1D HEC RAS Calibrated Manning
        # Manning Literature = 0.05
        ManningN_NHD = 0.06
    elif case_name == 'SFE_Scotia':
        slope_NHD_1m = 0.001  ####0.0003 from Regression (0.00021 rise/run)
        slope_NHD_10m = 0.001  # (0.00021 rise/run)
        # ManningN_NHD = 0.02 #1D HEC RAS Calibrated Manning
        # Manning Literature = 0.05
        ManningN_NHD = 0.06

    return slope_NHD_1m, slope_NHD_10m, ManningN_NHD

def set_fig_params(num_med):

    y_scale = 'log'  # 'linear' 'log'
    save_to_excel = True

    ## Figure properties
    lw = 5
    pcolor = '#a6a5a7'
    pcolor = '#d0cfd1'  # Grey
    mcolor = 'seagreen'
    # pcolor = '#ADD8E6'

    if num_med > 1:
        path_fig = './med_' + str(num_med) + '/'
    else:
        path_fig = './'

    return path_fig, y_scale, save_to_excel, lw, pcolor, mcolor

def cal_elev_downstream(path_tuflow):

    elev_dbf = Dbf5(path_tuflow + 'table_downstream.dbf')
    elev_df = elev_dbf.to_dataframe()
    elev_downstream = min(elev_df['FIRST_Z'])
    elev_dbf.f.close()

    return elev_downstream

def flow_Qnum_stages(path_tuflow, case_name, elev_downstream):

    Qnum_stages, flow_stages = [], []
    for jj in range(11):
        zero1 = ''
        str_tmp = str(jj + 1)
        str_len = str_tmp.__len__()

        for ind in range(0, 3 - str_len):
            zero1 = zero1 + '0'
        Q_num = zero1 + str_tmp

        table_bc = pd.read_csv(path_tuflow +'bc_dbase/'+case_name+'_bc_data_'+Q_num+'.csv')

        flow_tmp = table_bc['RPin'][0]
        stage_tmp = table_bc['RPout'][0]-elev_downstream

        flow_stage_tmp = [flow_tmp, stage_tmp]
        Qnum_stage_tmp = [Q_num, stage_tmp.round(5)]

        flow_stages.append(flow_stage_tmp)
        Qnum_stages.append(Qnum_stage_tmp)

    return flow_stages, Qnum_stages

def convert_unit_BM_to_T(flow_stages, mFactor_BM, mFactor_T):

    flow_stages = pd.DataFrame(flow_stages, columns=['flow_BM', 'stage_BM'])
    flow_stages['flow'] = flow_stages['flow_BM'] * mFactor_BM ** 3 / mFactor_T ** 3
    flow_stages['stage'] = flow_stages['stage_BM'] * mFactor_BM / mFactor_T

    return flow_stages

def prepare_gis_files():

    #Call Layers
    HAND = './hand.tif'
    Boundary = './boundary.shp'
    DEM = './dem.tif'

    ## Output
    tmp_path = './tmp'
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    #else:
    #    os.rmdir(tmp_path)

    #Extract Masked HAND and Masked DEM
    arcpy.env.overwriteOutput = True
    arcpy.CheckOutExtension("Spatial")
    HAND_Mask = ExtractByMask(HAND, Boundary) #, "INSIDE", Boundary)
    HAND_Mask.save('./tmp/HAND_Mask.tif')
    DEM_Mask = ExtractByMask(DEM, Boundary) #, "INSIDE", Boundary)
    DEM_Mask.save('./tmp/DEM_Mask.tif')

    return HAND_Mask, DEM_Mask

def get_max_height(p):

    list = p.patches

    # Getting length of list
    i = 0
    h = []

    # Iterating using while loop
    while i < len(list):
        tmp = list[i].get_height()
        h = np.append(h, tmp)
        i += 1

    h_max = max(h)

    return h_max

def load_stages_df_extended(case_name, Stages_DF, extended_SRC):

    if case_name == 'SFE_Leggett':
        if extended_SRC == 1:
            ## Manning's n = 0.05
            Stages_DF.at[11, 'Stage'] = 2.6
            Stages_DF.at[12, 'Stage'] = 2.8
            Stages_DF.at[13, 'Stage'] = 3.0
            Stages_DF.at[14, 'Stage'] = 3.2
            Stages_DF.at[15, 'Stage'] = 3.4
            Stages_DF.at[16, 'Stage'] = 3.7
            Stages_DF.at[17, 'Stage'] = 4
            Stages_DF.at[18, 'Stage'] = 4.3
            Stages_DF.at[11, 'Code'] = '012'
            Stages_DF.at[12, 'Code'] = '013'
            Stages_DF.at[13, 'Code'] = '014'
            Stages_DF.at[14, 'Code'] = '015'
            Stages_DF.at[15, 'Code'] = '016'
            Stages_DF.at[16, 'Code'] = '017'
            Stages_DF.at[17, 'Code'] = '018'
            Stages_DF.at[18, 'Code'] = '019'

            ## Calibrated Manning's n
            # Stages_DF.at[11, 'Stage'] = 2.6
            # Stages_DF.at[12, 'Stage'] = 2.8
            # Stages_DF.at[13, 'Stage'] = 3.0
            # Stages_DF.at[11, 'Code'] = '012'
            # Stages_DF.at[12, 'Code'] = '013'
            # Stages_DF.at[13, 'Code'] = '014'
    elif case_name == 'SFE_Miranda':
        if extended_SRC == 1:
            Stages_DF.at[11, 'Stage'] = 4.5
            Stages_DF.at[12, 'Stage'] = 5
            Stages_DF.at[13, 'Stage'] = 5.5
            Stages_DF.at[14, 'Stage'] = 6
            Stages_DF.at[15, 'Stage'] = 6.5
            Stages_DF.at[16, 'Stage'] = 7
            Stages_DF.at[17, 'Stage'] = 7.5
            Stages_DF.at[11, 'Code'] = '012'
            Stages_DF.at[12, 'Code'] = '013'
            Stages_DF.at[13, 'Code'] = '014'
            Stages_DF.at[14, 'Code'] = '015'
            Stages_DF.at[15, 'Code'] = '016'
            Stages_DF.at[16, 'Code'] = '017'
            Stages_DF.at[17, 'Code'] = '018'

            # With Calibrated Manning's n
            # Stages_DF.at[11, 'Stage'] = 4.3
            # Stages_DF.at[12, 'Stage'] = 4.6
            # Stages_DF.at[11, 'Code'] = '012'
            # Stages_DF.at[12, 'Code'] = '013'
    elif case_name == 'SFE_Scotia':
        if extended_SRC == 1:
            # With n = 0.05
            Stages_DF.at[11, 'Stage'] = 6
            Stages_DF.at[12, 'Stage'] = 6.5
            Stages_DF.at[13, 'Stage'] = 7
            Stages_DF.at[14, 'Stage'] = 7.5
            Stages_DF.at[15, 'Stage'] = 8
            Stages_DF.at[16, 'Stage'] = 8.5
            Stages_DF.at[17, 'Stage'] = 9
            Stages_DF.at[18, 'Stage'] = 9.5
            Stages_DF.at[19, 'Stage'] = 10
            Stages_DF.at[20, 'Stage'] = 10.5
            Stages_DF.at[11, 'Code'] = '012'
            Stages_DF.at[12, 'Code'] = '013'
            Stages_DF.at[13, 'Code'] = '014'
            Stages_DF.at[14, 'Code'] = '015'
            Stages_DF.at[15, 'Code'] = '016'
            Stages_DF.at[16, 'Code'] = '017'
            Stages_DF.at[17, 'Code'] = '018'
            Stages_DF.at[18, 'Code'] = '019'
            Stages_DF.at[19, 'Code'] = '020'
            Stages_DF.at[20, 'Code'] = '021'

            ## With Calibrated Mannig's n
            # Stages_DF.at[11, 'Stage'] = 6
            # Stages_DF.at[11, 'Code'] = '012'

    return Stages_DF

def calc_HAND_params(Stages_DF, HAND_Mask, DEM_Mask, path_HAND, reach_length_m, mFactor, unit, extended_SRC):

    HAND_params = []

    for ii in range(0, Stages_DF['Stage'].__len__()):
        ##########################################Output 1: Flow_Area##################################################################################
        # Flow Area Raster
        # Extracting Inundated pixels and assigning a value of 1
        stage_str = Stages_DF['Code'][ii]
        print(stage_str)

        Flow_Area_Raster = Con(Raster(HAND_Mask) <= Stages_DF.at[ii,'Stage'], 1, 0)

        Flow_Area_Raster_path = './tmp/Flow_Area_Raster_'+stage_str+'.tif'
        Flow_Area_Raster.save(Flow_Area_Raster_path)

        #Counting pixels with Value of 1
        Dict_FAR = {row[0]:row[1] for row in arcpy.da.SearchCursor(Flow_Area_Raster_path, ['Value','Count'])}
        DF_FAR = pd.DataFrame([Dict_FAR])
        desc_FAR = arcpy.Describe(Flow_Area_Raster_path)

        #Extracting Individual Pixel Area
        x_cell = desc_FAR.meanCellWidth
        y_cell = desc_FAR.meanCellHeight
        Pixel_Area = x_cell*y_cell

        #Calculating the Flooded Area
        Flow_Area_Value = Pixel_Area*DF_FAR.iat[0,1]

        #######################################Output 2: Wetted Bed Area###################################################################################
        #Extracting Raster values of 1 only and setting to null the rest
        Flow_Area_Raster_1V = SetNull(Flow_Area_Raster_path, Flow_Area_Raster_path,"VALUE < 1")
        Flow_Area_Raster_1V.save('./tmp/Flow_Area_Raster_1V.tif')

        #Converting Raster to Polygon
        Flow_Area_Polygon_path = './tmp/Flow_Area_Polygon.shp'
        Flow_Area_Polygon = arcpy.conversion.RasterToPolygon(Flow_Area_Raster_1V,
                                                             Flow_Area_Polygon_path, "NO_SIMPLIFY", "VALUE")



        #Generating slope raster from Masked DEM
        Slope_raster_path = './tmp/Slope_raster.tif'
        Slope_Raster = arcpy.ddd.Slope(DEM_Mask, Slope_raster_path, "PERCENT_RISE", 1)

        #Calculating SQRTP1 Raster
        SQRTP1 =RasterCalculator([Slope_raster_path],["Slope_raster_path"],"(((Slope_raster_path/100)**2)+1)**0.5")
        SQRTP1_path = './tmp/SQRTP1.tif'
        SQRTP1.save(SQRTP1_path)


        #Getting the Average SQRTP1 Based on Polygon
        MEAN_sqrtp1_path = './tmp/Mean_sqrtp_1_table.dbf'
        MEAN_sqrtp1 = ZonalStatisticsAsTable(Flow_Area_Polygon, "FID",
                                                 SQRTP1, MEAN_sqrtp1_path,
                                                 "DATA","MEAN")
        MEAM_sqrtp1_dbf5 = Dbf5('./tmp/Mean_sqrtp_1_table.dbf')

        DF_MEAN_sqrtp1 = MEAM_sqrtp1_dbf5.to_dataframe()
        MEAN_VALUE_sqrtp1 = DF_MEAN_sqrtp1["MEAN"].mean()
        MEAM_sqrtp1_dbf5.f.close()


        #Wetted Area
        Wetted_Bed_Area_M2 = Flow_Area_Value * MEAN_VALUE_sqrtp1

        # Water Volume

        Statement = str(Stages_DF.at[ii,'Stage']) + "-(HAND_Mask / Con(Flow_Area_Raster_1V, 1, 0))"
        Water_depth = RasterCalculator([HAND_Mask, Flow_Area_Raster_1V],
                                       ["HAND_Mask", "Flow_Area_Raster_1V"], Statement)
        Water_depth_path = './tmp/Water_depth_table.dbf'
        MEAN_water_depth = ZonalStatisticsAsTable(Flow_Area_Polygon, "FID",
                                                  Water_depth, Water_depth_path,
                                                  "NODATA","MEAN")
        MEAN_water_depth_dbf5 = Dbf5(Water_depth_path)

        DF_MEAN_water_depth = MEAN_water_depth_dbf5.to_dataframe()
        MEAN_VALUE_water_depth = sum((DF_MEAN_water_depth["AREA"]*DF_MEAN_water_depth["MEAN"])/DF_MEAN_water_depth["AREA"].sum())
        MEAN_water_depth_dbf5.f.close()

        MEAN_VALUE_water_depth
        Water_Volume = Wetted_Bed_Area_M2 * MEAN_VALUE_water_depth

        reach_length = reach_length_m / mFactor # meter to feet
        # XS area
        XS_area = Water_Volume / reach_length

        # Perimeter
        Perimeter = Wetted_Bed_Area_M2 / reach_length

        # Hydraulic_radius
        Hydraulic_radius = XS_area / Perimeter

        # Width
        Width = Flow_Area_Value / reach_length

        ##############################################Summary#############################################################################################
        HAND_params_tmp = [Stages_DF.at[ii, 'Stage'], Flow_Area_Value, Wetted_Bed_Area_M2, XS_area, Perimeter, Hydraulic_radius, Width]
        HAND_params.append(HAND_params_tmp)

    HAND_params_DF = pd.DataFrame(HAND_params, columns=['Stage', 'Flow_Area', 'Wetted_Bed_Area', 'XS_area', 'Perimeter', 'Hydraulic_radius', 'Width'])

    if not unit == 'meter':
        HAND_params_DF['Stage'] = HAND_params_DF['Stage'] * mFactor
        HAND_params_DF['Flow_Area'] = HAND_params_DF['Flow_Area'] * mFactor ** 2
        HAND_params_DF['Wetted_Bed_Area'] = HAND_params_DF['Wetted_Bed_Area'] * mFactor ** 2
        HAND_params_DF['XS_area'] = HAND_params_DF['XS_area'] * mFactor ** 2
        HAND_params_DF['Perimeter'] = HAND_params_DF['Perimeter'] * mFactor
        HAND_params_DF['Hydraulic_radius'] = HAND_params_DF['Hydraulic_radius'] * mFactor
        HAND_params_DF['Width'] = HAND_params_DF['Width'] * mFactor

    if extended_SRC == 1:
        HAND_params_DF.to_excel(path_HAND+'HAND_params_extended.xlsx')

    else:
        HAND_params_DF.to_excel(path_HAND + 'HAND_params.xlsx')


def plot_HAND_params_dist(data_all, data_med, HAND_1m, HAND_10m, num_med, ind_med, jj, fig_params):

    A_all, P_all, R_all, W_all, d_all, h_all = data_all
    A_med, P_med, R_med, W_med, Line_ID_med = data_med

    path_fig, y_scale, save_to_excel, lw, pcolor, mcolor = fig_params

    if num_med == 1:
        legend = ['T10', 'T1', 'BM Median', 'BM']
    elif num_med == 5:
        legend = ['T10', 'T1', 'BM Median ' + str(int(ind_med[0])), 'BM Median ' + str(int(ind_med[1])),
                  'BM Median ' + str(int(ind_med[2])), 'BM Median ' + str(int(ind_med[3])),
                  'BM Median ' + str(int(ind_med[4])), 'BM']
    kde_bool = False

    plt.figure(10)
    p = sns.histplot(A_all, kde=kde_bool, stat="percent", color=pcolor, edgecolor=pcolor)
    # kde; compute a kernel density estimate to smooth the distribution and show on the plot as (one or more) line(s).
    # Only relevant with univariate data.
    h_max = get_max_height(p)
    plt.plot([HAND_10m['XS_area'][jj], HAND_10m['XS_area'][jj]], [0, h_max],
             color='black', linewidth=lw, linestyle='--')
    plt.plot([HAND_1m['XS_area'][jj], HAND_1m['XS_area'][jj]], [0, h_max],
             color='black', linewidth=lw)
    if num_med == 1:
        plt.plot([A_med[0], A_med[0]], [0, h_max], linewidth=lw, color=mcolor)
    else:
        for ind_med_ii in range(len(A_med)):
            plt.plot([A_med[ind_med_ii], A_med[ind_med_ii]], [0, h_max], linewidth=lw)
    plt.yscale(y_scale)
    plt.legend(legend, fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title('Flow Area ($m$)', fontsize=15)
    plt.ylabel('Percent (%)', fontsize=15)
    plt.xlabel('XS Area ($m^2$)', fontsize=15)
    plt.savefig(path_fig + str(jj + 1) + '_XS_Area_' + y_scale + '.png')
    plt.close()

    plt.figure(11)
    p = sns.histplot(P_all, kde=kde_bool, stat="percent", color=pcolor, edgecolor=pcolor)
    h_max = get_max_height(p)
    plt.plot([HAND_10m['Perimeter'][jj], HAND_10m['Perimeter'][jj]], [0, h_max],
             color='black', linewidth=lw, linestyle='--')
    plt.plot([HAND_1m['Perimeter'][jj], HAND_1m['Perimeter'][jj]], [0, h_max],
             color='black', linewidth=lw)

    if num_med == 1:
        plt.plot([P_med[0], P_med[0]], [0, h_max], linewidth=lw, color=mcolor)
    else:
        for ind_med_ii in range(len(P_med)):
            plt.plot([P_med[ind_med_ii], P_med[ind_med_ii]], [0, h_max], linewidth=lw)

    plt.yscale(y_scale)
    plt.legend(legend, fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title('Perimeter ($m$)', fontsize=15)
    plt.ylabel('Percent (%)', fontsize=15)
    plt.xlabel('Perimeter ($m$)', fontsize=15)
    plt.savefig(path_fig + str(jj + 1) + '_Perimeter_' + y_scale + '.png')
    plt.close()

    plt.figure(12)
    p = sns.histplot(R_all, kde=kde_bool, stat="percent", color=pcolor, edgecolor=pcolor)
    h_max = get_max_height(p)
    plt.plot([HAND_10m['Hydraulic_radius'][jj], HAND_10m['Hydraulic_radius'][jj]], [0, h_max],
             color='black', linewidth=lw, linestyle='--')
    plt.plot([HAND_1m['Hydraulic_radius'][jj], HAND_1m['Hydraulic_radius'][jj]], [0, h_max],
             color='black', linewidth=lw)

    if num_med == 1:
        plt.plot([R_med[0], R_med[0]], [0, h_max], linewidth=lw, color=mcolor)
    else:
        for ind_med_ii in range(len(R_med)):
            plt.plot([R_med[ind_med_ii], R_med[ind_med_ii]], [0, h_max], linewidth=lw)

    plt.yscale(y_scale)
    plt.legend(legend, fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title('Hydraulic Radius ($m$)')
    plt.ylabel('Percent (%)', fontsize=15)
    plt.xlabel('Hydraulic Radius ($m$)', fontsize=15)
    plt.savefig(path_fig + str(jj + 1) + '_Hydraulic_Radius_' + y_scale + '.png')
    plt.close()

    plt.figure(13)
    p = sns.histplot(W_all, kde=kde_bool, stat="percent", color=pcolor, edgecolor=pcolor)
    h_max = get_max_height(p)
    plt.plot([HAND_10m['Width'][jj], HAND_10m['Width'][jj]], [0, h_max],
             color='black', linewidth=lw, linestyle='--')
    plt.plot([HAND_1m['Width'][jj], HAND_1m['Width'][jj]], [0, h_max],
             color='black', linewidth=lw)
    if num_med == 1:
        plt.plot([W_med[0], W_med[0]], [0, h_max], linewidth=lw, color=mcolor)
    else:
        for ind_med_ii in range(len(W_med)):
            plt.plot([W_med[ind_med_ii], W_med[ind_med_ii]], [0, h_max], linewidth=lw)

    plt.xlabel('')
    plt.yscale(y_scale)
    plt.legend(legend, fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.title('Channel Width ($m$)')
    plt.ylabel('Percent (%)', fontsize=15)
    plt.xlabel('Channel Width ($m$)', fontsize=15)
    plt.savefig(path_fig + str(jj + 1) + '_Channel_Width_' + y_scale + '.png')
    plt.close()
def calc_XS_params(x, z0, ind):

    A, P, W, sum_Z, len_Z = 0, 0, 0, 0, 0

    for ii in range(0, ind.__len__() - 1, 2):
        m1 = (z0[ind[ii]] - z0[ind[ii] + 1]) / (x[ind[ii]] - x[ind[ii] + 1])
        xi1 = (-z0[ind[ii]] + m1 * x[ind[ii]]) / m1

        m2 = (z0[ind[ii + 1]] - z0[ind[ii + 1] + 1]) / (x[ind[ii + 1]] - x[ind[ii + 1] + 1])
        xi2 = (-z0[ind[ii + 1]] + m2 * x[ind[ii + 1]]) / m2

        X = np.hstack((xi1, x[ind[ii]:ind[ii + 1] + 1], xi2))
        Z = -np.hstack((0, z0[ind[ii]:ind[ii + 1] + 1], 0))

        dA = np.trapz(Z, x=X)

        dx = X[1:] - X[:-1]
        dz = Z[1:] - Z[:-1]
        dP = np.sum((dx ** 2 + dz ** 2) ** (1 / 2))

        A = A + dA
        P = P + dP
        W = W + (X[-1] - X[0])
        sum_Z = sum_Z + sum(Z)
        len_Z = len_Z + len(Z)

        return A, P, W, sum_Z, len_Z

def calc_HAND_params_dist(case_name,
                          flow_stages, Qnum_stages, mFactor_BM,
                          xsectshp, pointshp, terrain, num_med, fig_params):

    set_arcpy_env(xsectshp)

    path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, '0')
    path_GIS, path_HAND_1m, path_HAND_10m = set_path_HAND(case_name)
    path_fig, y_scale, save_to_excel, lw, pcolor, mcolor = fig_params

    for jj in range(flow_stages.__len__()):
        Q_num = Qnum_stages[jj][0]

        # Load Water Surface Elevation raster (From Benchmark)
        wse = path_tuflow + 'results/' + case_name + '/grids/' + case_name + '_' + Q_num + '_h_010_00.flt'
        print(jj + 1)

        # Stack Profile
        xsecttab = './gis/2d_xsect_table.dbf'

        ## Extract values to points
        inPointFeatures = pointshp
        inRaster = wse
        outPointFeatures = os.path.abspath(path_GIS + 'points_elev.shp')

        arcpy.CheckOutExtension("Spatial")
        arcpy.sa.ExtractValuesToPoints(inPointFeatures, inRaster, outPointFeatures,
                                       "INTERPOLATE", "VALUE_ONLY")

        ## Get WSE at station points
        wse_point = outPointFeatures.replace('shp', 'dbf')

        wsedbf = simpledbf.Dbf5(wse_point)
        wsedf = wsedbf.to_dataframe()
        wsedbf.f.close()

        ############################################
        ## Calculating Median Depth
        # Get XSs at station lines
        arcpy.CheckOutExtension("3D")
        arcpy.StackProfile_3d(xsectshp, profile_targets=[terrain], out_table=xsecttab)

        xsectdbf = simpledbf.Dbf5(xsecttab)
        xsectdf = xsectdbf.to_dataframe()
        xsectdbf.f.close()

        Line_IDs = xsectdf['LINE_ID'].unique()
        start_ind = 0
        h_tmp = []

        for Line_ID in range(start_ind, len(Line_IDs)):

            # Construct a functional relationship between A and h
            x = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_DIST'])
            z = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_Z'])

            elevation = min(z)
            water_stage = wsedf['RASTERVALU'][Line_ID]  # assuming ET_STATION is in ascending order

            if water_stage > 0:
                h_tmp = np.append(h_tmp, water_stage - elevation)

        h_perc50_tmp = np.percentile(h_tmp, 50)
        h_tmp = np.abs(h_tmp - h_perc50_tmp)

        h_tmp_order = h_tmp.argsort()
        h_tmp_ranks = h_tmp_order.argsort()

        ##################################################
        ## Calculating parameters at each XS

        Line_IDs_effective = []
        start_ind = 0

        for Line_ID in range(start_ind, len(Line_IDs)):

            # Construct a functional relationship between A and h
            x = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_DIST'])
            z = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_Z'])

            elevation = min(z)
            water_stage = wsedf['RASTERVALU'][Line_ID]  # assuming ET_STATION is in ascending order

            max_vwz = max(max(z), water_stage)
            x = np.insert(x, 0, x[0] - 0.01)
            x = np.insert(x, -1, x[-1] + 0.01)
            z = np.insert(z, 0, max_vwz + 0.1)
            z = np.insert(z, -1, max_vwz + 0.1)

            z0 = z - water_stage

            if water_stage > 0:
                Line_IDs_effective = np.append(Line_IDs_effective, Line_ID)

        ind_med = []
        for ind_med_ii in h_tmp_order:
            if ind_med_ii in Line_IDs_effective:
                if len(ind_med) < num_med:
                    ind_med = np.append(ind_med, ind_med_ii)

        A_all, P_all, R_all, W_all, d_all, h_all = [], [], [], [], [], []
        A_med, P_med, R_med, W_med, Line_ID_med = [], [], [], [], []
        # ind_med_jj = 0

        for Line_ID in range(start_ind, len(Line_IDs)):

            # Construct a functional relationship between A and h
            x = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_DIST'])
            z = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_Z'])

            elevation = min(z)
            water_stage = wsedf['RASTERVALU'][Line_ID]  # assuming ET_STATION is in ascending order

            max_vwz = max(max(z), water_stage)
            x = np.insert(x, 0, x[0] - 0.01)
            x = np.insert(x, -1, x[-1] + 0.01)
            z = np.insert(z, 0, max_vwz + 0.1)
            z = np.insert(z, -1, max_vwz + 0.1)

            z0 = z - water_stage

            if water_stage > 0:
                ind = []

                for ii in range(0, z.__len__() - 1):
                    if np.sign(z0[ii] * z0[ii + 1]) < 0 or z0[ii] == 0:
                        ind.append(ii)


                A, P, W, sum_Z, len_Z = calc_XS_params(x, z0, ind)

                R = A / P

                A_all = np.append(A_all, A)
                P_all = np.append(P_all, P)
                R_all = np.append(R_all, R)
                W_all = np.append(W_all, W)
                d_all = np.append(d_all, sum_Z / len_Z)
                h_all = np.append(h_all, water_stage - elevation)

                print('Line_ID = ' + str(Line_ID))

                if Line_ID in ind_med:
                    print('ind-med Line_ID = ' + str(Line_ID))

                    A_med = np.append(A_med, A)
                    P_med = np.append(P_med, P)
                    R_med = np.append(R_med, R)
                    W_med = np.append(W_med, W)
                    Line_ID_med = np.append(Line_ID_med, Line_ID)

                    ## XS profile
                    plt.figure(20)
                    plt.plot(x, z, color='black', linewidth=lw, linestyle='-')
                    plt.plot(x, np.ones(len(x)) * water_stage, color='black', linewidth=lw, linestyle='--')
                    plt.legend('XS profile', 'WSE', fontsize=15)
                    plt.xticks(fontsize=13)
                    plt.yticks(fontsize=13)
                    plt.title('Line ID = ' + str(Line_ID), fontsize=15)
                    ind_med_jj = np.where(ind_med == Line_ID)[0][0]
                    plt.savefig(
                        path_fig + Q_num + '_XS_profile_' + str(ind_med_jj + 1) + '_LineID_' + str(Line_ID) + '.png')
                    plt.close()

                """
                plt.figure(1)
                plt.plot(x, z, '-')
                plt.plot([np.min(x), np.max(x)], [water_stage, water_stage], '-')
                plt.xlabel('Lateral Distance ' + '(' + unit_T + ')')
                plt.ylabel('Elevation ' + '(' + unit_T + ')')
                plt.title('Cross-sectional profile at the upstream (or downstream)')
                plt.show()
                plt.close()
                """

        ###################################################################
        # Unit conversion of BM # feet to meter
        A_all = A_all * mFactor_BM ** 2
        P_all = P_all * mFactor_BM
        R_all = R_all * mFactor_BM
        W_all = W_all * mFactor_BM

        data_all = [A_all, P_all, R_all, W_all, d_all, h_all]
        data_med = [A_med, P_med, R_med, W_med, Line_ID_med]

        HAND_1m = pd.read_excel(path_HAND_1m + 'HAND_params.xlsx')  # in meters
        HAND_10m = pd.read_excel(path_HAND_10m + 'HAND_params.xlsx')  # in meters
        plot_HAND_params_dist(data_all, data_med, HAND_1m, HAND_10m, num_med, ind_med, jj, fig_params)

        if save_to_excel:
            df = pd.DataFrame({'A': A_all, 'P': P_all, 'R': R_all, 'W': W_all, 'avg_d': d_all, 'max_d': h_all})

            df.to_excel('./HG_' + Q_num + '.xlsx')
            # df.to_excel('./HG.xlsx', sheet_name=Q_num)

def plot_rating_curve(case_name,
                     flow_stages, Qnum_stages, channel_param,
                     xsectshp, pointshp, terrain, extended_SRC):

    Q_all = []
    h_array, h_per_array, Q_array, Qnum_array = [], [], [], []
    h_perc10, h_perc50, h_perc90 = [], [], []

    path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, '0')
    path_GIS, path_HAND_1m, path_HAND_10m = set_path_HAND(case_name)
    slope_NHD_1m, slope_NHD_10m, ManningN_NHD = read_slope_n(case_name)

    for jj in range(len(flow_stages)):
        Q_num = Qnum_stages[jj][0]
        Q = flow_stages['flow'][jj].round(5)

        slope_BM = channel_param['Slope'][channel_param['site'] == case_name]

        # Load Water Surface Elevation raster (From Benchmark)
        wse = path_tuflow + 'results/' + case_name + '/grids/' + case_name + '_' + Q_num + '_h_010_00.flt'
        print(jj + 1)

        set_arcpy_env(xsectshp)

        # Stack Profile
        xsecttab = path_BM + 'gis/2d_xsect_table_rc.dbf'

        ## Extract values to points
        inPointFeatures = pointshp
        inRaster = wse
        outPointFeatures = os.path.abspath(path_BM + 'gis/points_elev_rc.shp')

        arcpy.CheckOutExtension("Spatial")
        arcpy.sa.ExtractValuesToPoints(inPointFeatures, inRaster, outPointFeatures,
                                       "INTERPOLATE", "VALUE_ONLY")

        ## Get WSE at station points
        wse_point = outPointFeatures.replace('shp', 'dbf')

        wsedbf = simpledbf.Dbf5(wse_point)
        wsedf = wsedbf.to_dataframe()
        wsedbf.f.close()

        ############################################
        ## Calculating Median Depth
        # Get XSs at station lines
        arcpy.CheckOutExtension("3D")
        arcpy.StackProfile_3d(xsectshp, profile_targets=[terrain], out_table=xsecttab)

        xsectdbf = simpledbf.Dbf5(xsecttab)
        xsectdf = xsectdbf.to_dataframe()
        xsectdbf.f.close()

        Line_IDs = xsectdf['LINE_ID'].unique()
        start_ind = 0
        h_tmp, h_all = [], []

        for Line_ID in range(start_ind, len(Line_IDs)):

            # Construct a functional relationship between A and h
            x = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_DIST'])
            z = np.array(xsectdf.loc[xsectdf['LINE_ID'] == Line_ID]['FIRST_Z'])

            elevation = min(z)
            water_stage = wsedf['RASTERVALU'][Line_ID]  # assuming ET_STATION is in ascending order

            if water_stage > 0:
                h_tmp = np.append(h_tmp, water_stage - elevation)
                h_all = np.append(h_all, water_stage - elevation)

        # h_perc50_tmp = np.percentile(h_tmp, 50)
        # h_tmp = np.abs(h_tmp - h_perc50_tmp)

        h_tmp_order = h_tmp.argsort()
        h_tmp_ranks = h_tmp_order.argsort()
        h_per_tmp = np.ceil((h_tmp_ranks / (len(h_tmp) - 1)) * 10) * 10
        Q_tmp = Q * np.ones(np.size(h_tmp))
        Qnum_tmp = np.array(['%s' % Q_num for kk in range(len(h_tmp))])
        h_perc10_tmp = np.percentile(h_tmp, 10)
        h_perc50_tmp = np.percentile(h_tmp, 50)
        h_perc90_tmp = np.percentile(h_tmp, 90)

        arcpy.management.Delete(outPointFeatures)

        h_array = np.append(h_array, h_all)
        h_per_array = np.append(h_per_array, h_per_tmp)
        Q_array = np.append(Q_array, Q_tmp)
        Qnum_array = np.append(Qnum_array, Qnum_tmp)
        h_perc10 = np.append(h_perc10, h_perc10_tmp)
        h_perc50 = np.append(h_perc50, h_perc50_tmp)
        h_perc90 = np.append(h_perc90, h_perc90_tmp)
        Q_all = np.append(Q_all, Q)

    Qnum_array = np.array(Qnum_array)
    df = pd.DataFrame({'h': h_array, 'h_per': h_per_array, 'Q': Q_array, 'Q_num': Qnum_array})

    # df = df[df['d'] > 0]

    """
    # Option 1, violin plot -
    sns.catplot(data=df, x=0, y=1, kind="violin", orient="h")
    """

    # Option 2, Scatter plot
    # plt.figure(5)
    fig, ax = plt.subplots()

    sp = sns.scatterplot(data=df, x='h', y='Q', hue='h_per', s=200,
                         palette='Spectral', edgecolor='none', alpha=0.3)
    # sns.color_palette("Spectral", as_cmap=True)
    # ax.legend(loc=2)
    ax.get_legend().remove()

    """
    # Option 3, Scatter plot colored by probability density
    hist, bin_edges = np.histogram(x)
    bin_midpoint = (bin_edges[1:] + bin_edges[:-1]) / 2
    xx = bin_midpoint
    yy = y[0]*np.ones(np.size(xx))
    ddf = pd.DataFrame(np.transpose(np.vstack((xx, yy, hist))))
    sns.scatterplot(data=ddf, x=0, y=1,
                    hue=2, size=2, alpha=0.5)
    """

    ## HAND rating curves
    if extended_SRC == 1:
        HAND_1m = pd.read_excel(path_HAND_1m + 'HAND_params_extended.xlsx')
        HAND_10m = pd.read_excel(path_HAND_10m + 'HAND_params_extended.xlsx')
    else:
        HAND_1m = pd.read_excel(path_HAND_1m + 'HAND_params.xlsx')
        HAND_10m = pd.read_excel(path_HAND_10m + 'HAND_params.xlsx')

    HAND_1m['Q'] = (1 / ManningN_NHD) * HAND_1m['XS_area'] * HAND_1m['Hydraulic_radius'] ** (2 / 3) * slope_NHD_1m ** (
                1 / 2)
    HAND_10m['Q'] = (1 / ManningN_NHD) * HAND_10m['XS_area'] * HAND_10m['Hydraulic_radius'] ** (
                2 / 3) * slope_NHD_10m ** (1 / 2)

    ax2 = ax.twiny()

    # ax2.plot(h_perc10, Q_all, 'r')
    ax2.plot(h_perc50, Q_all, 'g')
    # ax2.plot(h_perc90, Q_all, 'b')
    ax2.plot(HAND_10m['Stage'], HAND_10m['Q'], marker='^', markersize=10, c='black')  # , linestyle='None')
    ax2.plot(HAND_1m['Stage'], HAND_1m['Q'], marker='X', markersize=10, c='black')  # , linestyle='None')
    # ax2.plot(3.19, 568, marker='o', markersize=13, c='blue')

    ax.set_xlim([0, 7.5])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([])
    # s_T1 = sns.scatterplot(d_T1, Q_T1, marker='s', color='black', s=200)

    ax2.legend(['50 Percentile',  # '50 Percentile', '90 Percentile',
                'T10', 'T1'], loc=2)

    # print(flow_stages['stage'])
    # print(HAND_1m['Q'])
    # print(HAND_10m['Q'])
    # print(df)

    if extended_SRC == 1:
        plt.savefig(path_BM + 'SRCs_extended.png')
        df.to_excel(path_BM + 'df_SRCs_extended.xlsx')

    else:
        plt.savefig(path_BM + 'SRCs.png')
        df.to_excel(path_BM + 'df_SRCs.xlsx')