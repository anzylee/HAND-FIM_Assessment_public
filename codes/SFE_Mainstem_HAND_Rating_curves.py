from fun import *

## sfe_Leggett
case_name = 'SFE_Leggett' # 'SFE_Leggett', 'SFE_Miranda'
extended_SRC = 1

# Load params
unit_BM, mFactor_BM, reach_length_m, thal_interval = set_params(case_name)

## Unit of terrain alternatives
unit_T = 'meter'
mFactor_T = 1

# Setting paths
path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, '0')
os.chdir(path_main)

# Load channel params (n, slope, ...) used for tuflow sim
channel_param = read_channel_params(case_name)

# Downstream elev, flow, stage, and Qnum
elev_downstream = cal_elev_downstream(path_tuflow)
flow_stages, Qnum_stages = flow_Qnum_stages(path_tuflow, case_name, elev_downstream)

# Unit conversion if needed
flow_stages = convert_unit_BM_to_T(flow_stages, mFactor_BM, mFactor_T)

# Load paths to XS/point shp and terrain
xsectshp, pointshp, terrain = set_path_shp(case_name, thal_interval)

# Plot rating curves
plot_rating_curve(case_name,
                  flow_stages, Qnum_stages, channel_param,
                  xsectshp, pointshp, terrain, extended_SRC)