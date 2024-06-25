from fun import *

case_name = 'SFE_Leggett'   ## Base:002, Bankfull: 011
case_name = 'SFE_Miranda'   ## Base:001, Bankfull: 011

num_med = 5 # number of XSs whose depth is close to median

# Load params
unit_BM, mFactor_BM, reach_length_m, thal_interval = set_params(case_name)

# Setting paths
path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, '0')
os.chdir(path_BM)

# Load channel params (n, slope, ...) used for tuflow sim
channel_param = read_channel_params(case_name)

# Define the unit of Benchmark terrain
unit, mFactor = define_mFactor('1m')

# Downstream elev, flow, stage, and Qnum
elev_downstream = cal_elev_downstream(path_tuflow)
flow_stages, Qnum_stages = flow_Qnum_stages(path_tuflow, case_name, elev_downstream)

# Load paths to XS/point shp and terrain
xsectshp, pointshp, terrain = set_path_shp(case_name, thal_interval)

# Load figure parameters for HAND param distribution plot
fig_params = set_fig_params(num_med)

# Calculate/plot HAND parameter distribution
calc_HAND_params_dist(case_name,
                      flow_stages, Qnum_stages, mFactor_BM,
                      xsectshp, pointshp, terrain, num_med, fig_params)