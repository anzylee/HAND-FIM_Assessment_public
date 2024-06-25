from fun import *

## Input ####################################################################################
case_name = 'SFE_Leggett' # 'SFE_Leggett', 'SFE_Miranda'

# Load params
unit_BM, mFactor_BM, reach_length_m, thal_interval = set_params(case_name)
extended_SRC = 1 # 1 if you want to plot the extended version

for res in ['1m', '10m']:
    print(res)

    # Setting paths
    path_main, path_BM, path_HAND, path_tuflow = set_path(case_name, res)
    os.chdir(path_HAND)

    unit, mFactor = define_mFactor(res) # unit conversion

    # Downstream elev, flow, stage, and Qnum
    elev_downstream = cal_elev_downstream(path_tuflow) # downstream elevation of terrain
    flow_stages, Qnum_stages = flow_Qnum_stages(path_tuflow, case_name, elev_downstream) # flow, Qnum, and stages

    # HAND and DEM
    HAND_Mask, DEM_Mask = prepare_gis_files()

    print(Qnum_stages)
    Stages_DF = pd.DataFrame(Qnum_stages, columns=['Code','Stage_BM'])
    Stages_DF['Stage'] = Stages_DF['Stage_BM'] * mFactor_BM / mFactor  # unit conversion

    # Additional points for SRC
    Stages_DF = load_stages_df_extended(case_name, Stages_DF, extended_SRC)

    # Calculate HAND Parameters
    calc_HAND_params(Stages_DF, HAND_Mask, DEM_Mask, path_HAND, reach_length_m, mFactor, unit, extended_SRC)