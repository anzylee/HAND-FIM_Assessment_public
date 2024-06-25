# Program file path
import os

## Inputs
#       DEM (dem.tif)
#       Stream raster (src.tif)

## Outputs
#       HAND raster (hand.tif)

# # Format and performance parameters
np = os.cpu_count()
# dsepsg="EPSG:4269"
# bufferdist='0.2'

print(os.path.abspath(os.path.curdir))
os.chdir('D:/CIROH/HAND-FIM_Assessment/codes/SFE_Miranda_hand_param_calc/HAND_1m')

# Pit remove
os.system('mpiexec -np '+str(np)+' pitremove -z dem.tif -fel fel.tif')

# Flow direction (inf)
os.system('mpiexec -np '+str(np)+' dinfflowdir -fel fel.tif -ang ang.tif -slp slp.tif')

# Flow direction (d8)
os.system('mpiexec -np '+str(np)+' d8flowdir -fel fel.tif -p p.tif -sd8 sd8.tif')

# Calculate HAND
os.system('mpiexec -np '+str(np)+' dinfdistdown -fel fel.tif -ang ang.tif -src src.tif -dd hand.tif -m ave v')


