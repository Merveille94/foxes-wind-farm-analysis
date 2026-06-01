import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import foxes
import foxes.variables as FV
import foxes.constants as FC
from scipy.spatial.distance import cdist
import os
from pathlib import Path

LEN = 0
SCRIPT_DIR = Path(__file__).resolve().parent
path_aoi = str(SCRIPT_DIR / "turbine-info/area_of_interest/*.csv")
path_existing = str(SCRIPT_DIR / "turbine-info/existing/*.csv")
path_planned = str(SCRIPT_DIR / "turbine-info/planned_future/*.csv")

ALLOWED_DISTANCE = 50000 # in meters

farms_fields = {
    "aoi": glob.glob(path_aoi),
    "existing": glob.glob(path_existing),
    "planned":glob.glob(path_planned)
    }

turbines = {
    "aoi": ["IEA15MW"],
    "existing": ["NREL5MW"],
    "planned":["IEA15MW"]
}

# Read all coordinates in the areas N-10.1/N-10.2 and concatenate them into a single 2D array
aio_coords = pd.concat([pd.read_csv(file)[['x', 'y']] for file in farms_fields["aoi"]], ignore_index=True).values

# intialize farms
farm_all = foxes.WindFarm(name="farm_all")


for category, files in farms_fields.items():
    for file in files:
         print(f"processing file: {file}...")
         df_temp = pd.read_csv(file)
         
         
         if category != "aoi":
            # Calculate the distance from each turbine in df_temp to the nearest turbine in aio_coords
            distances = cdist(df_temp[['x', 'y']].values, aio_coords)
            min_distance_to_aio = np.min(distances, axis=1)
            df_temp = df_temp[min_distance_to_aio <= ALLOWED_DISTANCE]
        
         for index, row in df_temp.iterrows():
                farm_all.add_turbine(foxes.Turbine(
                    xy=[row['x'],row['y']],
                    name=category,
                    turbine_models=turbines[category]
                ))
         LEN += len(df_temp)
#breakpoint()

states = foxes.input.states.StatesTable(
        data_source=str(SCRIPT_DIR / "reanalysis/reanalysis_2006.csv"),
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        var2col={FV.WS: "WS100", FV.WD: "WD100"},
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.05},
)


mbook = foxes.ModelBook()

algo = foxes.algorithms.Downwind(
    mbook=mbook,
    farm=farm_all,
    states=states,
    rotor_model="centre",
    wake_models=["Bastankhah2014_linear_k004"],
    wake_frame="rotor_wd",
    partial_wakes=None,
)

engine = foxes.Engine.new(
    engine_type="process",
    n_procs=4,
    chunk_size_states=73,
    verbosity=0,
)

aoi_inds = [
            i for i, t in enumerate(farm_all.turbines)
            if t.name == "aoi"
]

print(f"AOI turbines: {len(aoi_inds)}")


assert len(farm_all.turbines) == LEN, "Error: Wrong Number of Turbines"
assert len([t for t in farm_all.turbines if t.name=="aoi"]) == 166, "Error: Wrong Number of AOI TurbineError: Wrong Number of AOI Turbiness"

print ("sanity checks okay")
with engine:
        farm_results = algo.calc_farm()

farm_results.to_netcdf("farm_results_full1.nc")
farm_results.to_dataframe().to_csv("farm_results_full1.csv")

farm_results_aoi = farm_results.isel(turbine=aoi_inds)


o_aoi = foxes.output.FarmResultsEval(farm_results_aoi)

P0_mean = o_aoi.calc_mean_farm_power(ambient=True)
P_mean = o_aoi.calc_mean_farm_power()
print(f"\nFarm power        : {P_mean/1000:.1f} MW")
print(f"Farm ambient power: {P0_mean/1000:.1f} MW")
print(f"Farm efficiency   : {o_aoi.calc_farm_efficiency()*100:.2f} %")

o_full = foxes.output.FarmResultsEval(farm_results)

yld = o_full.calc_turbine_yield(algo=algo, annual=True,)

aoi_yield = yld.iloc[aoi_inds, 0].sum()

print(f"Annual farm yield : {aoi_yield:.2f} GWh")
