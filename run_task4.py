import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import foxes
import foxes.variables as FV
import foxes.constants as FC

path_aoi = '/dss/work/wika1715/wind/turbine-info/area_of_interest/*csv'
path_existing = '/dss/work/wika1715/wind/turbine-info/existing/*csv'
path_planned = '/dss/work/wika1715/wind/turbine-info/planned_future/*csv'

farm_all = foxes.WindFarm(name="farm_all")

files_aoi = glob.glob(path_aoi)
files_existing = glob.glob(path_existing)
files_planned = glob.glob(path_planned)

for file in files_aoi:
    df_temp = pd.read_csv(file)

    for turbine in range(len(df_temp)):
        farm_all.add_turbine(foxes.Turbine(
            xy=[df_temp['x'][turbine],df_temp['y'][turbine]],
            name="aoi",
            turbine_models=["IEA15MW"]
        ))

for file in files_existing:
        df_temp = pd.read_csv(file)

        for turbine in range(len(df_temp)):
            farm_all.add_turbine(foxes.Turbine(
                xy=[df_temp['x'][turbine], df_temp['y'][turbine]],
                name="existing",
                turbine_models=["NREL5MW"]
            ))

for file in files_planned:
    df_temp = pd.read_csv(file)
    
    for turbine in range(len(df_temp)):
        farm_all.add_turbine(foxes.Turbine(
            xy=[df_temp['x'][turbine], df_temp['y'][turbine]],
            name="planned",
            turbine_models=["IEA15MW"]
        ))

states = foxes.input.states.StatesTable(
        data_source="reanalysis/reanalysis_2006.csv",
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
    n_procs=128,
    chunk_size_states=73,
    verbosity=0,
)

aoi_inds = [
            i for i, t in enumerate(farm_all.turbines)
            if t.name == "aoi"
]

print(f"AOI turbines: {len(aoi_inds)}")


assert len(farm_all.turbines) == 5143, "Error: Wrong Number of Turbines"
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
