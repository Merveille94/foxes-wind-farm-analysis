"""
=============================================================================
PMAWE 2026 – Task 3 & Task 6 (No-Wake scenario)
Energy Yield Analysis for Offshore Wind Farm Layouts N-10.1 and N-10.2
=============================================================================

This module contains:
  - Task 3 : Yield computation (per-turbine, farm total, mean power)
  - Task 6 : Spatial Pattern-of-Production plots (no-wake)
  - Extended: Directional sector analysis
  - Extended: Temporal extremes (highest / lowest energy periods)
  - Extended: Seasonality  (monthly aggregation)
  - Extended: Diurnal variation (hour-of-day aggregation)
  - Extended: Side-by-side comparison N-10.1 vs N-10.2

All functions accept the foxes results DataFrames produced by
    results.to_dataframe()
and the corresponding farm / algorithm objects returned by compute_farm().

Dependencies: numpy, pandas, matplotlib, seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Colour palette (consistent across all plots)
# ---------------------------------------------------------------------------
C1 = "#1f77b4"   # N-10.1 (blue)
C2 = "#d62728"   # N-10.2 (red)
SEASON_COLORS = {"Winter": "#4e79a7", "Spring": "#59a14f",
                 "Summer": "#f28e2b", "Autumn": "#e15759"}


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def _month_to_season(month: int) -> str:
    """Map month number to meteorological season (Northern Hemisphere)."""
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring",  5: "Spring",
            6: "Summer", 7: "Summer",  8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn"}[month]


def prepare_results_df(results_xr) -> pd.DataFrame:
    """
    Convert the raw xarray results from algorithm.calc_farm() to a clean
    pandas DataFrame with a proper DatetimeIndex on the state level.

    Parameters
    ----------
    results_xr : xarray.Dataset
        Output of algorithm.calc_farm()

    Returns
    -------
    df : pd.DataFrame
        MultiIndex (state [datetime], turbine [int]), power in kW in col 'P'
    """
    df = results_xr.to_dataframe().reset_index()
    df["state"] = pd.to_datetime(df["state"])
    df = df.set_index(["state", "turbine"])
    return df


def _farm_power_series(df: pd.DataFrame) -> pd.Series:
    """Total farm power [kW] at every timestep."""
    return df["P"].groupby(level="state").sum()


def _n_turbines(df: pd.DataFrame) -> int:
    return df.index.get_level_values("turbine").nunique()


# ============================================================================
# TASK 3 — YIELD COMPUTATION
# ============================================================================

def print_yield_summary(df: pd.DataFrame, farm_name: str = "Farm"):
    """
    Print per-turbine and farm-total annual yield summary, computed
    directly from the results DataFrame using the foxes `weight` column.

    weight_i = Δt_i / 8760, so:
        AEP_turbine [GWh] = 8760 * sum(P_i * weight_i) / 1e6
        Mean power   [kW] = AEP_farm [GWh] * 1e6 / 8760

    This avoids passing algo/out objects, which can cause shape-mismatch
    errors when multiple compute_farm() calls share the same states object.

    Parameters
    ----------
    df        : prepared results DataFrame (MultiIndex state/turbine)
    farm_name : str, label for the printout
    """
    print(f"\n{'='*60}")
    print(f"  YIELD SUMMARY — {farm_name}")
    print(f"{'='*60}")

    # Per-turbine annual yield [GWh]
    turbine_yield = (
        df.groupby(level="turbine")
          .apply(lambda g: (g["P"] * g["weight"]).sum() * 8760 / 1e6)
    )

    farm_yield    = turbine_yield.sum()               # [GWh/year]
    mean_power_MW = farm_yield * 1e3 / 8760           # [MW]  (GWh*1e3/h = MW)

    n_t = len(turbine_yield)
    print(f"  Number of turbines      : {n_t}")
    print(f"  Turbine yield (annual)  :")
    print(f"    Min  : {turbine_yield.min():.2f} GWh")
    print(f"    Max  : {turbine_yield.max():.2f} GWh")
    print(f"    Mean : {turbine_yield.mean():.2f} GWh")
    print(f"  Total farm yield/year   : {farm_yield:.2f} GWh")
    print(f"  Mean farm power         : {mean_power_MW:.2f} MW")

    rated_power_MW  = 15.0 * n_t                      # IEA-15 MW × n turbines
    aep_theor       = rated_power_MW * 8760 / 1e3     # GWh if always at rated
    capacity_factor = farm_yield / aep_theor * 100
    print(f"  Rated capacity          : {rated_power_MW:.0f} MW")
    print(f"  Capacity factor         : {capacity_factor:.1f} %")
    print(f"{'='*60}\n")
    return turbine_yield, farm_yield, mean_power_MW


# ============================================================================
# TASK 6 — SPATIAL PATTERN OF PRODUCTION  (no-wake)
# ============================================================================

def _compute_turbine_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-turbine annual yield [GWh] and spatial position.
    weight column = Δt_i / 8760  (foxes normalisation),
    so  AEP [GWh] = 8760 * sum(P * weight) / 1e6
    """
    return (
        df.groupby(level="turbine")
          .apply(lambda g: pd.Series({
              "X"          : g["X"].iloc[0],
              "Y"          : g["Y"].iloc[0],
              "tname"      : g["tname"].iloc[0],
              "annual_GWh" : (g["P"] * g["weight"]).sum() * 8760 / 1e6
          }))
    )


def plot_pattern_of_production(df: pd.DataFrame,
                               farm_name: str = "Farm",
                               ax=None):
    """
    Spatial scatter plot of per-turbine total power output (Pattern of
    Production).  Marker colour encodes annual yield [GWh].

    In the no-wake case all turbines show nearly identical colours because
    every turbine sees the same undisturbed inflow.  The spatial contrast
    becomes apparent only in the wake scenario (Task 4/6).

    Parameters
    ----------
    df        : prepared results DataFrame (MultiIndex state/turbine)
    farm_name : string label used in the title
    ax        : matplotlib Axes; a new figure is created when None
    """
    turbine_df = _compute_turbine_yield(df)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 8))

    sc = ax.scatter(
        turbine_df["X"] / 1000,   # m → km
        turbine_df["Y"] / 1000,
        c=turbine_df["annual_GWh"],
        cmap="plasma",
        s=80,
        edgecolors="k",
        linewidths=0.4,
        zorder=3
    )
    plt.colorbar(sc, ax=ax, label="Annual yield [GWh]", pad=0.02)
    ax.set_xlabel("Easting [km]", fontsize=11)
    ax.set_ylabel("Northing [km]", fontsize=11)
    ax.set_title(f"Pattern of Production (No Wake)\n{farm_name}", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.show()

    return turbine_df


def plot_pattern_of_production_comparison(df1: pd.DataFrame,
                                          df2: pd.DataFrame,
                                          names=("N-10.1", "N-10.2")):
    """
    Side-by-side Pattern of Production for two layouts, using a shared
    colour scale so yields are directly comparable across farms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    t1 = plot_pattern_of_production(df1, farm_name=names[0], ax=axes[0])
    t2 = plot_pattern_of_production(df2, farm_name=names[1], ax=axes[1])

    # Harmonise colour scale across both subplots
    vmin = min(t1["annual_GWh"].min(), t2["annual_GWh"].min())
    vmax = max(t1["annual_GWh"].max(), t2["annual_GWh"].max())
    for ax in axes:
        ax.collections[0].set_clim(vmin, vmax)

    fig.suptitle("Pattern of Production – No Wake Scenario (2006 ERA5)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXTENDED ANALYSIS 1 — DIRECTIONAL SECTOR ANALYSIS
# ============================================================================

def plot_directional_yield(df: pd.DataFrame,
                           farm_name: str = "Farm",
                           n_sectors: int = 12,
                           ax=None,
                           color=C1):
    """
    Bar chart showing how much of the annual energy yield originates from
    each wind-direction sector (wind rose for energy, not frequency).

    Uses AMB_WD (ambient wind direction at the first turbine per timestep)
    since all turbines share the same undisturbed conditions in no-wake.

    Parameters
    ----------
    n_sectors : number of directional bins  (12 → 30° wide each)
    """
    per_state = (
        df.groupby(level="state")
          .agg(P_farm=("P", "sum"),
               WD=("AMB_WD", "first"),
               weight=("weight", "first"))
    )

    sector_width = 360 / n_sectors
    per_state["sector"] = (
        (per_state["WD"] + sector_width / 2) % 360 // sector_width
    ).astype(int)

    # Annual energy contribution per sector [GWh]
    sector_yield = (
        per_state.groupby("sector")
                 .apply(lambda g: (g["P_farm"] * g["weight"]).sum() * 8760 / 1e6)
                 .reindex(range(n_sectors), fill_value=0.0)
    )

    sector_labels = [
        f"{int(i * sector_width)}°–{int((i+1)*sector_width)}°"
        for i in range(n_sectors)
    ]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(range(n_sectors), sector_yield.values,
           color=color, edgecolor="white", alpha=0.85)

    # Highlight the dominant sector in gold
    dom = int(sector_yield.idxmax())
    ax.bar(dom, sector_yield[dom], color="gold", edgecolor="k",
           label=f"Dominant: {sector_labels[dom]}")

    ax.set_xticks(range(n_sectors))
    ax.set_xticklabels(sector_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annual energy contribution [GWh]", fontsize=10)
    ax.set_title(f"Energy Yield by Wind Direction Sector\n{farm_name}", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()

    return sector_yield


def plot_directional_yield_comparison(df1, df2,
                                      names=("N-10.1", "N-10.2"),
                                      n_sectors=12):
    """Side-by-side directional yield bar charts for two layouts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_directional_yield(df1, farm_name=names[0],
                           n_sectors=n_sectors, ax=axes[0], color=C1)
    plot_directional_yield(df2, farm_name=names[1],
                           n_sectors=n_sectors, ax=axes[1], color=C2)
    fig.suptitle("Energy Yield Contribution by Wind Direction Sector (No Wake)",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXTENDED ANALYSIS 2 — TEMPORAL EXTREMES (highest / lowest energy periods)
# ============================================================================

def print_temporal_extremes(df: pd.DataFrame,
                             farm_name: str = "Farm",
                             top_n: int = 10):
    """
    Print the timesteps with the highest and lowest total farm power,
    together with the ambient wind speed and direction at that moment.
    """
    per_state = (
        df.groupby(level="state")
          .agg(P_farm=("P", "sum"),
               WS=("AMB_REWS", "mean"),
               WD=("AMB_WD", "first"))
    )
    per_state["P_farm_MW"] = per_state["P_farm"] / 1e3

    print(f"\n{'='*65}")
    print(f"  TOP-{top_n} HIGHEST POWER STATES — {farm_name}")
    print(f"{'='*65}")
    print(per_state.nlargest(top_n, "P_farm_MW")[["P_farm_MW", "WS", "WD"]]
                   .to_string())

    print(f"\n  TOP-{top_n} LOWEST POWER STATES (P > 0) — {farm_name}")
    print(f"{'='*65}")
    print(per_state[per_state["P_farm"] > 0]
                   .nsmallest(top_n, "P_farm_MW")[["P_farm_MW", "WS", "WD"]]
                   .to_string())


# ============================================================================
# EXTENDED ANALYSIS 3 — SEASONALITY (monthly and meteorological seasons)
# ============================================================================

def plot_seasonality(df: pd.DataFrame,
                     farm_name: str = "Farm",
                     color=C1,
                     ax_month=None,
                     ax_season=None):
    """
    Two sub-plots:
      Left  — absolute energy yield per calendar month [GWh]
      Right — aggregated by meteorological season (DJF/MAM/JJA/SON)

    Expects either both axes to be provided (for embedding in a larger
    figure) or neither (creates its own 1×2 figure).
    """
    per_state = (
        df.groupby(level="state")
          .agg(P_farm=("P", "sum"),
               weight=("weight", "first"))
    )
    per_state.index = pd.to_datetime(per_state.index)
    per_state["energy_GWh"] = per_state["P_farm"] * per_state["weight"] * 8760 / 1e6
    per_state["month"]      = per_state.index.month
    per_state["season"]     = per_state.index.month.map(_month_to_season)

    monthly  = per_state.groupby("month")["energy_GWh"].sum()
    seasonal = per_state.groupby("season")["energy_GWh"].sum()
    season_order = ["Winter", "Spring", "Summer", "Autumn"]

    standalone = ax_month is None
    if standalone:
        fig, (ax_month, ax_season) = plt.subplots(1, 2, figsize=(14, 5))

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax_month.bar(month_names, [monthly.get(m, 0) for m in range(1, 13)],
                 color=color, edgecolor="white", alpha=0.85)
    ax_month.set_ylabel("Energy yield [GWh]", fontsize=10)
    ax_month.set_title(f"Monthly Energy Yield\n{farm_name}", fontsize=12)
    ax_month.grid(axis="y", linestyle="--", alpha=0.4)
    ax_month.tick_params(axis="x", rotation=30)

    s_colors = [SEASON_COLORS[s] for s in season_order]
    ax_season.bar(season_order, [seasonal.get(s, 0) for s in season_order],
                  color=s_colors, edgecolor="white", alpha=0.85)
    ax_season.set_ylabel("Energy yield [GWh]", fontsize=10)
    ax_season.set_title(f"Seasonal Energy Yield\n{farm_name}", fontsize=12)
    ax_season.grid(axis="y", linestyle="--", alpha=0.4)

    if standalone:
        plt.tight_layout()
        plt.show()

    return monthly, seasonal


def plot_seasonality_comparison(df1, df2, names=("N-10.1", "N-10.2")):
    """Four-panel comparison: monthly & seasonal for both layouts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_seasonality(df1, farm_name=names[0], color=C1,
                     ax_month=axes[0, 0], ax_season=axes[0, 1])
    plot_seasonality(df2, farm_name=names[1], color=C2,
                     ax_month=axes[1, 0], ax_season=axes[1, 1])
    fig.suptitle("Seasonality Analysis – No Wake Scenario (2006 ERA5)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXTENDED ANALYSIS 4 — DIURNAL VARIATION (hour-of-day)
# ============================================================================

def plot_diurnal(df: pd.DataFrame,
                 farm_name: str = "Farm",
                 color=C1,
                 ax=None):
    """
    Mean farm power [MW] for each hour of the day aggregated over 2006.
    The shaded band shows ±1 std dev across all days in the year.

    Offshore ERA5 typically shows a weak diurnal signal (unlike land sites),
    but the pattern is still worth examining and documenting.
    """
    per_state = (
        df.groupby(level="state")
          .agg(P_farm=("P", "sum"))
    )
    per_state.index = pd.to_datetime(per_state.index)
    per_state["hour"] = per_state.index.hour
    per_state["P_MW"] = per_state["P_farm"] / 1e3

    hourly_mean = per_state.groupby("hour")["P_MW"].mean()
    hourly_std  = per_state.groupby("hour")["P_MW"].std()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(hourly_mean.index, hourly_mean.values,
            color=color, lw=2, label="Mean farm power")
    ax.fill_between(hourly_mean.index,
                    hourly_mean - hourly_std,
                    hourly_mean + hourly_std,
                    color=color, alpha=0.2, label="±1 std dev")
    ax.set_xlabel("Hour of day (UTC)", fontsize=10)
    ax.set_ylabel("Farm power [MW]", fontsize=10)
    ax.set_title(f"Diurnal Variation of Farm Power\n{farm_name}", fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()

    return hourly_mean


def plot_diurnal_comparison(df1, df2, names=("N-10.1", "N-10.2")):
    """Overlay diurnal power curves for both layouts on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for df, color, name in zip([df1, df2], [C1, C2], names):
        per_state = df.groupby(level="state").agg(P_farm=("P", "sum"))
        per_state.index = pd.to_datetime(per_state.index)
        per_state["hour"] = per_state.index.hour
        per_state["P_MW"] = per_state["P_farm"] / 1e3

        hmean = per_state.groupby("hour")["P_MW"].mean()
        hstd  = per_state.groupby("hour")["P_MW"].std()

        ax.plot(hmean.index, hmean.values, color=color, lw=2, label=name)
        ax.fill_between(hmean.index, hmean - hstd, hmean + hstd,
                        color=color, alpha=0.15)

    ax.set_xlabel("Hour of day (UTC)", fontsize=10)
    ax.set_ylabel("Farm power [MW]", fontsize=10)
    ax.set_title("Diurnal Variation – No Wake Scenario (2006 ERA5)", fontsize=12)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# ============================================================================
# COMPREHENSIVE DASHBOARD — 6-panel summary for a single layout
# ============================================================================

def plot_full_dashboard(df: pd.DataFrame,
                        farm_name: str = "Farm",
                        color=C1):
    """
    6-panel dashboard for one farm layout:
      [top-left]    Pattern of Production (spatial, tall)
      [top-right]   Directional sector yield
      [mid-left]    Monthly energy yield
      [mid-right]   Seasonal energy yield
      [bot-left/mid] Diurnal variation
      [bot-right]   Power duration curve
    """
    fig = plt.figure(figsize=(20, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_spatial  = fig.add_subplot(gs[0:2, 0])   # tall left panel (rows 0–1)
    ax_dir      = fig.add_subplot(gs[0, 1:])    # wide top-right
    ax_monthly  = fig.add_subplot(gs[1, 1])
    ax_seasonal = fig.add_subplot(gs[1, 2])
    ax_diurnal  = fig.add_subplot(gs[2, 0:2])   # wide bottom-left
    ax_pdc      = fig.add_subplot(gs[2, 2])

    # Panel 1 — spatial
    plot_pattern_of_production(df, farm_name=farm_name, ax=ax_spatial)

    # Panel 2 — directional
    plot_directional_yield(df, farm_name="", n_sectors=12,
                           ax=ax_dir, color=color)

    # Panels 3 & 4 — seasonality
    plot_seasonality(df, farm_name="", color=color,
                     ax_month=ax_monthly, ax_season=ax_seasonal)

    # Panel 5 — diurnal
    plot_diurnal(df, farm_name="", color=color, ax=ax_diurnal)

    # Panel 6 — power duration curve
    per_state = df.groupby(level="state").agg(P_farm=("P", "sum"))
    p_sorted  = per_state["P_farm"].sort_values(ascending=False).values / 1e3
    hours     = np.arange(len(p_sorted))
    ax_pdc.plot(hours, p_sorted, color=color, lw=1.5)
    ax_pdc.fill_between(hours, 0, p_sorted, color=color, alpha=0.15)
    ax_pdc.set_xlabel("Hours [h]", fontsize=9)
    ax_pdc.set_ylabel("Farm power [MW]", fontsize=9)
    ax_pdc.set_title("Power Duration Curve", fontsize=11)
    ax_pdc.grid(linestyle="--", alpha=0.4)

    fig.suptitle(
        f"No-Wake Energy Analysis Dashboard — {farm_name} (2006 ERA5)",
        fontsize=16, y=1.01, fontweight="bold"
    )
    plt.show()


# ============================================================================
# MAIN ENTRY POINT — call this from your notebook
# ============================================================================

def run_full_analysis(results1_xr, results2_xr):
    """
    Run and display every analysis for both layouts.

    Parameters
    ----------
    results1_xr, results2_xr : raw xarray Dataset from algorithm.calc_farm()

    Usage in notebook
    -----------------
    import nowake_analysis as ta
    ta.run_full_analysis(results1, results2)
    """
    # ---- Convert to DataFrames ----------------------------------------
    df1 = prepare_results_df(results1_xr)
    df2 = prepare_results_df(results2_xr)

    # ---- Task 3: Yield Summary (no algo/out needed) -------------------
    print_yield_summary(df1, farm_name="N-10.1")
    print_yield_summary(df2, farm_name="N-10.2")

    # ---- Task 6: Spatial Pattern of Production ------------------------
    plot_pattern_of_production_comparison(df1, df2)

    # ---- Temporal extremes (printed, no plot needed) ------------------
    print_temporal_extremes(df1, farm_name="N-10.1", top_n=5)
    print_temporal_extremes(df2, farm_name="N-10.2", top_n=5)

    # ---- Directional analysis -----------------------------------------
    plot_directional_yield_comparison(df1, df2)

    # ---- Seasonality --------------------------------------------------
    plot_seasonality_comparison(df1, df2)

    # ---- Diurnal ------------------------------------------------------
    plot_diurnal_comparison(df1, df2)

    # ---- Per-layout dashboards ----------------------------------------
    plot_full_dashboard(df1, farm_name="N-10.1", color=C1)
    plot_full_dashboard(df2, farm_name="N-10.2", color=C2)

    print("✓ All analyses complete.")