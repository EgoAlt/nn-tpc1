
"""
Heat budget for regrowing one palm of skin (NN TPC 1)

How fast can a full-thickness palm wound regrow before the heat released 
by building the new tissue damages it? This script:
    1. prints the simple numbers the report quotes (mass, heat, power, no-cooling warming),
    2. solves the 1D, time-dependent Pennes bioheat equation through the palm,
    3. scores damage with the CEM43 thermal dose and finds the shortest safe build time,
    4. draws the report's one figure: the new tissue's peak temperature against build time.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_banded

# INPUTS-------------------------------------------------------------------------------------

# Tissue properties from IT'IS database v5.0: conductivity k (W/m/K), density rho (kg/m^3)
# heat capacity c (J/kg/K), resting blood flow (ml of blood per minute per kg)
SKIN = {"k": 0.37, "rho": 1109.0, "c": 3391.0, "flow": 106.0}
FAT = {"k": 0.21, "rho": 911.0, "c": 2348.0, "flow": 33.0}
MUSCLE = {"k": 0.49, "rho": 1090.0, "c": 3421.0, "flow": 37.0}
BLOOD_RHO, BLOOD_C = 1050.0, 3617 # IT'IS

PALM_AREA = 0.005 * 1.85            # m^2: 0.5% of a 1.85 m^2 body (Rhodes 2013; Sender 2016)
SKIN_DEPTH = 0.924e-3               # m: palm epidermis + dermis, summed (Dande 2023, Table 1)
UPPER_SKIN = (350.5 + 139.3) * 1e-6 # m: epidermis + papillary dermis only (Dande 2023, Table 1)
FAT_DEPTH = 0.33e-3                 # m: hand fat layer (Stoerchle 2018, Table 5)
DEEP_BOUNDARY = 20e-3               # m: held at 37 C, 2 cm below the surface (Gowrishankar 2004)


# Energy expended per kg of lean tissue built (Obikawa 2025)
# Obikawa says "expended", that it is released as heat in the wound is our assumption
Q_SYN = 13.9e6          # J/kg
T_ARTERIAL = 37.0       # C, arterial blood and deep boundary (Gowrishankar 2004)
T_AIR = 25.0            # C, room air (Norrbrand 2017's lab)
T_PALM_REST = 32.5      # C, palm surface of seated subjects, non-immersed hand (Norrbrand, 2017, p. 1147)
RAISED_FLOW = 1000.0    # ml/min/kg: skin flow rises 15-fold to 100 ml per 100 g per min (Gowrishankar 2004, p. 2)
DOSE_LIMIT = 21.0       # CEM43 minutes, low end of skin damage (Yarmolenko 2011)

GRID = 25e-6                            # m, spacing of the depth grid
STEPS = 600                             # for the build, and again for the cool-down
SCENE = 10.0                            # s, chosen "sci-fi" regrowth movie scene
SHORTEST, LONGEST = 1.0, 60 * 86400.0   # s, range searched for the safe build time


# The three wounds the report compares: depth, line style, what the depth removes
WOUNDS = [
    (UPPER_SKIN, ":", "epidermis + papillary dermis"),
    (SKIN_DEPTH, "-", "full skin"),
    (SKIN_DEPTH + FAT_DEPTH, "--", "skin + fat layer"),
]

# TISSUE-------------------------------------------------------------------------------------
# Pennes blood term w*rho_b*c_b in W/m^3/K, from a flow in ml/min/kg
def blood_coef(flow, rho):
    w = flow * 1e-6 / 60.0 * rho    # m^3 of blood per second per m^3 of tissue
    return w * BLOOD_RHO * BLOOD_C


# Layers from the surface down. The wound's top wound_depth is new tissue,
# all given skin properties: Obikawa's 13.9 kJ/g is for lean tissue.
# Below the wound, the palm keeps its own skin, fat and muscle.
def palm_layers(wound_depth, new_flow):
    palm = [(SKIN, SKIN_DEPTH), (FAT, FAT_DEPTH), 
            (MUSCLE, DEEP_BOUNDARY - SKIN_DEPTH - FAT_DEPTH)]
    layers = [(SKIN, wound_depth, new_flow, True)]
    top = 0.0 # depth where the current palm layer starts
    for tissue, thickness in palm:
        bottom = top + thickness
        remaining = bottom - max(top, wound_depth) # part of this layer below the wound
        if remaining > 1e-9:
            layers.append((tissue, remaining, tissue["flow"], False))
        top = bottom
    return layers


# Cells from the surface down: new tissue, then what remains of the palm
def make_slab(wound_depth=SKIN_DEPTH, new_flow=SKIN["flow"]):
    layers = palm_layers(wound_depth, new_flow)
    k, rho, c, beta, dz, heated = [], [], [], [], [], []
    for tissue, depth, flow, is_new_skin in layers:
        n = max(1, round(depth / GRID)) # Number of cells in this layer
        k += [tissue["k"]] * n
        rho  += [tissue["rho"]] * n
        c += [tissue["c"]] * n
        beta += [blood_coef(flow, tissue["rho"])] * n
        dz += [depth / n] * n
        heated += [is_new_skin] * n
    
    slab = {
        "k": np.array(k),
        "rho": np.array(rho),
        "c": np.array(c),
        "beta": np.array(beta),
        "dz": np.array(dz),
        "heated": np.array(heated),
    }

    # Conductance between neighboring cells (W/m^2/K): two half-cells in series
    half_above = slab["dz"][:-1] / (2 * slab["k"][:-1])
    half_below = slab["dz"][1:] / (2 * slab["k"][1:])
    slab["g"] = 1.0 / (half_above + half_below)
    slab["g_deep"] = 2 * slab["k"][-1] / slab["dz"][-1]
    return slab


# Air film h in series with half of the top cell
def surface_conductance(slab, h):
    return 1.0 / (1.0 / h + slab["dz"][0] / (2 * slab["k"][0]))


# Banded matrix of conduction, blood and boundary terms (plus storage for a time step)
def heat_matrix(slab, h, storage=None):
    diag = slab["beta"] * slab["dz"]
    diag[:-1] += slab["g"]
    diag[1:] += slab["g"]
    diag[0] += surface_conductance(slab, h)
    diag[-1] += slab["g_deep"]
    if storage is not None:
        diag = diag + storage
    bands = np.zeros((3, len(diag)))
    bands[0, 1:] = -slab["g"]
    bands[1] = diag
    bands[2, :-1] = -slab["g"]
    return bands


# Heat flowing in from arterial blood, the air and the deep boundary
def fixed_sources(slab, h):
    rhs = slab["beta"] * slab["dz"] * T_ARTERIAL
    rhs[0] += surface_conductance(slab, h) * T_AIR
    rhs[-1] += slab["g_deep"] * T_ARTERIAL
    return rhs


# Steady temperature through the palm with no regrowth
def resting_profile(slab, h):
    return solve_banded((1, 1), heat_matrix(slab, h), fixed_sources(slab, h))


# Skin surface temperature, between the top cell and the air
def surface_temperature(slab, h, T):
    return T_AIR + surface_conductance(slab, h) * (T[0] - T_AIR) / h


# Air coef h that puts the palm surface at rest at measured T_PALM_REST
def calibrate_h(skin_depth=SKIN_DEPTH):
    low, high = 1.0, 200.0
    for _ in range(60):
        h = 0.5 * (low + high)
        slab = make_slab(skin_depth)
        if surface_temperature(slab, h, resting_profile(slab, h)) > T_PALM_REST:
            low = h
        else:
            high = h
    return 0.5 * (low + high)


# DAMAGE-------------------------------------------------------------------------------------
# CEM43 minutes per minute at temperature T (Yarmolenko 2011, Eq. 1)
def dose_rate(T):
    T = np.minimum(T, 100.0) # past boiling the tissue is gone (also avoids overflow)
    R = np.where(T < 43.0, 0.25, 0.5)
    return R ** (43.0 - T)


# Temperature that, held for the whole build, gives exactly the dose limit
def safe_line(t_build):
    minutes = t_build / 60.0
    if minutes < DOSE_LIMIT:
        return 43.0 + math.log2(DOSE_LIMIT / minutes)
    return 43.0 - math.log(minutes / DOSE_LIMIT, 4)


# ONE REGROWTH-------------------------------------------------------------------------------
# Build the new skin in t_build seconds, then let it cool
# Returns the peak temperature of the new tissue (C) and the worst extra thermal dose (CEM43 min)
def regrow(slab, h, t_build, eta=1.0):
    q = eta * SKIN["rho"] * Q_SYN / t_build # W/m^3 released in the new skin
    source = np.where(slab["heated"], q, 0.0) * slab["dz"]
    storage = slab["rho"] * slab["c"] * slab["dz"] # J/m^2/K held by each cell
    T_rest = resting_profile(slab, h)
    rest_rate = dose_rate(T_rest)
    T, dose, peak = T_rest.copy(), np.zeros_like(T_rest), 0.0
    t_cool = max(t_build, 3 * 3600.0)
    for duration, heat in ((t_build, source), (t_cool, 0.0 * source)):
        dt = duration / STEPS
        bands = heat_matrix(slab, h, storage / dt)
        rhs_fixed = fixed_sources(slab, h) + heat
        for _ in range(STEPS): # implicit (backward Euler) steps
           T = solve_banded((1, 1), bands, rhs_fixed + storage / dt * T)
           dose += (dose_rate(T) - rest_rate) * dt / 60.0 # only dose the build adds
           peak = max(peak, T[slab["heated"]].max()) # hottest point of the new tissue
    return peak, dose.max()


# Shortest build time (s) that keeps every depth under the dose limit
def safe_build_time(slab, h, eta=1.0):
    if regrow(slab, h, LONGEST, eta)[1] > DOSE_LIMIT:
        return math.inf # unsafe even at the longest build
    low, high = math.log(SHORTEST), math.log(LONGEST)
    for _ in range(30): # bisection in log time
        mid = 0.5 * (low + high)
        if regrow(slab, h, math.exp(mid), eta)[1] > DOSE_LIMIT:
            low = mid
        else:
            high = mid
    return math.exp(high)



# OUTPUTS------------------------------------------------------------------------------------
def print_simple_numbers():
    mass = PALM_AREA * SKIN_DEPTH * SKIN["rho"]
    heat = mass * Q_SYN
    print(f"Palm wound: {PALM_AREA * 1e4:.1f} cm^2 x {SKIN_DEPTH * 1e3:.3f} mm = {mass * 1e3:.2f} g")
    print(f"Heat to build it (eta = 1): {heat:.3g} J")
    print(f"Power in a {SCENE:.0f} s scene: {heat / SCENE / 1e3:.1f} kW")
    print(f"Warming if no heat escaped: {Q_SYN / SKIN['c']:.0f} K\n")

def print_safe_times(h):
    print(f"Shortest safe build time (worst depth at or under {DOSE_LIMIT:.0f} CEM43 min):")
    for depth, _, removed in WOUNDS:
        hours = []
        for flow in (0.0, SKIN["flow"], RAISED_FLOW): # none, resting, raised
            hours.append(safe_build_time(make_slab(depth, flow), h) / 3600)
        print(f"  {depth * 1e3:.2f} mm ({removed}):", end=" ")
        print(f"no flow {hours[0]:.2f} h, resting {hours[1]:.2f} h, raised {hours[2]:.2f} h")
    for eta in (0.5, 0.1):
        t = safe_build_time(make_slab(), h, eta)
        orders = math.log10(t / SCENE)
        print(f"  {SKIN_DEPTH * 1e3:.2f} mm, resting flow, eta = {eta}: {t / 3600:.2f} h", end=" ")
        print(f"({orders:.1f} orders above {SCENE:.0f} s)")
    print()


# Peak temperature of the new tissue vs. build time, with the safe line:
# the temperature that, held for the whole build, reaches the dose limit
# Left: the whole range on log axes
# Right: a close-up of where the curves cross the line
def draw_plot(h, path):
    week = 7 * 86400.0
    times = np.logspace(math.log10(SCENE), math.log10(week), 60)
    safe = [safe_line(t) for t in times]
    with plt.rc_context({"font.size": 10}): # two square panels across the full page width
        fig, (ax, zoom) = plt.subplots(1, 2, figsize=(7.0, 4.0))
        notes = [] # lines of the close-up's value box
        for depth, style, removed in WOUNDS:
            slab = make_slab(depth)
            peaks = [regrow(slab, h, t)[0] for t in times]
            t_safe = safe_build_time(slab, h) # the dot: this depth's shortest safe build
            peak_safe = regrow(slab, h, t_safe)[0] # the peak at that build, just above the line
            notes.append(f"{depth * 1e3:.2f} mm: {t_safe / 3600:.1f} h, {peak_safe:.1f} °C")
            for axes in (ax, zoom):
                axes.plot(times, peaks, style, color="C0")
                axes.plot(t_safe, peak_safe, "o", color="C0", markersize=3)
            ax.plot([], [], style, color="C0", label=f"{depth * 1e3:.2f} mm: {removed}")
        for axes in (ax, zoom):
            axes.plot(times, safe, color="C3", linewidth=1)
        ax.plot([], [], color="C3", linewidth=1, label=f"safe line ({DOSE_LIMIT:.0f} CEM43 min)")

        ticks = [10,  60, 3600, 86400, week]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(30, 5000)
        ax.set_yticks([30, 50, 100, 1000, 3000], ["30", "50", "100", "1000", "3000"])
        ax.set_xticks(ticks, ["10 s", "1 min", "1 h", "1 d", "1 wk"])
        ax.minorticks_off()
        ax.set_xlabel("time to regrow")
        ax.set_ylabel("peak temperature of new tissue (°C)")
        ax.set_title("(a) full range")

        zoom.set_xscale("log") # linear in temperature, to read the dots
        zoom.set_xlim(3600, week)
        zoom.set_ylim(32, 46)
        zoom.set_yticks([34, 38, 42, 46])
        zoom.set_xticks(ticks[2:], ["1 h", "1 d", "1 wk"])
        zoom.minorticks_off()
        zoom.set_xlabel("time to regrow")
        zoom.set_title("(b) close-up of the crossings")
        zoom.text(0.96, 0.96, "shortest safe build\n" + "\n".join(notes), transform=zoom.transAxes,
                  ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"))

        for axes in (ax, zoom):
            axes.set_box_aspect(1) # each panel a full square
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 2, 1, 3] # the legend fills by column, this reads by row
        fig.legend([handles[i] for i in order], [labels[i] for i in order], 
                   frameon=False, loc="lower center", ncol=2)
        fig.tight_layout(rect=(0, 0.12, 1, 1)) # leave room for the legend underneath
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path)
    print(f"Plot written to {path}")


def main():
    print_simple_numbers()
    h = calibrate_h()
    print(f"Air coefficient for a {T_PALM_REST} C palm at rest: h = {h:.1f} W/m^2/K\n")
    print_safe_times(h)
    draw_plot(h, Path(__file__).parent / "figures" / "budget-plot.pdf")

if __name__ == "__main__":
    main()
    