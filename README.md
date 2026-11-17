# NN TPC1

Code for Homework 1 of Nanotechnologies and Nanoelectronics at IST, 2026/27: "Superhero Nanotechnology". Inspired by the film Clayface (2026) and the titular character's regenerative powers, our report asks how fast a palm of skin can regrow before the heat from building it damages the tissue.

## What the code does

`nn_tpc1.py` prints:
- the mass of the palm wound, and the heat and power to build it
- the temperature rise if no heat escapes
- the shortest safe build time for each wound depth, at three blood flows in the new tissue: 
    - none: 0 ml/min/kg (no working blood vessels)
    - resting: 106 ml/min/kg (normal skin circulation, source: IT'IS)
    - raised: 1000 ml/min/kg (vessels open fully due to the heated skin, source: Gowrishankar)
- the shortest safe build time for the full skin, at two lower heat fractions

It also draws the report's plot, `figures/budget-plot.pdf`. The plot shows the peak temperature of the new tissue against the time to regrow. A red line shows the CEM43 safe limit. Panel (a) shows the full range on log axes, and panel (b) shows a close-up where the curves cross the line. Each dot marks the shortest safe build time for one depth. The curves and dots use the resting blood flow and the full heat fraction (η = 1). The other flows and heat fractions appear only in the printed output.

The model uses three wound depths, each from a source:

| Depth | Tissue removed | Source |
|---|---|---|
| 0.49 mm | epidermis and papillary dermis | Dande et al. (2023), Table 1 |
| 0.92 mm | full skin | Dande et al. (2023), Table 1, layers summed |
| 1.25 mm | skin and the fat layer | Dande et al. (2023); Störchle et al. (2018), Table 5 |

All new tissue is approximated to have the properties of skin, because the energy value the model uses is for lean tissue and no value currently exists for fat. Below each wound, the palm keeps its own remaining layers.

## How to run it

```
python3 nn_tpc1.py
```

Python 3 with NumPy, SciPy and Matplotlib 3.5 or later is required. A full run takes about one minute.

## From the equations to the code

Our report gives the model as two equations: the Pennes bioheat equation and the CEM43 thermal dose. The code solves both on a grid. It divides the palm into cells about 25 µm thick, about 800 cells down to 2 cm, and keeps one temperature per cell. Each derivative becomes a difference between neighbor cells or between time steps.

The bioheat equation is:

$$\rho c\,\frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left(\kappa\frac{\partial T}{\partial z}\right) + w\rho_b c_b\,(T_a - T) + q$$

Multiply each term by the cell thickness $\Delta z$. Here $g$ is the heat conductance between neighbor cells, and $n$ counts the time steps. For cell $i$, the terms become:

| Report term | Code term (cell $i$) | Change |
|---|---|---|
| $\rho c\,\partial T/\partial t$ | $\rho c\,\Delta z\,(T_i^{n+1} - T_i^{n})/\Delta t$ | the time derivative becomes a step |
| $\partial_z(\kappa\,\partial_z T)$ | $g_{i-1}(T_{i-1} - T_i) + g_i(T_{i+1} - T_i)$ | the space derivative becomes differences between neighbor cells |
| $w\rho_b c_b\,(T_a - T)$ | $\beta\,\Delta z\,(T_a - T_i)$, with $\beta = w\rho_b c_b$ | only the unit of $w$: IT'IS gives flow in ml/min/kg (`blood_coef`) |
| $q$ | $q\,\Delta z$ | none |

In one uniform tissue, $g = \kappa/\Delta z$. The conduction term is then the standard finite-difference second derivative, $\Delta z\cdot\kappa\,(T_{i-1} - 2T_i + T_{i+1})/\Delta z^2$. The code calculates $g$ as two half-cells in series, $g_i = 1/\big(\Delta z_i/2\kappa_i + \Delta z_{i+1}/2\kappa_{i+1}\big)$. This gives the correct heat flow where $\kappa$ changes, at the skin, fat and muscle boundaries.

### Boundaries

The surface loses heat to air at 25 °C through the coefficient $h$. The bottom of the slab, 2 cm below the surface, stays at 37 °C.

### Time steps

The code uses backward Euler (implicit) steps: it takes the conduction, blood, boundary and heat-source terms at the new time $n+1$. Each step is then one tridiagonal linear system, which `scipy.linalg.solve_banded` solves. An implicit step is stable at any step size, so 600 steps per phase work for builds from seconds to weeks. A grid twice as fine, with four times as many steps, changes no safe build time by more than 0.002 h.

### Thermal dose

Our report's dose, $\text{CEM}_{43} = \sum \Delta t\, R^{\,43-T}$, is one line in `regrow`:

```python
dose += (dose_rate(T) - rest_rate) * dt / 60.0
```

The line adds one term per time step, in every cell. It subtracts the dose that tissue at rest gets anyway. 

The red safe line is the same dose for one constant temperature held for the full build, with $t$ in minutes:

- builds shorter than 21 minutes: $T = 43 + \log_2(21/t)$
- longer builds: $T = 43 - \log_4(t/21)$

## Inputs

Two inputs are our own assumptions: the 10 s length of the film's rapid "superpowered" regrowth scene, and that the energy expended to build tissue becomes heat in the wound. The heat fraction η scales that heat. Every other input has a source:

| Input | Value | Source |
|---|---|---|
| Tissue properties (skin, fat, muscle, blood) | see `nn_tpc1.py` | IT'IS database v5.0 |
| Palm area | 92.5 cm² (0.5% of a 1.85 m² body) | Rhodes et al. (2013); Sender et al. (2016) |
| Palm skin depth | 0.924 mm (epidermis and papillary dermis: 0.49 mm) | Dande et al. (2023) |
| Fat layer under the skin (hand mean) | 0.33 mm | Störchle et al. (2018) |
| Deep boundary | 37 °C, 2 cm below the surface | Gowrishankar et al. (2004) |
| Energy expended per gram of lean tissue built | 13.9 kJ/g | Obikawa et al. (2025) |
| Palm surface at rest (seated subjects, hand not immersed) | 32.5 °C | Norrbrand et al. (2017) |
| Room air | 25 °C | Norrbrand et al. (2017) |
| Raised skin blood flow | 1000 ml/min/kg, about 9 times the resting flow (the source's 15-fold rise starts from its own baseline) | Gowrishankar et al. (2004) |
| Damage dose limit | 21 CEM43 min | Yarmolenko et al. (2011) |
| Bioheat equation | | Pennes (1948) |

The code calibrates the surface heat-loss coefficient $h$ so that the palm surface at rest is at the measured 32.5 °C.