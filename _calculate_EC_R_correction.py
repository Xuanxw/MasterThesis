import os
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
Inputs:
Refraction correction (*delta ro):
@ param Z0: flying height above sea level [km]
@ param Z: height of the ground [km]
@ param c: pricipal distance(distance from perspective center to image center)
@ param ro: distance between image points to image center

Earth curvature correction (*delta ro):
@ param ro: radial distance from principal point
@ param h: flying height above ground
@ param R: radius of the Earth = 6370 km
@ param c: principal distance

"""
# GSD: 10M -> Flight Height: ca. 2400 m
# Image extent: (34.008, 52.920) mm
# PPA : (-0.08, 0) mm
Z = 0.020  # [km]
h = 2.400  # [km]
Z0 = h + Z  # [km]
R = 6370  # [km]
c = 100.5  # [mm]
K = 0.00241 * (Z0 / (Z0**2 - 6 * Z0 + 250) - Z**2 / Z0 / (Z**2 - 6 * Z + 250))
ro_max = math.sqrt((34.008 + 0.08)**2 + 52.920**2)
ro_range = np.arange(0, ro_max)
Refra_corr = ro_range * K * (1 + ro_range**2 / c**2)  # mm
print(Refra_corr)
EC_corr = ro_range ** 3 * h / 2 / R / c**2  # mm
print(EC_corr)

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=["Refraction correction[mm]",
                                    "Earth curvature correction [mm]"],
                    shared_xaxes=True)
fig.add_trace(go.Scatter(
    mode='lines',
    x=ro_range,
    y=Refra_corr,
    legendgroup="Refraction correction",
    showlegend=True,
    name="Refraction correction"
),
    row=1,
    col=1
)
fig.add_trace(go.Scatter(
    mode='lines',
    x=ro_range,
    y=EC_corr,
    legendgroup="Refraction correction",
    showlegend=True,
    name="Refraction correction"
),
    row=2,
    col=1
)

fig.update_xaxes(title_text="Radial distance [mm]", row=2, col=1)

fig.update_layout(
    xaxis=dict(
        showexponent='all',
        exponentformat='e'
    )
)
fig.show()
