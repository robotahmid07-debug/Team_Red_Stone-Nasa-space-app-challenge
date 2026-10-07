"""
SAR-Tensor: Multi-Temporal Strain & Compound Geohazard Predictor
NASA Space Apps Challenge - "Dancing with the SARs" (Advanced Track)

Pipeline: earthaccess/CMR search (3 retries) -> displacement field -> 2D horizontal
strain tensor (finite differences) -> gamma_max, dilatation, vulnerability index -> maps.
If anything in the live path fails, the app falls back to synthetic geodetic fields.
"""
import os
import time
import numpy as np
import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from scipy.ndimage import gaussian_filter
from matplotlib import colormaps

st.set_page_config(page_title="SAR-Tensor", page_icon="🛰️", layout="wide")

# ----------------------------------------------------------------------------
# Case studies: centre (lat, lon), half-extent (deg), CMR search keyword set
# ----------------------------------------------------------------------------
CASES = {
    "Urban Subsidence (Mexico City)": dict(lat=19.43, lon=-99.13, ext=0.12, kind="urban"),
    "Volcanic Unrest (Kilauea, Hawaii)": dict(lat=19.41, lon=-155.28, ext=0.10, kind="volcano"),
    "Delta Collapse (Mekong Delta)": dict(lat=10.00, lon=105.80, ext=0.15, kind="delta"),
}
N = 140                      # grid size (pixels per side)
THETA = np.deg2rad(39.0)     # radar incidence angle
HEAD = np.deg2rad(-10.0)     # approximate heading (ascending, right-looking)


# ----------------------------------------------------------------------------
# 1. DATA ACQUISITION (live NASA with retry + guaranteed fallback)
# ----------------------------------------------------------------------------
def try_live_nasa(case, retries=3):
    """Search CMR via earthaccess for OPERA Sentinel-1 displacement products and
    attempt to read one granule. Returns (field_dict | None, message)."""
    token = os.environ.get("EARTHDATA_TOKEN")
    if not token:
        try:
            token = st.secrets["EARTHDATA_TOKEN"]
        except Exception:  # no secrets file configured
            token = None
    if token:
        os.environ["EARTHDATA_TOKEN"] = token
    last_err = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            import earthaccess
            earthaccess.login(strategy="environment")  # needs EARTHDATA_TOKEN or user/pass env vars
            bbox = (case["lon"] - case["ext"], case["lat"] - case["ext"],
                    case["lon"] + case["ext"], case["lat"] + case["ext"])
            results = earthaccess.search_data(short_name="OPERA_L3_DISP-S1_V1",
                                              bounding_box=bbox, count=5)
            if not results:
                results = earthaccess.search_data(short_name="NISAR_L2_GUNW_BETA_V1",
                                                  bounding_box=bbox, count=5)
            if not results:
                raise RuntimeError("CMR returned 0 granules for this AOI")
            field = read_granule(earthaccess, results[0], case)
            field["note"] = f"{len(results)} granule(s) found in CMR; parsed 1."
            return field, field["note"]
        except Exception as e:  # noqa: BLE001 - any failure triggers retry/fallback
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.5 * attempt)
    return None, last_err


def read_granule(earthaccess, granule, case):
    """Best-effort read of a displacement raster (HDF5/NetCDF) -> LOS velocity grid (mm/yr)."""
    import h5py
    files = earthaccess.open([granule])
    with h5py.File(files[0], "r") as f:
        key = next(k for k in ("displacement", "unwrapped_phase") if k in f)
        arr = f[key]
        step = max(1, int(max(arr.shape) // N))
        d = np.array(arr[::step, ::step], dtype="float64")[:N, :N]
    d = np.where(np.isfinite(d), d, np.nanmedian(d))
    d = d * 1000.0 if np.nanmax(np.abs(d)) < 5 else d      # metres -> mm heuristic
    v = d / 0.5                                            # ~6-month baseline -> mm/yr (approx.)
    return dict(los=v, ue=None, un=None, dx=30.0 * step, dy=30.0 * step)


def synthetic_field(case, seed=7):
    """Pre-calculated synthetic geodetic fields (mm/yr): E, N, U components + LOS projection."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(float)
    x, y = (x - N / 2) / N, (N / 2 - y) / N          # north-up, normalised [-0.5, 0.5]
    kind = case["kind"]
    if kind == "urban":      # Gaussian subsidence bowls + inward horizontal flow
        g = np.exp(-((x + .08) ** 2 + (y - .05) ** 2) / (2 * .09 ** 2)) \
            + .6 * np.exp(-((x - .18) ** 2 + (y + .15) ** 2) / (2 * .06 ** 2))
        uz = -45 * g
        ue = 18 * (x + .08) / .09 * np.exp(-((x + .08) ** 2 + (y - .05) ** 2) / (2 * .09 ** 2))
        un = 18 * (y - .05) / .09 * np.exp(-((x + .08) ** 2 + (y - .05) ** 2) / (2 * .09 ** 2))
        ue, un = -ue, -un                                  # inward
    elif kind == "volcano":  # Mogi point source: uplift + radial outward
        r2 = x ** 2 + y ** 2
        s = (r2 + .12 ** 2) ** -1.5 * .12 ** 3
        uz = 60 * s
        ue, un = 40 * x / .12 * s, 40 * y / .12 * s
    else:                    # delta: seaward-increasing compaction + patchy hotspots
        uz = -25 * (1 / (1 + np.exp(-6 * (-y)))) - 12 * np.exp(-((x - .2) ** 2 + (y + .2) ** 2) / .01)
        ue = 6 * np.sin(6 * x) * (0.5 - y)
        un = -8 * (1 / (1 + np.exp(-6 * (-y))))
    noise = lambda s: gaussian_filter(rng.normal(0, s, (N, N)), 2)
    uz, ue, un = uz + noise(1.5), ue + noise(.8), un + noise(.8)
    los = uz * np.cos(THETA) + np.sin(THETA) * (-ue * np.cos(HEAD) + un * np.sin(HEAD))
    px = 2 * case["ext"] * 111_320 * np.cos(np.deg2rad(case["lat"])) / N
    py = 2 * case["ext"] * 110_574 / N
    return dict(los=los, ue=ue, un=un, uz=uz, dx=px, dy=py)


@st.cache_data(show_spinner=False)
def acquire(case_name, use_live):
    case = CASES[case_name]
    if use_live:
        field, msg = try_live_nasa(case)
        if field is not None:
            return field, "live", msg
        return synthetic_field(case), "sim", f"Live fetch failed -> fallback. ({msg})"
    return synthetic_field(case), "sim", "Live mode disabled by user."


# ----------------------------------------------------------------------------
# 2. GEODETIC ENGINE: strain tensor, gamma_max, vulnerability
# ----------------------------------------------------------------------------
def strain_tensor(ue, un, dx, dy, sigma=1.5):
    """2D horizontal strain from displacement (mm) on a north-up grid -> microstrain."""
    ue, un = gaussian_filter(ue, sigma), gaussian_filter(un, sigma)   # suppress noise before differentiating
    due_dy_rows, due_dx = np.gradient(ue, dy, dx)
    dun_dy_rows, dun_dx = np.gradient(un, dy, dx)
    due_dn, dun_dn = -due_dy_rows, -dun_dy_rows                       # rows increase southward
    k = 1e-3 * 1e6                                                    # mm/m -> microstrain
    exx, eyy = due_dx * k, dun_dn * k
    gxy = (due_dn + dun_dx) * k                                       # engineering shear strain
    return exx, eyy, gxy


def derived_metrics(exx, eyy, gxy):
    gmax = np.sqrt((exx - eyy) ** 2 + gxy ** 2)                       # = 2 * max tensor shear
    dil = exx + eyy
    e1 = .5 * (exx + eyy) + .5 * gmax
    e2 = .5 * (exx + eyy) - .5 * gmax
    nz = lambda a: np.clip(a / (np.percentile(a, 99) + 1e-9), 0, 1)
    vi = 100 * (0.5 * nz(gmax) + 0.3 * nz(np.abs(dil)) + 0.2 * nz(np.abs(e1 - e2) / 2))
    return gmax, dil, e1, e2, vi


def risk_label(vi_peak):
    return ("CRITICAL", "#d62728") if vi_peak > 75 else ("HIGH", "#ff7f0e") if vi_peak > 50 \
        else ("MODERATE", "#e6b800") if vi_peak > 25 else ("LOW", "#2ca02c")


def evolve(field, t_months):
    """Cumulative displacement at t months: linear trend + seasonal term (mm)."""
    f = t_months / 12.0
    seas = 1 + 0.12 * np.sin(2 * np.pi * t_months / 12.0)
    return f * seas


# ----------------------------------------------------------------------------
# 3. MAP RENDERING
# ----------------------------------------------------------------------------
def to_rgba(a, cmap, sym=False):
    lim = np.percentile(np.abs(a), 98) + 1e-9
    n = (a / lim + 1) / 2 if sym else a / (np.percentile(a, 98) + 1e-9)
    rgba = colormaps[cmap](np.clip(n, 0, 1))
    return rgba


def build_map(case, layer_arr, cmap, sym):
    e = case["ext"]
    bounds = [[case["lat"] - e, case["lon"] - e], [case["lat"] + e, case["lon"] + e]]
    m = folium.Map(location=[case["lat"], case["lon"]], zoom_start=11, tiles="OpenStreetMap")
    ImageOverlay(image=to_rgba(layer_arr, cmap, sym), bounds=bounds, opacity=0.72,
                 interactive=False, zindex=2).add_to(m)
    peak = np.unravel_index(np.argmax(np.abs(layer_arr)), layer_arr.shape)
    plat = case["lat"] + e - (peak[0] + .5) / N * 2 * e
    plon = case["lon"] - e + (peak[1] + .5) / N * 2 * e
    folium.Marker([plat, plon], tooltip="Peak signal", icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)
    return m


# ----------------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------------
st.markdown("""<style>.badge{padding:6px 14px;border-radius:20px;font-weight:700;color:white;display:inline-block}
.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:14px 18px}
.card h4{margin:0;color:#9ca3af;font-size:.8rem}.card p{margin:2px 0 0;font-size:1.7rem;font-weight:700;color:#f9fafb}</style>""",
            unsafe_allow_html=True)
st.title("🛰️ SAR-Tensor")
st.caption("Multi-Temporal Strain & Compound Geohazard Predictor · NASA Space Apps · Dancing with the SARs")

with st.sidebar:
    st.header("Controls")
    case_name = st.selectbox("Case study", list(CASES))
    use_live = st.toggle("Try live NASA Earthdata", value=False,
                         help="Needs EARTHDATA_TOKEN env var / Colab secret. Falls back automatically.")
    t = st.slider("Temporal scrubber (months)", 1, 36, 12)
    layer = st.radio("Map layer", ["Displacement (LOS, mm)", "Max shear strain γmax (µε)",
                                   "Vulnerability index (0-100)"])
    sigma = st.slider("Pre-smoothing σ (px)", 0.5, 4.0, 1.5, 0.5)

case = CASES[case_name]
with st.spinner("Acquiring SAR displacement data..."):
    field, mode, msg = acquire(case_name, use_live)

if mode == "live":
    st.markdown('<span class="badge" style="background:#16a34a">● Live NASA Cloud Data</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="badge" style="background:#d97706">● Simulated High-Resolution Geodetic Test Mode</span>',
                unsafe_allow_html=True)
st.caption(msg)

# --- compute fields at time t -------------------------------------------------
scale = evolve(field, t)
los_cum = field["los"] * scale
if field["ue"] is not None:
    ue, un = field["ue"] * scale, field["un"] * scale
else:   # LOS-only (live) proxy: horizontal field estimated from LOS along look direction
    ue, un = los_cum * np.cos(HEAD) / np.tan(THETA), los_cum * np.sin(HEAD) / np.tan(THETA)
exx, eyy, gxy = strain_tensor(ue, un, field["dx"], field["dy"], sigma)
gmax, dil, e1, e2, vi = derived_metrics(exx, eyy, gxy)

label, color = risk_label(np.percentile(vi, 99.5))
c1, c2, c3, c4 = st.columns(4)
for col, title, val in [(c1, "Peak LOS velocity", f"{np.max(np.abs(field['los'])):.1f} mm/yr"),
                        (c2, f"Peak γmax @ {t} mo", f"{gmax.max():.0f} µε"),
                        (c3, "Mean dilatation", f"{dil.mean():+.1f} µε"),
                        (c4, "Risk status", f"<span style='color:{color}'>{label}</span>")]:
    col.markdown(f"<div class='card'><h4>{title}</h4><p>{val}</p></div>", unsafe_allow_html=True)

arr, cmap, sym = {"Displacement (LOS, mm)": (los_cum, "RdBu", True),
                  "Max shear strain γmax (µε)": (gmax, "inferno", False),
                  "Vulnerability index (0-100)": (vi, "YlOrRd", False)}[layer]
st_folium(build_map(case, arr, cmap, sym), height=520, use_container_width=True, returned_objects=[])

# --- time series of peak metrics ----------------------------------------------
import matplotlib.pyplot as plt
ts = np.arange(1, 37)
pk = [np.max(np.abs(field["los"])) * evolve(field, m) for m in ts]
fig, ax = plt.subplots(figsize=(8, 2.6), facecolor="#0e1117")
ax.plot(ts, pk, color="#38bdf8"); ax.axvline(t, color="#f43f5e", ls="--")
ax.set_facecolor("#0e1117"); ax.tick_params(colors="w"); ax.set_xlabel("Months", color="w")
ax.set_ylabel("Peak |LOS| (mm)", color="w"); [s.set_color("#444") for s in ax.spines.values()]
st.pyplot(fig)

with st.expander("📐 Scientific Methodology & Math"):
    st.markdown(r"""
**1. Observation model.** LOS displacement from the 3D vector $(u_e,u_n,u_z)$:
$d_{LOS}=u_z\cos\theta-\sin\theta\,(u_e\cos\alpha-u_n\sin\alpha)$, with incidence $\theta=39^\circ$ and heading $\alpha$.
Phase relates to LOS by $d=-\frac{\lambda}{4\pi}\phi$ (Sentinel-1: $\lambda=5.55$ cm).

**2. Strain tensor** (small-strain, plane, central finite differences after Gaussian pre-smoothing):
$$\epsilon_{xx}=\frac{\partial u_e}{\partial x},\quad \epsilon_{yy}=\frac{\partial u_n}{\partial y},\quad
\gamma_{xy}=\frac{\partial u_e}{\partial y}+\frac{\partial u_n}{\partial x}$$
**3. Invariants.** Principal strains $\epsilon_{1,2}=\frac{\epsilon_{xx}+\epsilon_{yy}}{2}\pm\frac{1}{2}\sqrt{(\epsilon_{xx}-\epsilon_{yy})^2+\gamma_{xy}^2}$,
maximum engineering shear $\gamma_{max}=\sqrt{(\epsilon_{xx}-\epsilon_{yy})^2+\gamma_{xy}^2}$, dilatation $\Delta=\epsilon_{xx}+\epsilon_{yy}$.

**4. Subsurface Structural Vulnerability Index (SSVI).**
$SSVI=100\,[0.5\,\hat\gamma_{max}+0.3\,|\hat\Delta|+0.2\,\hat\tau]$, where $\hat{\cdot}$ is normalisation by the 99th percentile
(clipped to 1) and $\tau=(\epsilon_1-\epsilon_2)/2$. Risk bands: <25 Low, 25-50 Moderate, 50-75 High, >75 Critical.
This is a screening heuristic, not a calibrated engineering threshold.

**5. Assumptions & limits.** In *Live* mode only LOS is observed, so horizontal components are approximated by projecting LOS
along the look direction (single geometry); true $u_e,u_n$ need ascending+descending pairs. Live granule parsing is best-effort
(OPERA DISP-S1 / NISAR GUNW formats); any failure triggers the simulated fallback. Simulated mode uses Gaussian-bowl, Mogi-source
and delta-compaction models plus spatially correlated noise.
""")
