import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
from io import StringIO

st.set_page_config(page_title="President Forecast Demo", layout="wide")

# -----------------------------
# Sources
# -----------------------------
# MIT Election Data & Science Lab presidential returns (1976–2020),
# accessed via a public mirror (dataset card describes MIT compilation).
MIT_MIRROR_URL_DEFAULT = "https://huggingface.co/datasets/fdaudens/us-presidential-elections/resolve/main/1976-2020-president.csv"

# Electoral votes allocation (hardcoded): National Archives says these are
# effective for 2024 and 2028 elections, total 538, 270 to win.
# https://www.archives.gov/electoral-college/allocation
EV_2024 = {
    "AL":9,"AK":3,"AZ":11,"AR":6,"CA":54,"CO":10,"CT":7,"DE":3,"DC":3,"FL":30,"GA":16,"HI":4,"ID":4,"IL":19,"IN":11,
    "IA":6,"KS":6,"KY":8,"LA":8,"ME":4,"MD":10,"MA":11,"MI":15,"MN":10,"MS":6,"MO":10,"MT":4,"NE":5,"NV":6,"NH":4,
    "NJ":14,"NM":5,"NY":28,"NC":16,"ND":3,"OH":17,"OK":7,"OR":8,"PA":19,"RI":4,"SC":9,"SD":3,"TN":11,"TX":40,"UT":6,
    "VT":3,"VA":13,"WA":12,"WV":4,"WI":10,"WY":3
}

@st.cache_data
def load_csv(url: str) -> pd.DataFrame:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception:
        # Fallback: use the Harvard Dataverse DOI landing page as a “source of truth”
        # (You can manually swap to Dataverse download if HF changes again.)
        raise

st.title("President Forecast Demo (MIT baseline + Monte Carlo)")
st.caption(
    "Baseline results come from MIT Election Data & Science Lab presidential returns (1976–2020), "
    "loaded from a public mirror dataset. The forecast is a lightweight probabilistic model: "
    "baseline margin + user swing + correlated national error + state noise."
)

with st.sidebar:
    st.header("Data")
    mit_url = st.text_input("MIT presidential returns CSV URL", MIT_MIRROR_URL_DEFAULT)

    st.header("Model assumptions")
    baseline_year = st.selectbox(
        "Baseline year",
        [1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020],
        index=11
    )
    n_sims = st.slider("Simulations", 2000, 50000, 15000, step=1000)
    national_swing = st.slider("National swing (Dem − Rep, points)", -15.0, 15.0, 0.0, 0.5)
    nat_sigma = st.slider("National correlated error σ (points)", 0.0, 6.0, 1.5, 0.5)
    state_sigma = st.slider("State-specific error σ (points)", 1.0, 10.0, 4.0, 0.5)

    st.divider()
    st.header("Optional: upload polls")
    st.write("Upload a CSV with columns: `state_po` (2-letter) and `poll_margin` (Dem−Rep).")
    poll_file = st.file_uploader("Poll snapshot CSV", type=["csv"])

# -----------------------------
# Load MIT dataset
# -----------------------------
try:
    mit = load_csv(mit_url)
except Exception as e:
    st.error(f"Failed to load MIT data: {e}")
    st.stop()

# Normalize columns
mit.columns = [c.lower() for c in mit.columns]

required = {"year", "state_po", "party_simplified", "candidatevotes", "totalvotes"}
if not required.issubset(set(mit.columns)):
    st.error(
        "MIT dataset schema not recognized. Expected columns including: "
        "year, state_po, party_simplified, candidatevotes, totalvotes."
    )
    st.stop()

# -----------------------------
# Build baseline state margins from MIT returns
# -----------------------------
base = mit[mit["year"] == baseline_year].copy()
base = base[base["state_po"].notna()].copy()
base["state_po"] = base["state_po"].astype(str).str.upper().str.strip()

# Dem/Rep shares by state
grp = base.groupby(["state_po", "party_simplified"], as_index=False)[["candidatevotes", "totalvotes"]].sum()

# totalvotes repeats; take max per state for denominator
denom = base.groupby("state_po", as_index=False)["totalvotes"].max().rename(columns={"totalvotes": "den_total"})
grp = grp.merge(denom, on="state_po", how="left")
grp["share"] = grp["candidatevotes"] / grp["den_total"]

pivot = grp.pivot_table(index="state_po", columns="party_simplified", values="share", aggfunc="sum").fillna(0.0)

if "DEMOCRAT" not in pivot.columns or "REPUBLICAN" not in pivot.columns:
    st.error("Could not find DEMOCRAT and REPUBLICAN in party_simplified for the selected year.")
    st.stop()

pivot["baseline_margin"] = (pivot["DEMOCRAT"] - pivot["REPUBLICAN"]) * 100.0  # points
df = pivot[["baseline_margin"]].reset_index().rename(columns={"state_po": "state"})

# Add EV (2024/2028 allocation)
df["ev"] = df["state"].map(EV_2024).fillna(0).astype(int)

# -----------------------------
# Optional: polls override baseline
# -----------------------------
if poll_file is not None:
    polls = pd.read_csv(poll_file)
    polls.columns = [c.lower() for c in polls.columns]
    if not {"state_po", "poll_margin"}.issubset(set(polls.columns)):
        st.warning("Poll upload ignored (needs columns: state_po, poll_margin).")
        df["base_margin_for_model"] = df["baseline_margin"]
    else:
        polls["state_po"] = polls["state_po"].astype(str).str.upper().str.strip()
        polls = polls.rename(columns={"state_po": "state"})
        df = df.merge(polls[["state", "poll_margin"]], on="state", how="left")
        df["base_margin_for_model"] = df["poll_margin"].combine_first(df["baseline_margin"])
else:
    df["base_margin_for_model"] = df["baseline_margin"]

# Apply user national swing
df["model_margin"] = df["base_margin_for_model"] + national_swing

# -----------------------------
# Monte Carlo simulation
# -----------------------------
rng = np.random.default_rng(2024)
states = df["state"].tolist()
evs = df["ev"].to_numpy()
base_margin = df["model_margin"].to_numpy()

nat_err = rng.normal(0, nat_sigma, size=n_sims)
state_err = rng.normal(0, state_sigma, size=(n_sims, len(states)))
sim_margin = base_margin[None, :] + nat_err[:, None] + state_err

dem_wins = (sim_margin > 0).astype(int)
dem_ec = (dem_wins * evs[None, :]).sum(axis=1)

p_dem_win = float((dem_ec >= 270).mean())
expected_ec = float(dem_ec.mean())

state_win_prob = dem_wins.mean(axis=0)
out = df.copy()
out["p_dem_win_state"] = state_win_prob

# Most likely outcome (simple mode on integer EC counts)
vals, counts = np.unique(dem_ec.astype(int), return_counts=True)
mode_ec = int(vals[np.argmax(counts)])

# -----------------------------
# UI: toplines
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("P(Dem wins EC ≥ 270)", f"{p_dem_win:.1%}")
c2.metric("Expected Dem EV", f"{expected_ec:.1f}")
c3.metric("Median Dem EV", f"{np.median(dem_ec):.0f}")
c4.metric("Most-likely Dem EV", f"{mode_ec}")
c5.metric("10–90% Dem EV", f"{np.percentile(dem_ec,10):.0f}–{np.percentile(dem_ec,90):.0f}")

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Map: P(Dem win) by state")
    fig_map = px.choropleth(
        out,
        locations="state",
        locationmode="USA-states",
        scope="usa",
        color="p_dem_win_state",
        hover_data={
            "model_margin":":.1f",
            "base_margin_for_model":":.1f",
            "baseline_margin":":.1f",
            "ev":True,
            "p_dem_win_state":":.2f",
        },
        labels={"p_dem_win_state":"P(Dem win)"},
    )
    st.plotly_chart(fig_map, use_container_width=True)

with right:
    st.subheader("EC distribution")
    fig_hist = px.histogram(pd.DataFrame({"Dem_EV": dem_ec}), x="Dem_EV", nbins=45)
    st.plotly_chart(fig_hist, use_container_width=True)

# Quick “Swing States” section
st.subheader("Swing states (0.35 < P(Dem win) < 0.65)")
swing = out[(out["p_dem_win_state"] > 0.35) & (out["p_dem_win_state"] < 0.65)].copy()
swing = swing.sort_values("p_dem_win_state")
st.dataframe(
    swing[["state","ev","model_margin","p_dem_win_state"]].rename(columns={"p_dem_win_state":"p_dem"}),
    use_container_width=True
)

st.subheader("State table")
show = out[["state","ev","baseline_margin","base_margin_for_model","model_margin","p_dem_win_state"]].copy()
show = show.sort_values("p_dem_win_state", ascending=False)
st.dataframe(show, use_container_width=True)

with st.expander("Assumptions (print this during the demo)"):
    st.write(
        f"""
**Baseline year:** {baseline_year}  
**National swing applied:** {national_swing:+.1f} points (Dem−Rep)  
**National correlated error σ:** {nat_sigma:.1f} points  
**State-specific error σ:** {state_sigma:.1f} points  
**Simulations:** {n_sims:,}  
**EV allocation:** 2024/2028 allocation per U.S. National Archives (538 total, 270 to win).  
        """
    )

st.caption(
    "Notes: This demo uses a simple probabilistic model for presentation. "
    "Baseline results are from MIT Election Data & Science Lab (1976–2020) via a public mirror. "
    "Electoral vote allocation uses the 2024/2028 distribution published by the U.S. National Archives. "
    "Maine and Nebraska split electoral votes by district in real life, but this demo treats states as winner-take-all."
)
