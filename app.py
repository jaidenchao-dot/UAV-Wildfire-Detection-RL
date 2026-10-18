# -*- coding: utf-8 -*-
"""Streamlit GUI Runner for UAV Wildfire Patrol Simulation (test mode)

Runs pre-trained Q-weights (no training) through seeded test scenarios using
test_q_weights.py. The weights file used always matches the selected drone.
"""

import sys
import subprocess
from pathlib import Path

import streamlit as st

DRONES = {
    "Heavy Duty": "Max Steps: 30 | Radius: 10 units",
    "All-In-One": "Max Steps: 50 | Radius: 5 units",
    "Explorer": "Max Steps: 100 | Radius: 2 units",
}


def weights_filename(drone_name, kind="best"):
    """Must match the naming used by v2_uav_wildfires.py."""
    return f"{kind}_q_weights_{drone_name.replace(' ', '_')}.pt"


# Page configuration
st.set_page_config(
    page_title="UAV Wildfire Patrol Dashboard",
    page_icon="🚁",
    layout="wide"
)

st.title("🔥 UAV Wildfire Patrol Test Dashboard")
st.markdown(
    "Fly trained Deep Q-Network (DQN) drones through randomly generated wildfire "
    "scenarios across the Santa Monica Mountains boundary."
)

# Sidebar controls for configuration
st.sidebar.header("Configuration Setup")

drone_model = st.sidebar.selectbox(
    "Select Drone Model",
    list(DRONES),
    index=1,
    help="\n".join(f"• {name} ({spec})" for name, spec in DRONES.items())
)
st.sidebar.caption(DRONES[drone_model])

weights_kind = st.sidebar.radio(
    "Weights to Use",
    ["best", "final"],
    format_func=lambda k: "Best (highest rolling-average reward)" if k == "best" else "Final (end of training)",
)

weights_folder = st.sidebar.text_input(
    "Weights Folder",
    value="weights",
    help="Folder containing the .pt files saved by v2_uav_wildfires.py."
)

num_cases = st.sidebar.slider(
    "Number of Test Cases",
    min_value=1,
    max_value=20,
    value=5,
    help="Each case is a new randomly generated fire scenario."
)

seed = st.sidebar.number_input(
    "Base Seed",
    min_value=0,
    value=0,
    step=1,
    help="Case i uses seed + i. The same seed reproduces the same scenarios."
)

output_folder = st.sidebar.text_input(
    "Output Folder Name",
    value="test_results",
    help="Local folder where test plots will be saved (cleared on each run)."
)

# Weights file matching the selected drone
weights_path = Path(weights_folder) / weights_filename(drone_model, weights_kind)
if weights_path.exists():
    st.sidebar.success(f"Using `{weights_path}`")
else:
    st.sidebar.error(
        f"`{weights_path}` not found. Train this drone first:\n\n"
        f"`python v2_uav_wildfires.py -d \"{drone_model}\"`"
    )

# Run test button
if st.sidebar.button("🚀 Run Test", type="primary", disabled=not weights_path.exists()):
    with st.spinner(f"Running {num_cases} test case(s) with the {drone_model} drone..."):
        cmd = [
            sys.executable, "test_q_weights.py",
            "--weights", str(weights_path),
            "--cases", str(num_cases),
            "--seed", str(int(seed)),
            "--output_folder", output_folder,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        st.success(f"Test run completed for {drone_model} ({weights_kind} weights).")
        st.session_state["last_summary"] = result.stdout
        st.session_state["last_run"] = f"{drone_model} — {weights_kind} weights — seeds {int(seed)}–{int(seed) + num_cases - 1}"
    else:
        st.error("An error occurred during the test run:")
        st.code(result.stderr or result.stdout)

st.divider()

# Results summary
if "last_summary" in st.session_state:
    st.header("📋 Test Summary")
    st.caption(st.session_state["last_run"])
    st.code(st.session_state["last_summary"], language=None)

# Results Gallery Section
st.header("📊 Test Flight Visualizations")
out_path = Path(output_folder)

if out_path.exists():
    image_files = sorted(out_path.glob("test_case_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    if image_files:
        cols = st.columns(2)
        for i, img in enumerate(image_files):
            with cols[i % 2]:
                st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No test plots found yet. Click 'Run Test' in the sidebar to generate results.")
else:
    st.info("Output folder does not exist yet. Click 'Run Test' in the sidebar to begin.")