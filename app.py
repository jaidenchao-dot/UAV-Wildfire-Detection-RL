# -*- coding: utf-8 -*-
"""Streamlit GUI Runner for UAV Wildfire Patrol Simulation"""

import streamlit as st
import subprocess
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="UAV Wildfire Patrol Dashboard",
    page_icon="🚁",
    layout="wide"
)

st.title("🔥 UAV Wildfire Patrol Training Dashboard")
st.markdown(
    "Configure and execute Deep Q-Network (DQN) drone patrol simulations across the "
    "Santa Monica Mountains boundary."
)

# Sidebar controls for configuration
st.sidebar.header("Configuration Setup")

drone_model = st.sidebar.selectbox(
    "Select Drone Model[cite: 1]",
    ["Heavy Duty", "All-In-One", "Explorer"],
    help=(
        "• Heavy Duty (Max Steps: 30 | Radius: 10 units)\n"
        "• All-In-One (Max Steps: 50 | Radius: 5 units)\n"
        "• Explorer (Max Steps: 100 | Radius: 2 units)"
    )
)

num_episodes = st.sidebar.slider(
    "Number of Training Episodes[cite: 1]",
    min_value=300,
    max_value=6000,
    value=300,
    step=50,
    help="Select between 300 to 6000 training episodes[cite: 1]."
)

output_folder = st.sidebar.text_input(
    "Output Folder Name",
    value="output_results",
    help="Local folder where generated figures will be saved."
)

# Run training button
if st.sidebar.button("🚀 Start Training", type="primary"):
    with st.spinner("Training simulation in progress... Please check your terminal for logs."):
        cmd = [
            "python", "v2_uav_wildfires.py",
            "--output_folder", output_folder,
            "--drone", drone_model,
            "--episodes", str(num_episodes)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Training completed successfully!")
        else:
            st.error("An error occurred during training execution:")
            st.code(result.stderr)

st.divider()

# Results Gallery Section
st.header("📊 Simulation Results & Visualizations")
out_path = Path(output_folder)

if out_path.exists():
    image_files = sorted(list(out_path.glob("*.png")))
    if image_files:
        selected_image = st.selectbox(
            "Select Generated Plot to View",
            image_files,
            format_func=lambda x: x.name
        )
        st.image(str(selected_image), caption=selected_image.name, use_container_width=True)
    else:
        st.info("No PNG plots found in the output folder yet. Run training to generate results.")
else:
    st.info("Output folder does not exist yet. Click 'Start Training' in the sidebar to begin.")