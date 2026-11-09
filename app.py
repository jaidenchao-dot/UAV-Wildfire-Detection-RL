# -*- coding: utf-8 -*-
"""Streamlit GUI Runner for UAV Wildfire Patrol Simulation (test mode)

Runs pre-trained Q-weights (no training) through seeded test scenarios using
test_q_weights_v2.py, which uses the current v3_uav_wildfires.py environment
(with recharges). The weights file used always matches the selected drone.

A second tab displays the final research paper, including the figures
embedded in the markdown file and a local dashboard screenshot.
"""

import re
import sys
import base64
import subprocess
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PAPER_CANDIDATES = [
    BASE_DIR / "UAV Wildfire Prevention Final Research Paper.md",
    BASE_DIR / "UAV_Wildfire_Prevention_Final_Research_Paper.md",
]
PAPER_PATH = next((p for p in PAPER_CANDIDATES if p.exists()), PAPER_CANDIDATES[0])
DASHBOARD_FIGURE = BASE_DIR / "PapersFigures" / "image.png"
LOCAL_FIGURE_TOKEN = "__local_dashboard_figure__"

DRONES = {
    "Heavy Duty": "Max Steps: 30 | Radius: 10 units",
    "All-In-One": "Max Steps: 50 | Radius: 5 units",
    "Explorer": "Max Steps: 100 | Radius: 2 units",
}


def weights_filename(drone_name, kind="best"):
    """Must match the naming used by v3_uav_wildfires.py."""
    return f"{kind}_q_weights_{drone_name.replace(' ', '_')}.pt"


@st.cache_data
def load_paper(path_str, mtime):
    """Split the paper into text chunks and image references.

    Returns a list of ("text", str) and ("image", image_id) parts, plus a
    dict mapping image ids to decoded image bytes. `mtime` is only used so the
    cache refreshes when the markdown file changes.
    """
    text = Path(path_str).read_text(encoding="utf-8")

    # Reference-style image definitions: [image1]: <data:image/png;base64,...>
    images = {}
    def_pattern = re.compile(
        r"^\[(?P<id>[^\]]+)\]:\s*<data:image/[\w.+-]+;base64,(?P<data>[^>]+)>\s*$",
        re.MULTILINE,
    )
    for match in def_pattern.finditer(text):
        images[match.group("id")] = base64.b64decode(match.group("data"))
    text = def_pattern.sub("", text)

    # Replace the "OPTIONAL FIGURE 5" placeholder with the local dashboard image
    text, replaced = re.subn(
        r"OPTIONAL FIGURE 5:.*?Results section\.\s*",
        f"![][{LOCAL_FIGURE_TOKEN}]\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )

    parts = []
    pieces = re.split(r"!\[[^\]]*\]\[([^\]]+)\]", text)
    for i, piece in enumerate(pieces):
        if i % 2 == 0:
            if piece.strip():
                parts.append(("text", piece))
        else:
            parts.append(("image", piece))

    return parts, images, bool(replaced)


@st.cache_data
def read_checkpoint_info(path_str, mtime):
    """Return (n_observations, obs_version) stored in a weights file.

    Weights from the old v2_uav_wildfires.py have no obs_version. `mtime` only
    refreshes the cache when the file changes.
    """
    import torch
    ckpt = torch.load(path_str, map_location="cpu", weights_only=True)
    return ckpt.get("n_observations"), ckpt.get("obs_version")


@st.cache_resource
def current_obs_version():
    """Observation version of the current environment (v3_uav_wildfires.py)."""
    from v3_uav_wildfires import OBS_VERSION
    return OBS_VERSION


def render_paper():
    if not PAPER_PATH.exists():
        st.error(f"Paper not found: `{PAPER_PATH.name}`")
        return

    parts, images, placeholder_found = load_paper(str(PAPER_PATH), PAPER_PATH.stat().st_mtime)

    with open(PAPER_PATH, "rb") as f:
        st.download_button(
            "⬇️ Download paper (Markdown)",
            data=f.read(),
            file_name=PAPER_PATH.name,
            mime="text/markdown",
        )

    figure_num = 0
    for kind, value in parts:
        if kind == "text":
            st.markdown(value)
        elif value == LOCAL_FIGURE_TOKEN:
            show_dashboard_figure()
        elif value in images:
            figure_num += 1
            st.image(images[value], caption=f"Figure {figure_num}", use_container_width=True)
        else:
            st.warning(f"Image `{value}` is referenced in the paper but not defined.")

    # If the placeholder text was removed from the paper, still show the figure
    if not placeholder_found:
        show_dashboard_figure()


def show_dashboard_figure():
    if DASHBOARD_FIGURE.exists():
        st.image(
            str(DASHBOARD_FIGURE),
            caption="Figure 5: Streamlit testing dashboard",
            use_container_width=True,
        )
    else:
        st.info(f"Add the dashboard screenshot at `PapersFigures/{DASHBOARD_FIGURE.name}` to show Figure 5.")


# Page configuration
st.set_page_config(
    page_title="UAV Wildfire Patrol Dashboard",
    page_icon="🚁",
    layout="wide"
)

st.title("🔥 UAV Wildfire Patrol Test Dashboard")

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
    help="Folder containing the .pt files saved by v3_uav_wildfires.py."
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
retrain_cmd = f"`python v3_uav_wildfires.py -d \"{drone_model}\"`"
weights_ok = False
if not weights_path.exists():
    st.sidebar.error(f"`{weights_path}` not found. Train this drone first:\n\n{retrain_cmd}")
else:
    try:
        n_obs, obs_version = read_checkpoint_info(str(weights_path), weights_path.stat().st_mtime)
        if obs_version == current_obs_version():
            weights_ok = True
            st.sidebar.success(f"Using `{weights_path}` (v3, {n_obs} inputs, with recharges)")
        else:
            st.sidebar.error(
                f"`{weights_path}` was trained with the old v2 environment "
                f"({n_obs} inputs, no recharges). Retrain it with v3:\n\n{retrain_cmd}"
            )
    except Exception as e:
        st.sidebar.error(f"Could not read `{weights_path}`: {e}")

run_clicked = st.sidebar.button("🚀 Run Test", type="primary", disabled=not weights_ok)

tab_test, tab_paper = st.tabs(["🚁 Test Dashboard", "📄 Research Paper"])

with tab_test:
    st.markdown(
        "Fly trained Deep Q-Network (DQN) drones through randomly generated wildfire "
        "scenarios across the Santa Monica Mountains boundary."
    )

    # Run test
    if run_clicked:
        with st.spinner(f"Running {num_cases} test case(s) with the {drone_model} drone..."):
            cmd = [
                sys.executable, str(BASE_DIR / "test_q_weights_v2.py"),
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

with tab_paper:
    render_paper()