from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from base_engine import extract_notebook_headings, load_inputs_from_uploads, render_route_map
from compare_backends import (
    BackendConfig,
    build_compare_excel,
    load_saved_backend_artifacts,
    load_uploaded_esri_workbook,
    run_backend_suite,
    save_backend_artifacts,
)

st.set_page_config(page_title="VRP Backend Live Compare Studio", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")


LOCAL_SYNTHETIC_DATA_CANDIDATES = [
    os.getenv("VRP_SYNTHETIC_DATA_DIR", "").strip(),
    r"/mnt/c/Users/AnubhaAnubha/OneDrive - Pearce Services, LLC/onedrive_ubuntu/project/exec_aware_arcgis_vrp_poc_all_scenarios/synthetic_data",
    str(Path(__file__).resolve().parent / "synthetic_data"),
    str(Path(__file__).resolve().parent.parent / "synthetic_data"),
]

LOCAL_NOTEBOOK_ARTIFACTS_CANDIDATES = [
    os.getenv("VRP_NOTEBOOK_ARTIFACTS_DIR", "").strip(),
    r"/mnt/c/Users/AnubhaAnubha/OneDrive - Pearce Services, LLC/onedrive_ubuntu/project/exec_aware_arcgis_vrp_poc_all_scenarios/notebook_artifacts",
    str(Path(__file__).resolve().parent / "notebook_artifacts"),
    str(Path(__file__).resolve().parent.parent / "notebook_artifacts"),
]


def _pick_first_match(folder: Path, patterns: list[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_local_input_files() -> tuple[Optional[Path], Dict[str, Optional[Path]]]:
    for raw_dir in LOCAL_SYNTHETIC_DATA_CANDIDATES:
        if not raw_dir:
            continue
        folder = Path(raw_dir)
        if not folder.exists() or not folder.is_dir():
            continue

        files = {
            "orders": _pick_first_match(folder, ["orders_*.csv", "orders*.csv"]),
            "routes": _pick_first_match(folder, ["routes_*.csv", "routes*.csv"]),
            "accounts": _pick_first_match(folder, ["accounts.csv", "accounts_*.csv", "accounts*.csv"]),
            "depots": _pick_first_match(folder, ["depots_*.csv", "depots*.csv"]),
            "goal_profiles": _pick_first_match(folder, ["goal_profiles.csv", "goal_profiles_*.csv", "goal*.csv"]),
        }
        if all(files.values()):
            return folder, files
    return None, {
        "orders": None,
        "routes": None,
        "accounts": None,
        "depots": None,
        "goal_profiles": None,
    }


def find_default_artifact_root() -> Path:
    for raw_dir in LOCAL_NOTEBOOK_ARTIFACTS_CANDIDATES:
        if not raw_dir:
            continue
        folder = Path(raw_dir)
        if folder.exists() and folder.is_dir():
            return folder
    for raw_dir in LOCAL_NOTEBOOK_ARTIFACTS_CANDIDATES:
        if raw_dir:
            return Path(raw_dir)
    return Path("notebook_artifacts")


st.markdown("""
<style>
.block-container {padding-top:1rem; padding-bottom:1rem;}
.metric-card {background:#f7f9fc;border:1px solid #e6ebf2;padding:0.85rem 1rem;border-radius:16px;}
.small-note {color:#5b6577; font-size:0.92rem;}
.backend-tag {padding:0.15rem 0.5rem;border-radius:999px;background:#eef3ff;border:1px solid #d5def7;}
</style>
""", unsafe_allow_html=True)

st.title("🧪 VRP Backend Live Compare Studio")
st.caption(
    "One input set. Compare your impacted-subset greedy + 2-opt engine, the Hybrid Execution-Aware Rolling VRP Solver, the Adaptive Execution-Aware Metaheuristic Solver, OSRM road routing, PyVRP metaheuristic solve, OR-Tools routing solve, and Esri either from live inference or from an already-inferenced workbook."
)

local_folder, local_files = find_local_input_files()
auto_local_ready = local_folder is not None and all(local_files.values())
default_artifact_root = find_default_artifact_root()

with st.sidebar:
    st.header("📦 Same input for all compare sources")
    use_local_files = False
    if auto_local_ready:
        st.success(f"Auto-loaded input folder found: {local_folder}")
        use_local_files = st.toggle("Use local synthetic_data files automatically", value=True)
        with st.expander("Auto-detected files", expanded=False):
            for label, path in local_files.items():
                st.write(f"**{label}**: `{path.name}`")
    else:
        st.info("Local synthetic_data folder with all five required CSVs was not found. Upload the files below.")

    orders_file = None
    routes_file = None
    accounts_file = None
    depots_file = None
    goal_profiles_file = None

    if use_local_files:
        orders_file = str(local_files["orders"])
        routes_file = str(local_files["routes"])
        accounts_file = str(local_files["accounts"])
        depots_file = str(local_files["depots"])
        goal_profiles_file = str(local_files["goal_profiles"])
        with st.expander("Manual override uploads", expanded=False):
            orders_override = st.file_uploader("Orders CSV", type=["csv"], key="orders_override")
            routes_override = st.file_uploader("Routes CSV", type=["csv"], key="routes_override")
            accounts_override = st.file_uploader("Accounts CSV", type=["csv"], key="accounts_override")
            depots_override = st.file_uploader("Depots CSV", type=["csv"], key="depots_override")
            goal_profiles_override = st.file_uploader("Goal profiles CSV", type=["csv"], key="goal_profiles_override")
            if orders_override is not None:
                orders_file = orders_override
            if routes_override is not None:
                routes_file = routes_override
            if accounts_override is not None:
                accounts_file = accounts_override
            if depots_override is not None:
                depots_file = depots_override
            if goal_profiles_override is not None:
                goal_profiles_file = goal_profiles_override
    else:
        orders_file = st.file_uploader("Orders CSV", type=["csv"], key="orders")
        routes_file = st.file_uploader("Routes CSV", type=["csv"], key="routes")
        accounts_file = st.file_uploader("Accounts CSV", type=["csv"], key="accounts")
        depots_file = st.file_uploader("Depots CSV", type=["csv"], key="depots")
        goal_profiles_file = st.file_uploader("Goal profiles CSV", type=["csv"], key="goal_profiles")

    notebook_file = st.file_uploader("Reference notebook (.ipynb)", type=["ipynb"], key="notebook")

    st.divider()
    st.header("🧠 Backends to include")
    selected_backends = st.multiselect(
        "Choose backends",
        [
            "Impacted-Subset Greedy + 2-opt",
            "Hybrid Execution-Aware Rolling VRP Solver",
            "Adaptive Execution-Aware Metaheuristic Solver",
            "OSRM",
            "PyVRP",
            "OR-Tools",
            "OR-Tools Execution-Aware",
        ],
        default=["Impacted-Subset Greedy + 2-opt"],
    )

    st.divider()
    st.header("🗂️ Notebook artifact cache")
    notebook_artifacts_root = st.text_input(
        "Notebook artifacts root folder (WSL path)",
        value=str(default_artifact_root),
        help="A subfolder will be created for each backend. Example: /mnt/c/Users/AnubhaAnubha/OneDrive - Pearce Services, LLC/onedrive_ubuntu/project/exec_aware_arcgis_vrp_poc_all_scenarios/notebook_artifacts",
    )
    save_live_backends_to_artifacts = st.toggle(
        "Save each live backend run to notebook_artifacts",
        value=True,
        help="Each selected backend gets its own folder with vrp_scenario_results_full.xlsx plus maps_html and maps_png.",
    )
    backend_source_modes: Dict[str, str] = {}
    if selected_backends:
        st.caption("Choose whether each selected backend should run now or be loaded from notebook_artifacts.")
        for backend in selected_backends:
            backend_source_modes[backend] = st.radio(
                f"{backend}",
                ["Run live now", "Load saved workbook from notebook_artifacts"],
                index=0,
                key=f"source_mode__{backend}",
            )

    st.divider()
    st.header("🌐 Esri comparison source")
    esri_mode = st.radio(
        "How should Esri be included?",
        [
            "Skip Esri",
            "Run live Esri backend",
            "Load saved Esri workbook from notebook_artifacts",
            "Upload already-inferenced Esri workbook",
        ],
        index=0,
    )
    uploaded_esri_workbook = None
    if esri_mode == "Load saved Esri workbook from notebook_artifacts":
        st.caption("Loads notebook_artifacts/Esri/vrp_scenario_results_full.xlsx using the same cache path above.")
    elif esri_mode == "Upload already-inferenced Esri workbook":
        uploaded_esri_workbook = st.file_uploader(
            "Uploaded Esri scenario workbook (.xlsx)",
            type=["xlsx"],
            key="uploaded_esri_workbook",
            help="Use a workbook like vrp_scenario_results_full.xlsx that already contains Esri route/stop outputs.",
        )

    st.divider()
    st.header("⚙️ Common run settings")
    fast_mode = st.toggle("Notebook-style impacted subset mode", value=True)

    if "max_orders_scope" not in st.session_state:
        st.session_state["max_orders_scope"] = 150
    if "max_routes_scope" not in st.session_state:
        st.session_state["max_routes_scope"] = 50

    if st.button("Reset scope defaults to 150 / 50", use_container_width=True):
        st.session_state["max_orders_scope"] = 150
        st.session_state["max_routes_scope"] = 50

    max_orders = st.slider("Orders in solve scope", 80, 1200, key="max_orders_scope", step=10)
    max_routes = st.slider("Routes in solve scope", 8, 80, key="max_routes_scope", step=1)
    random_seed = st.number_input("Random seed", min_value=1, max_value=99999, value=42, step=1)
    depot_name = st.text_input("Preferred depot focus (optional)", value="")
    speed_kmph = st.slider("Average travel speed (km/h) for heuristic fallback", 20, 80, 30, 1)
    sample_by_priority = st.toggle("Bias subset toward high priority & SLA orders", value=True)

    st.divider()
    st.header("🔧 Engine options")
    use_two_opt = st.toggle("Use 2-opt route refinement", value=True)
    strict_skill_bias = st.toggle("Strong specialty matching", value=True)

    st.divider()
    st.header("🗺️ OSRM backend")
    st.caption("Use the public demo server only for small tests. For larger comparisons, point this to your self-hosted OSRM URL.")
    osrm_base_url = st.text_input("OSRM base URL", value="https://router.project-osrm.org")
    osrm_profile = st.selectbox("OSRM profile", ["driving", "car", "truck"], index=0)
    osrm_timeout = st.slider("OSRM timeout (seconds)", 5, 120, 60, 5)
    osrm_auto_reduce = st.toggle("Auto-reduce OSRM scope on public demo server", value=True)
    osrm_demo_max_orders = st.slider("Public OSRM max orders", 10, 200, 80, 5)
    osrm_demo_max_routes = st.slider("Public OSRM max routes", 2, 20, 10, 1)

    st.divider()
    st.header("🧬 PyVRP backend")
    st.caption(
        "Metaheuristic VRPTW-style backend using PyVRP. This adapter builds a travel-time matrix from the app's haversine + speed model and can optionally leave lower-value orders unassigned."
    )
    pyvrp_runtime_seconds = st.slider("PyVRP solve time budget (seconds)", 1, 60, 30, 1)
    pyvrp_allow_drop_orders = st.toggle("Allow dropped/unassigned orders in PyVRP", value=True)

    st.divider()
    st.header("🧭 OR-Tools backend")
    st.caption(
        "Google OR-Tools CVRPTW-style backend using route capacities, start/end depots, time windows, guided local search, and optional dropped-order penalties."
    )
    ortools_runtime_seconds = st.slider("OR-Tools solve time budget (seconds)", 1, 120, 45, 1)
    ortools_allow_drop_orders = st.toggle("Allow dropped/unassigned orders in OR-Tools", value=True)

    st.divider()
    st.header("🔒 OR-Tools Execution-Aware backend")
    st.caption(
        "Beats plain OR-Tools on EAS/RT scenarios by embedding locked-prefix constraints directly into the OR-Tools model. "
        "Locked stops are pinned with zero-slack time windows, forced vehicle assignment, and sequence precedence. "
        "OR-Tools then spends its full GLS budget only on open/unlocked orders → smaller search space → better distance with zero route churn."
    )

    st.divider()
    st.header("🔐 Live Esri credentials")
    st.caption("These are only needed when 'Run live Esri backend' is selected above.")
    esri_portal_url = st.text_input("Portal URL", value=os.getenv("ARCGIS_PORTAL_URL", ""))
    esri_username = st.text_input("Username", value=os.getenv("ARCGIS_USERNAME", ""))
    esri_password = st.text_input("Password", value=os.getenv("ARCGIS_PASSWORD", ""), type="password")
    esri_travel_mode = st.text_input("Travel mode", value="Driving Time")

    run_clicked = st.button("▶️ Build comparison", use_container_width=True, type="primary")

required = [orders_file, routes_file, accounts_file, depots_file, goal_profiles_file]
if not all(required):
    st.info("The app needs the five core CSVs. It first checks the local synthetic_data folder and auto-loads them when found; otherwise it falls back to manual upload.")
    st.stop()

try:
    inputs = load_inputs_from_uploads(
        orders_file=orders_file,
        routes_file=routes_file,
        accounts_file=accounts_file,
        depots_file=depots_file,
        goal_profiles_file=goal_profiles_file,
    )
except Exception as exc:
    st.error(f"Input loading failed: {exc}")
    st.caption("Tip: replace both app.py and compare_backends.py together. This app now passes PyVRP and OR-Tools settings into BackendConfig.")
    st.stop()

headings = []
if notebook_file is not None:
    try:
        headings = extract_notebook_headings(notebook_file)
    except Exception as exc:
        st.warning(f"Could not read notebook headings: {exc}")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Orders", f"{len(inputs['orders']):,}")
with c2:
    st.metric("Routes", f"{len(inputs['routes']):,}")
with c3:
    st.metric("Depots", f"{len(inputs['depots']):,}")
with c4:
    st.metric("Accounts", f"{len(inputs['accounts']):,}")
with c5:
    st.metric("Goal profiles", f"{len(inputs['goal_profiles']):,}")

if headings:
    with st.expander("Notebook scenario headings detected", expanded=False):
        st.write(headings)

if osrm_base_url.strip().rstrip("/") == "https://router.project-osrm.org" and osrm_auto_reduce:
    st.info(
        f"OSRM is using the public demo server, so this run will auto-clamp OSRM scope to {osrm_demo_max_orders} orders and {osrm_demo_max_routes} routes."
    )

config = BackendConfig(
    speed_kmph=float(speed_kmph),
    use_two_opt=use_two_opt,
    strict_skill_bias=strict_skill_bias,
    random_seed=int(random_seed),
    max_orders=int(max_orders),
    max_routes=int(max_routes),
    depot_focus=depot_name.strip() or None,
    sample_by_priority=sample_by_priority,
    fast_mode=fast_mode,
    osrm_base_url=osrm_base_url.strip(),
    osrm_profile=osrm_profile.strip(),
    osrm_timeout=int(osrm_timeout),
    osrm_auto_reduce=bool(osrm_auto_reduce),
    osrm_demo_max_orders=int(osrm_demo_max_orders),
    osrm_demo_max_routes=int(osrm_demo_max_routes),
    pyvrp_max_runtime_seconds=int(pyvrp_runtime_seconds),
    pyvrp_allow_drop_orders=bool(pyvrp_allow_drop_orders),
    ortools_max_runtime_seconds=int(ortools_runtime_seconds),
    ortools_allow_drop_orders=bool(ortools_allow_drop_orders),
    esri_portal_url=esri_portal_url.strip(),
    esri_username=esri_username.strip(),
    esri_password=esri_password,
    esri_travel_mode=esri_travel_mode.strip() or "Driving Time",
)

if run_clicked:
    artifact_root_path = Path(notebook_artifacts_root).expanduser()
    run_backends = list(selected_backends)
    if esri_mode == "Run live Esri backend":
        run_backends.append("Esri")

    esri_saved_target_count = 1 if esri_mode == "Load saved Esri workbook from notebook_artifacts" else 0
    upload_target_count = 1 if esri_mode == "Upload already-inferenced Esri workbook" and uploaded_esri_workbook is not None else 0
    total_steps = max(len(run_backends) * 18 + len(selected_backends) + esri_saved_target_count + upload_target_count, 1)

    progress = st.progress(0.0)
    status = st.empty()
    counter = {"n": 0}

    def progress_cb(backend: str, idx: int, total: int, name: str):
        counter["n"] += 1
        progress.progress(min(counter["n"] / total_steps, 1.0))
        status.info(f"{backend}: {name}")

    results = {}
    errors = {}

    with st.spinner("Building the comparison pack..."):
        if save_live_backends_to_artifacts:
            artifact_root_path.mkdir(parents=True, exist_ok=True)

        for backend in selected_backends:
            source_mode = backend_source_modes.get(backend, "Run live now")
            try:
                if source_mode == "Load saved workbook from notebook_artifacts":
                    status.info(f"{backend}: loading saved workbook from {artifact_root_path}")
                    results[backend] = load_saved_backend_artifacts(artifact_root_path, backend)
                    counter["n"] += 1
                    progress.progress(min(counter["n"] / total_steps, 1.0))
                else:
                    part = run_backend_suite(inputs, config, [backend], progress_cb=progress_cb)
                    bundle = part[backend]
                    if save_live_backends_to_artifacts:
                        save_backend_artifacts(artifact_root_path, backend, bundle, inputs)
                    results[backend] = bundle
            except Exception as exc:
                errors[backend] = str(exc)

        if esri_mode == "Run live Esri backend":
            try:
                part = run_backend_suite(inputs, config, ["Esri"], progress_cb=progress_cb)
                bundle = part["Esri"]
                if save_live_backends_to_artifacts:
                    save_backend_artifacts(artifact_root_path, "Esri", bundle, inputs)
                results["Esri"] = bundle
            except Exception as exc:
                errors["Esri"] = str(exc)

        if esri_mode == "Load saved Esri workbook from notebook_artifacts":
            try:
                status.info(f"Esri: loading saved workbook from {artifact_root_path}")
                results["Esri"] = load_saved_backend_artifacts(artifact_root_path, "Esri")
                counter["n"] += 1
                progress.progress(min(counter["n"] / total_steps, 1.0))
            except Exception as exc:
                errors["Esri"] = str(exc)

        if esri_mode == "Upload already-inferenced Esri workbook":
            if uploaded_esri_workbook is None:
                errors["Esri (Uploaded)"] = "Choose an .xlsx workbook when upload mode is selected."
            else:
                try:
                    status.info("Esri (Uploaded): loading workbook")
                    uploaded_bundle = load_uploaded_esri_workbook(
                        uploaded_esri_workbook,
                        inputs=inputs,
                        config=config,
                        backend_name="Esri (Uploaded)",
                    )
                    results["Esri (Uploaded)"] = uploaded_bundle
                    counter["n"] += 1
                    progress.progress(min(counter["n"] / total_steps, 1.0))
                except Exception as exc:
                    errors["Esri (Uploaded)"] = str(exc)

    st.session_state["compare_results"] = results
    st.session_state["compare_errors"] = errors
    st.session_state["notebook_artifacts_root"] = str(artifact_root_path)
    progress.progress(1.0)
    status.success("Comparison pack is ready.")

if "compare_results" not in st.session_state:
    st.info("Choose your live backends and/or saved/uploaded Esri workbook, then click **Build comparison**.")
    st.stop()

results = st.session_state.get("compare_results", {})
errors = st.session_state.get("compare_errors", {})

if errors:
    with st.expander("Backend / workbook issues", expanded=bool(errors)):
        for backend, msg in errors.items():
            st.error(f"{backend}: {msg}")

if not results:
    st.warning("No comparison source completed successfully yet.")
    st.stop()

summary_frames = []
timing_frames = []
for backend, bundle in results.items():
    sdf = bundle["scenario_summary"].copy()
    sdf["Backend"] = backend
    summary_frames.append(sdf)
    tdf = bundle["scenario_timings"].copy()
    tdf["Backend"] = backend
    timing_frames.append(tdf)

summary_all = pd.concat(summary_frames, ignore_index=True)
timings_all = pd.concat(timing_frames, ignore_index=True)

def _safe_scenario_list(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "Scenario" not in df.columns:
        return []
    return df["Scenario"].dropna().astype(str).sort_values().unique().tolist()


def _safe_filter_scenario(df: pd.DataFrame, scenario_value: str) -> pd.DataFrame:
    if df is None or df.empty or "Scenario" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["Scenario"].astype(str) == str(scenario_value)].copy()


tabs = st.tabs(["Overview", "KPI Comparison", "Map Explorer", "Scenario Tables", "Export"])

with tabs[0]:
    st.subheader("Backend run overview")
    if st.session_state.get("notebook_artifacts_root"):
        st.caption(f"Notebook artifacts root: {st.session_state['notebook_artifacts_root']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Compare sources completed", len(results))
    with c2:
        st.metric("Scenarios per source", int(summary_all.groupby("Backend")["Scenario"].nunique().max()))
    with c3:
        st.metric("Uploaded orders", f"{len(inputs['orders']):,}")
    with c4:
        st.metric("Uploaded routes", f"{len(inputs['routes']):,}")

    st.dataframe(summary_all, use_container_width=True, hide_index=True)
    st.dataframe(timings_all, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Same input, different backend comparison")
    scenario_pick = st.selectbox("Scenario for comparison", sorted(summary_all["Scenario"].dropna().astype(str).unique().tolist()), index=0)
    subset = summary_all.loc[summary_all["Scenario"].astype(str) == scenario_pick].copy()

    a, b = st.columns(2)
    with a:
        fig = px.bar(subset, x="Backend", y="AssignedOrders", color="Backend", title=f"{scenario_pick} — assigned orders")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(subset, x="Backend", y="RoutesUsed", color="Backend", title=f"{scenario_pick} — routes used")
        st.plotly_chart(fig2, use_container_width=True)
    with b:
        fig3 = px.bar(subset, x="Backend", y="TotalDistanceKm", color="Backend", title=f"{scenario_pick} — total distance")
        st.plotly_chart(fig3, use_container_width=True)
        fig4 = px.bar(subset, x="Backend", y="TotalTravelMinutes", color="Backend", title=f"{scenario_pick} — travel minutes")
        st.plotly_chart(fig4, use_container_width=True)

    c, d = st.columns(2)
    with c:
        fig5 = px.bar(subset, x="Backend", y="LateMinutes", color="Backend", title=f"{scenario_pick} — late minutes")
        st.plotly_chart(fig5, use_container_width=True)
    with d:
        fig6 = px.bar(
            timings_all.loc[timings_all["Scenario"].astype(str) == scenario_pick],
            x="Backend",
            y="RuntimeSeconds",
            color="Backend",
            title=f"{scenario_pick} — runtime seconds",
        )
        st.plotly_chart(fig6, use_container_width=True)

with tabs[2]:
    st.subheader("Live map explorer")
    backend_pick = st.selectbox("Backend", list(results.keys()), key="map_backend")
    bundle = results[backend_pick]

    summary_df = bundle.get("scenario_summary", pd.DataFrame())
    routes_df = bundle.get("route_output_all", pd.DataFrame())
    stops_df = bundle.get("stop_output_all", pd.DataFrame())

    scenario_options = _safe_scenario_list(summary_df)
    if not scenario_options:
        st.info(f"No scenario summary rows are available for {backend_pick}.")
    else:
        scenario_pick = st.selectbox("Scenario", scenario_options, key="map_scenario")

        route_slice = _safe_filter_scenario(routes_df, scenario_pick)
        stop_slice = _safe_filter_scenario(stops_df, scenario_pick)

        if route_slice.empty and stop_slice.empty:
            st.info(f"No map data is available for {backend_pick} - {scenario_pick}.")
        else:
            route_options = []
            if not route_slice.empty and "RouteName" in route_slice.columns:
                route_options = (
                    route_slice["RouteName"]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )

            route_focus = st.selectbox(
                "Route focus",
                ["All routes"] + route_options,
                key="map_route",
            )

            map_obj = render_route_map(
                stops_df=stops_df if not stops_df.empty else pd.DataFrame(),
                depots_df=inputs["depots"],
                scenario_name=scenario_pick,
                route_name=None if route_focus == "All routes" else route_focus,
            )
            st_folium(map_obj, use_container_width=True, height=700)

with tabs[3]:
    st.subheader("Per-backend scenario tables")
    backend_pick = st.selectbox("Backend ", list(results.keys()), key="tbl_backend")
    bundle = results[backend_pick]

    summary_df = bundle.get("scenario_summary", pd.DataFrame())
    actions_df = bundle.get("scenario_actions", pd.DataFrame())
    routes_df = bundle.get("route_output_all", pd.DataFrame())
    stops_df = bundle.get("stop_output_all", pd.DataFrame())

    scenario_options = _safe_scenario_list(summary_df)
    if not scenario_options:
        st.info(f"No scenario table rows are available for {backend_pick}.")
    else:
        detail_scenario = st.selectbox(
            "Scenario detail",
            scenario_options,
            key="tbl_scenario",
        )

        st.write("Summary")
        st.dataframe(
            _safe_filter_scenario(summary_df, detail_scenario),
            use_container_width=True,
            hide_index=True,
        )

        st.write("Actions")
        if not actions_df.empty and "Scenario" in actions_df.columns:
            st.dataframe(
                _safe_filter_scenario(actions_df, detail_scenario),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No actions table available for this backend/scenario.")

        st.write("Routes")
        route_detail = _safe_filter_scenario(routes_df, detail_scenario)
        if route_detail.empty:
            st.info("No route rows available for this backend/scenario.")
        else:
            st.dataframe(route_detail, use_container_width=True, hide_index=True)

        st.write("Stops")
        stop_detail = _safe_filter_scenario(stops_df, detail_scenario)
        if stop_detail.empty:
            st.info("No stop rows available for this backend/scenario.")
        else:
            st.dataframe(stop_detail, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Download comparison workbook")
    excel_bytes = build_compare_excel(results)
    st.download_button(
        "Download compare workbook (.xlsx)",
        data=excel_bytes,
        file_name="vrp_backend_compare.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.json({backend: bundle["run_meta"] for backend, bundle in results.items()})
