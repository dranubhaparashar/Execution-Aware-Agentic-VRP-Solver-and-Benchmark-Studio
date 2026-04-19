from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np
import pandas as pd

from base_engine import (
    DEFAULT_SCENARIOS,
    AssignmentAgent,
    ExplainAgent,
    PolicyAgent,
    ScenarioAgent,
    build_impacted_subset,
    haversine_km,
    map_catalog_to_requirements,
    render_route_map,
)


@dataclass
class BackendConfig:
    speed_kmph: float = 35.0
    use_two_opt: bool = True
    strict_skill_bias: bool = True
    random_seed: int = 42
    max_orders: int = 350
    max_routes: int = 24
    depot_focus: Optional[str] = None
    sample_by_priority: bool = True
    fast_mode: bool = True
    hybrid_candidate_routes: int = 10
    hybrid_frontier_size: int = 28
    hybrid_local_search_passes: int = 4
    hybrid_max_hard_late_minutes: float = 30.0
    hybrid_shift_buffer_minutes: float = 15.0
    hybrid_no_improve_rounds: int = 4
    hybrid_skip_clean_route_polish: bool = False
    hybrid_polish_top_k: int = 24
    hybrid_travel_focus_weight: float = 1.55
    hybrid_distance_focus_weight: float = 0.55
    hybrid_route_activation_penalty: float = 42.0
    hybrid_compact_passes: int = 3
    hybrid_compact_route_limit: int = 10
    hybrid_global_relocate_passes: int = 3
    hybrid_global_swap_passes: int = 2
    hybrid_require_zero_late_overtime: bool = True
    osrm_base_url: str = "https://router.project-osrm.org"
    osrm_profile: str = "driving"
    osrm_timeout: int = 20
    osrm_auto_reduce: bool = True
    osrm_demo_max_orders: int = 30
    osrm_demo_max_routes: int = 6
    pyvrp_max_runtime_seconds: int = 5
    pyvrp_allow_drop_orders: bool = False
    ortools_max_runtime_seconds: int = 5
    ortools_allow_drop_orders: bool = True
    esri_portal_url: str = ""
    esri_username: str = ""
    esri_password: str = ""
    esri_travel_mode: str = "Driving Time"


class OSRMClient:
    def __init__(self, base_url: str = "https://router.project-osrm.org", profile: str = "driving", timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = timeout
        self._cache: Dict[Tuple[float, float, float, float], Tuple[float, float]] = {}

    def travel(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        key = (round(float(lat1), 6), round(float(lon1), 6), round(float(lat2), 6), round(float(lon2), 6))
        if key in self._cache:
            return self._cache[key]
        import requests
        coords = f"{lon1},{lat1};{lon2},{lat2}"
        url = f"{self.base_url}/route/v1/{self.profile}/{coords}"
        params = {"overview": "false", "steps": "false", "annotations": "false"}
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OSRM timed out at {self.base_url}. "
                f"If you are using the public demo server, reduce OSRM scope or switch to a self-hosted OSRM URL."
            ) from exc
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            raise RuntimeError(f"OSRM route failed: {data}")
        route = data["routes"][0]
        dist_km = float(route["distance"]) / 1000.0
        minutes = float(route["duration"]) / 60.0
        self._cache[key] = (dist_km, minutes)
        return self._cache[key]


class OSRMAssignmentAgent(AssignmentAgent):
    def __init__(self, depots_df: pd.DataFrame, osrm_client: OSRMClient, speed_kmph: float = 35.0, use_two_opt: bool = True):
        super().__init__(depots_df, speed_kmph=speed_kmph, use_two_opt=use_two_opt)
        self.osrm = osrm_client

    def _travel_between(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
        return self.osrm.travel(lat1, lon1, lat2, lon2)

    def _score_route(self, order: pd.Series, route: pd.Series, weights: Dict[str, float], route_state: Dict[str, Any]) -> float:
        depot_name = str(route.get("StartDepotName"))
        depot = self.depot_lookup.get(depot_name, {})
        d_km, travel_min = self._travel_between(
            float(order["Latitude"]),
            float(order["Longitude"]),
            float(depot.get("Latitude", order["Latitude"])),
            float(depot.get("Longitude", order["Longitude"])),
        )
        rating_penalty = (5.5 - float(route.get("TechnicianRating", 4.0))) * 10.0
        load_ratio = route_state["assigned_count"] / max(float(route.get("MaxOrderCount", 1)), 1.0)
        overtime_risk = max(0.0, route_state["current_minutes"] + float(order.get("ServiceTime", 45)) - float(route.get("OvertimeStartTime", 540)))
        tw_end = order.get("TimeWindowEnd1")
        eta_pressure = 0.0
        if pd.notna(tw_end) and pd.notna(route.get("EarliestStartTime")):
            start_gap = (tw_end - route.get("EarliestStartTime")).total_seconds() / 60.0
            eta_pressure = max(0.0, travel_min - max(start_gap, 0.0))
        route_skills = _skill_set(route.get("SpecialtyNames"))
        order_skills = _skill_set(order.get("SpecialtyNames") or order.get("WorkType"))
        skill_mismatch = 0.0 if (not order_skills or route_skills.intersection(order_skills)) else 1.0
        depot_mismatch = 0.0 if str(order.get("PreferredDepot")) == depot_name else 1.0
        priority_reward = float(order.get("Priority", 3)) * float(weights.get("priority", 0.0))
        score = (
            weights["travel"] * d_km
            + weights["rating"] * rating_penalty
            + weights["sla"] * eta_pressure
            + weights["overtime"] * (overtime_risk / 60.0)
            + weights["skill"] * skill_mismatch * 90.0
            + depot_mismatch * 4.0
            + load_ratio * 18.0
            - priority_reward
        )
        return float(score)

    def _path_distance(self, coords: list[tuple[float, float]]) -> float:
        dist = 0.0
        for (la1, lo1), (la2, lo2) in zip(coords[:-1], coords[1:]):
            d_km, _ = self._travel_between(la1, lo1, la2, lo2)
            dist += d_km
        return dist

    def assign_and_sequence(
        self,
        orders_df: pd.DataFrame,
        routes_df: pd.DataFrame,
        policy_agent: PolicyAgent,
        scenario_name: str,
        area: str,
        override_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, pd.DataFrame]:
        routes_df = routes_df.copy().reset_index(drop=True)
        orders_df = orders_df.copy().reset_index(drop=True)

        route_states: Dict[str, Dict[str, Any]] = {
            r["Name"]: {"assigned_count": 0, "current_minutes": 0.0, "assigned_orders": []}
            for _, r in routes_df.iterrows()
        }

        assignment_records: List[Dict[str, Any]] = []
        sort_orders = orders_df.sort_values(
            by=["SLAFlag", "Priority", "TimeWindowEnd1"],
            ascending=[False, False, True],
            na_position="last",
        )

        for _, order in sort_orders.iterrows():
            weights = policy_agent.weights_for_order(order, override=override_weights)
            candidates = routes_df.copy()

            if not pd.isna(order.get("PreferredDepot")):
                same_depot = candidates.loc[candidates["StartDepotName"].astype(str) == str(order.get("PreferredDepot"))]
                if not same_depot.empty:
                    candidates = same_depot

            feasible = []
            for _, route in candidates.iterrows():
                state = route_states[route["Name"]]
                if state["assigned_count"] >= int(route.get("MaxOrderCount", 0)):
                    continue
                score = self._score_route(order, route, weights, state)
                feasible.append((score, route["Name"]))

            if not feasible:
                assignment_records.append(
                    {"Scenario": scenario_name, "Area": area, "OrderId": order["OrderId"], "RouteName": None, "Assigned": 0, "Reason": "No feasible route"}
                )
                continue

            feasible.sort(key=lambda x: x[0])
            best_route_name = feasible[0][1]
            route_states[best_route_name]["assigned_count"] += 1
            route_states[best_route_name]["current_minutes"] += float(order.get("ServiceTime", 45))
            route_states[best_route_name]["assigned_orders"].append(order["OrderId"])
            assignment_records.append(
                {"Scenario": scenario_name, "Area": area, "OrderId": order["OrderId"], "AssignedRouteName": best_route_name, "Assigned": 1, "Reason": "OSRM road-weighted assignment"}
            )

        assignment_df = pd.DataFrame(assignment_records)
        orders_enriched = orders_df.merge(
            assignment_df[["OrderId", "AssignedRouteName", "Assigned", "Reason"]], on="OrderId", how="left"
        )
        orders_enriched["Assigned"] = orders_enriched["Assigned"].fillna(0).astype(int)
        orders_enriched["AssignedRouteName"] = orders_enriched["AssignedRouteName"].fillna("UNASSIGNED")
        orders_enriched["Scenario"] = scenario_name
        orders_enriched["Area"] = area

        route_rows: List[Dict[str, Any]] = []
        stop_rows: List[Dict[str, Any]] = []
        for _, route in routes_df.iterrows():
            route_name = route["Name"]
            assigned_orders = orders_enriched.loc[orders_enriched["AssignedRouteName"] == route_name].copy()
            if assigned_orders.empty:
                continue

            depot_name = str(route.get("StartDepotName"))
            depot = self.depot_lookup.get(depot_name, {})
            depot_lat = float(depot.get("Latitude", assigned_orders["Latitude"].mean()))
            depot_lon = float(depot.get("Longitude", assigned_orders["Longitude"].mean()))

            assigned_orders = assigned_orders.sort_values(
                by=["TimeWindowStart1", "Priority"],
                ascending=[True, False],
                na_position="last",
            ).reset_index(drop=True)

            ordered_idx = []
            remaining = assigned_orders.copy()
            current_lat, current_lon = depot_lat, depot_lon
            while not remaining.empty:
                dists = []
                for _, rr in remaining.iterrows():
                    d_km, _ = self._travel_between(current_lat, current_lon, float(rr["Latitude"]), float(rr["Longitude"]))
                    dists.append(d_km)
                pick = int(np.argmin(dists))
                row = remaining.iloc[pick]
                ordered_idx.append(int(row.name))
                current_lat, current_lon = float(row["Latitude"]), float(row["Longitude"])
                remaining = remaining.drop(index=row.name)

            assigned_orders = assigned_orders.loc[ordered_idx].reset_index(drop=True)
            assigned_orders = self._improve_route_two_opt(assigned_orders, depot_lat, depot_lon)

            current_time = route.get("EarliestStartTime")
            if pd.isna(current_time):
                current_time = pd.Timestamp("2026-04-06 07:00:00")
            current_lat, current_lon = depot_lat, depot_lon

            total_distance = 0.0
            total_travel_minutes = 0.0
            total_late = 0.0
            total_service = 0.0
            end_time = current_time

            for seq, (_, order) in enumerate(assigned_orders.iterrows(), start=1):
                d_km, travel_min = self._travel_between(current_lat, current_lon, float(order["Latitude"]), float(order["Longitude"]))
                arrival = current_time + pd.to_timedelta(travel_min, unit="m")
                tw_start = order.get("TimeWindowStart1")
                tw_end = order.get("TimeWindowEnd1")
                start_service = arrival
                if pd.notna(tw_start) and start_service < tw_start:
                    start_service = tw_start
                lateness = 0.0
                if pd.notna(tw_end) and start_service > tw_end:
                    lateness = (start_service - tw_end).total_seconds() / 60.0
                depart = start_service + pd.to_timedelta(float(order.get("ServiceTime", 45)), unit="m")

                total_distance += d_km
                total_travel_minutes += travel_min
                total_late += lateness
                total_service += float(order.get("ServiceTime", 45))
                current_time = depart
                current_lat, current_lon = float(order["Latitude"]), float(order["Longitude"])
                end_time = depart

                stop_rows.append(
                    {
                        "Scenario": scenario_name,
                        "Area": area,
                        "RouteName": route_name,
                        "Sequence": seq,
                        "OrderId": order["OrderId"],
                        "OrderName": order.get("Name"),
                        "Latitude": order["Latitude"],
                        "Longitude": order["Longitude"],
                        "ArrivalTime": arrival,
                        "ServiceStart": start_service,
                        "DepartTime": depart,
                        "LateMinutes": lateness,
                        "TravelMinutes": travel_min,
                        "ServiceMinutes": float(order.get("ServiceTime", 45)),
                        "Priority": order.get("Priority"),
                        "GoalProfile": order.get("GoalProfile"),
                        "SpecialtyNames": order.get("SpecialtyNames"),
                    }
                )

            return_d, return_min = self._travel_between(current_lat, current_lon, depot_lat, depot_lon)
            total_distance += return_d
            total_travel_minutes += return_min
            end_time = end_time + pd.to_timedelta(return_min, unit="m")
            shift_minutes = (end_time - route.get("EarliestStartTime")).total_seconds() / 60.0 if pd.notna(route.get("EarliestStartTime")) else total_travel_minutes + total_service
            overtime = max(0.0, shift_minutes - float(route.get("OvertimeStartTime", 540)))

            route_rows.append(
                {
                    "Scenario": scenario_name,
                    "Area": area,
                    "RouteName": route_name,
                    "StartDepotName": depot_name,
                    "AssignedOrders": len(assigned_orders),
                    "StartTime": route.get("EarliestStartTime"),
                    "EndTime": end_time,
                    "TotalDistanceKm": total_distance,
                    "TotalTravelMinutes": total_travel_minutes,
                    "TotalServiceMinutes": total_service,
                    "LateMinutes": total_late,
                    "OvertimeMinutes": overtime,
                    "TechnicianRating": route.get("TechnicianRating"),
                    "SpecialtyNames": route.get("SpecialtyNames"),
                }
            )

        routes_out = pd.DataFrame(route_rows)
        stops_out = pd.DataFrame(stop_rows)

        assigned_orders = int(orders_enriched["Assigned"].sum())
        unassigned = int((orders_enriched["Assigned"] == 0).sum())
        routes_used = int(routes_out["RouteName"].nunique()) if not routes_out.empty else 0
        total_distance = float(routes_out["TotalDistanceKm"].sum()) if not routes_out.empty else 0.0
        total_late = float(routes_out["LateMinutes"].sum()) if not routes_out.empty else 0.0
        total_overtime = float(routes_out["OvertimeMinutes"].sum()) if not routes_out.empty else 0.0
        total_travel = float(routes_out["TotalTravelMinutes"].sum()) if not routes_out.empty else 0.0

        summary = pd.DataFrame([{
            "Scenario": scenario_name,
            "Area": area,
            "AssignedOrders": assigned_orders,
            "UnassignedOrders": unassigned,
            "RoutesUsed": routes_used,
            "TotalDistanceKm": total_distance,
            "LateMinutes": total_late,
            "OvertimeMinutes": total_overtime,
            "TotalTravelMinutes": total_travel,
        }])

        return {"scenario_summary": summary, "orders": orders_enriched, "routes": routes_out, "stops": stops_out, "assignments": assignment_df}


def _skill_set(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {x.strip().lower() for x in str(value).replace(";", ",").split(",") if x.strip()}


def _is_public_osrm(base_url: str) -> bool:
    url = (base_url or "").strip().rstrip("/").lower()
    return url in {
        "https://router.project-osrm.org",
        "http://router.project-osrm.org",
    }


def _effective_osrm_scope(config: BackendConfig) -> tuple[int, int]:
    if config.osrm_auto_reduce and _is_public_osrm(config.osrm_base_url):
        return min(config.max_orders, config.osrm_demo_max_orders), min(config.max_routes, config.osrm_demo_max_routes)
    return config.max_orders, config.max_routes



def _base_time_for_routes(routes_df: pd.DataFrame) -> pd.Timestamp:
    if routes_df is not None and not routes_df.empty and "EarliestStartTime" in routes_df.columns:
        ser = pd.to_datetime(routes_df["EarliestStartTime"], errors="coerce").dropna()
        if not ser.empty:
            return pd.Timestamp(ser.min())
    return pd.Timestamp("2026-04-06 07:00:00")


def _safe_timestamp(value: Any, default: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return default if default is not None else pd.Timestamp("2026-04-06 07:00:00")
    return pd.Timestamp(ts)


def _safe_float_compare(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value)
    if pd.isna(value):
        return None
    try:
        return round(float(value), 6)
    except Exception:
        return str(value).strip()


def _rows_changed(left: pd.Series, right: pd.Series, cols: List[str]) -> bool:
    for col in cols:
        lv = _safe_float_compare(left.get(col))
        rv = _safe_float_compare(right.get(col))
        if lv != rv:
            return True
    return False


def _changed_order_ids(base_orders: pd.DataFrame, scenario_orders: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    compare_cols = [
        "ServiceTime", "TimeWindowStart1", "TimeWindowEnd1", "Priority", "PreferredDepot",
        "SLAFlag", "SpecialtyNames", "WorkType", "Latitude", "Longitude", "GoalProfile",
    ]
    left = base_orders.copy()
    right = scenario_orders.copy()
    left["OrderId"] = left["OrderId"].astype(str)
    right["OrderId"] = right["OrderId"].astype(str)
    left = left.set_index("OrderId", drop=False)
    right = right.set_index("OrderId", drop=False)
    left_ids = set(left.index.astype(str))
    right_ids = set(right.index.astype(str))
    added = right_ids - left_ids
    removed = left_ids - right_ids
    changed: set[str] = set()
    for oid in sorted(left_ids & right_ids):
        if _rows_changed(left.loc[oid], right.loc[oid], compare_cols):
            changed.add(str(oid))
    return changed, added, removed


def _changed_route_names(base_routes: pd.DataFrame, scenario_routes: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    compare_cols = [
        "EarliestStartTime", "LatestStartTime", "MaxOrderCount", "StartDepotName", "EndDepotName",
        "TechnicianRating", "SpecialtyNames", "OvertimeStartTime", "MaxTotalTime",
    ]
    left = base_routes.copy()
    right = scenario_routes.copy()
    left["Name"] = left["Name"].astype(str)
    right["Name"] = right["Name"].astype(str)
    left = left.set_index("Name", drop=False)
    right = right.set_index("Name", drop=False)
    left_ids = set(left.index.astype(str))
    right_ids = set(right.index.astype(str))
    added = right_ids - left_ids
    removed = left_ids - right_ids
    changed: set[str] = set()
    for rid in sorted(left_ids & right_ids):
        if _rows_changed(left.loc[rid], right.loc[rid], compare_cols):
            changed.add(str(rid))
    return changed, added, removed


def _route_depot_lookup(routes_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if routes_df is None or routes_df.empty:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in routes_df.iterrows():
        out[str(row.get("Name", row.get("RouteName", "")))] = row.to_dict()
    return out


def _nearest_route_names(order_rows: pd.DataFrame, scenario_routes: pd.DataFrame, depots_df: pd.DataFrame, top_n: int = 3) -> List[str]:
    if order_rows.empty or scenario_routes.empty or depots_df.empty:
        return []
    depot_lookup = depots_df.set_index("Name")[["Latitude", "Longitude"]].to_dict("index")
    target_lat = float(order_rows["Latitude"].astype(float).mean())
    target_lon = float(order_rows["Longitude"].astype(float).mean())
    scored = []
    for _, route in scenario_routes.iterrows():
        depot = depot_lookup.get(str(route.get("StartDepotName")), {})
        dep_lat = float(depot.get("Latitude", target_lat))
        dep_lon = float(depot.get("Longitude", target_lon))
        d_km = float(haversine_km(dep_lat, dep_lon, target_lat, target_lon))
        scored.append((d_km, str(route.get("Name"))))
    scored.sort(key=lambda x: x[0])
    return [name for _, name in scored[: max(1, top_n)]]


def _prepare_locked_prefixes(
    scenario_key: str,
    subset_orders: pd.DataFrame,
    subset_routes: pd.DataFrame,
    scenario_orders: pd.DataFrame,
    scenario_routes: pd.DataFrame,
    baseline_result: Dict[str, pd.DataFrame],
    depots_df: pd.DataFrame,
) -> Dict[str, Any]:
    changed_orders, added_orders, removed_orders = _changed_order_ids(subset_orders, scenario_orders)
    changed_routes, _, removed_routes = _changed_route_names(subset_routes, scenario_routes)

    baseline_orders = baseline_result["orders"].copy()
    baseline_orders["OrderId"] = baseline_orders["OrderId"].astype(str)
    baseline_stops = baseline_result["stops"].copy()
    if not baseline_stops.empty:
        baseline_stops["OrderId"] = baseline_stops["OrderId"].astype(str)
        baseline_stops["RouteName"] = baseline_stops["RouteName"].astype(str)
    baseline_routes = baseline_result["routes"].copy()
    if not baseline_routes.empty:
        baseline_routes["RouteName"] = baseline_routes["RouteName"].astype(str)

    impacted_order_ids = set(changed_orders) | set(added_orders)
    impacted_route_names = set(changed_routes) | set(removed_routes)

    if not baseline_orders.empty:
        base_assign = baseline_orders[["OrderId", "AssignedRouteName", "Assigned"]].copy()
        base_assign["AssignedRouteName"] = base_assign["AssignedRouteName"].astype(str)
        impacted_from_orders = base_assign.loc[
            base_assign["OrderId"].isin(list(set(changed_orders) | set(removed_orders))), "AssignedRouteName"
        ]
        impacted_route_names.update(r for r in impacted_from_orders.tolist() if r and r != "UNASSIGNED")

    scenario_route_names = set(scenario_routes["Name"].astype(str))
    locked_prefixes: Dict[str, pd.DataFrame] = {}
    fully_locked_routes: set[str] = set()

    if not baseline_routes.empty and not baseline_stops.empty:
        stop_counts = baseline_stops.groupby("RouteName").size().to_dict()
        for route_name, route_stops in baseline_stops.sort_values(["RouteName", "Sequence"]).groupby("RouteName"):
            route_name = str(route_name)
            route_stops = route_stops.sort_values("Sequence").copy()
            if route_name not in scenario_route_names:
                locked_prefixes[route_name] = route_stops.iloc[0:0].copy()
                continue
            if route_name in changed_routes or route_name in removed_routes:
                locked_prefixes[route_name] = route_stops.iloc[0:0].copy()
                continue

            impacted_here = route_stops.loc[route_stops["OrderId"].isin(list(impacted_order_ids))].copy()
            if impacted_here.empty:
                locked_prefixes[route_name] = route_stops.copy()
                fully_locked_routes.add(route_name)
            else:
                first_seq = int(impacted_here["Sequence"].min())
                locked_prefixes[route_name] = route_stops.loc[route_stops["Sequence"] < first_seq].copy()

        if scenario_key in {"RT_1", "RT_6", "RT_7"}:
            scenario_order_lookup = scenario_orders.copy()
            scenario_order_lookup["OrderId"] = scenario_order_lookup["OrderId"].astype(str)
            newly_open_ids = list((set(added_orders) | set(changed_orders) | set(removed_orders)) & set(scenario_order_lookup["OrderId"].astype(str)))
            newly_open_rows = scenario_order_lookup.loc[scenario_order_lookup["OrderId"].isin(newly_open_ids)].copy()
            candidate_routes = _nearest_route_names(newly_open_rows, scenario_routes, depots_df, top_n=3)
            for route_name in candidate_routes:
                route_stops = baseline_stops.loc[baseline_stops["RouteName"].astype(str) == str(route_name)].sort_values("Sequence").copy()
                if route_stops.empty:
                    continue
                tail_size = min(2, len(route_stops))
                new_locked = route_stops.iloc[: max(0, len(route_stops) - tail_size)].copy()
                locked_prefixes[route_name] = new_locked
                if len(new_locked) < stop_counts.get(route_name, 0):
                    fully_locked_routes.discard(route_name)

    locked_order_ids: set[str] = set()
    for route_name, locked_df in locked_prefixes.items():
        if locked_df is not None and not locked_df.empty:
            locked_order_ids.update(locked_df["OrderId"].astype(str).tolist())

    return {
        "locked_prefixes": locked_prefixes,
        "fully_locked_routes": fully_locked_routes,
        "changed_order_ids": changed_orders,
        "added_order_ids": added_orders,
        "removed_order_ids": removed_orders,
        "changed_route_names": changed_routes,
        "removed_route_names": removed_routes,
        "locked_order_ids": locked_order_ids,
        "impacted_route_names": impacted_route_names,
    }





def _urgency_value(order: pd.Series) -> float:
    priority = float(order.get("Priority", 3))
    sla_flag = float(order.get("SLAFlag", 0))
    tw_start = pd.to_datetime(order.get("TimeWindowStart1"), errors="coerce")
    tw_end = pd.to_datetime(order.get("TimeWindowEnd1"), errors="coerce")
    window_span = 240.0
    if pd.notna(tw_start) and pd.notna(tw_end):
        window_span = max(15.0, (tw_end - tw_start).total_seconds() / 60.0)
    return float(priority * 4.0 + sla_flag * 12.0 + max(0.0, 180.0 - window_span) / 20.0)

def _hybrid_filter_candidate_routes(
    order: pd.Series,
    routes_df: pd.DataFrame,
    route_states: Dict[str, Dict[str, Any]],
    strict_skill_bias: bool,
    max_candidates: int,
) -> pd.DataFrame:
    candidates = routes_df.copy()
    preferred_depot = order.get("PreferredDepot")
    if pd.notna(preferred_depot):
        same_depot = candidates.loc[candidates["StartDepotName"].astype(str) == str(preferred_depot)].copy()
        if not same_depot.empty:
            candidates = same_depot

    order_skills = _skill_set(order.get("SpecialtyNames") or order.get("WorkType"))
    if order_skills:
        skill_mask = candidates["SpecialtyNames"].apply(lambda value: bool(_skill_set(value).intersection(order_skills)))
        skill_matches = candidates.loc[skill_mask].copy()
        if strict_skill_bias and not skill_matches.empty:
            candidates = skill_matches

    target_lat = float(order.get("Latitude", 0.0))
    target_lon = float(order.get("Longitude", 0.0))
    shortlist_rows: List[Tuple[float, int]] = []
    for idx, route in candidates.iterrows():
        route_name = str(route.get("Name"))
        state = route_states.get(route_name, {})
        anchor_lat = float(state.get("anchor_lat", target_lat))
        anchor_lon = float(state.get("anchor_lon", target_lon))
        d_km = float(haversine_km(anchor_lat, anchor_lon, target_lat, target_lon))
        remaining_cap = int(route.get("OriginalMaxOrderCount", route.get("MaxOrderCount", 0))) - int(state.get("locked_count", 0)) - int(state.get("assigned_open_count", 0))
        depot_bonus = -8.0 if str(route.get("StartDepotName")) == str(preferred_depot) else 0.0
        shortlist_rows.append((d_km + max(0, -remaining_cap) * 1000.0 + depot_bonus, idx))
    shortlist_rows.sort(key=lambda x: x[0])
    chosen_idx = [idx for _, idx in shortlist_rows[: max(1, int(max_candidates))]]
    return candidates.loc[chosen_idx].copy().reset_index(drop=True)


def _hybrid_hard_late_cap(order: pd.Series, fallback_minutes: float) -> float:
    user_cap = _scalar_float(order.get("MaxViolationTime1"), np.nan)
    if np.isfinite(user_cap) and user_cap > 0:
        return float(max(5.0, min(120.0, user_cap)))
    return float(max(5.0, fallback_minutes))


def _hybrid_route_fixed_penalty(
    order: pd.Series,
    route: pd.Series,
    weights: Dict[str, float],
    resulting_load_ratio: float,
    apply_continuity_bias: bool,
) -> float:
    route_skills = _skill_set(route.get("SpecialtyNames"))
    order_skills = _skill_set(order.get("SpecialtyNames") or order.get("WorkType"))
    skill_mismatch = 0.0 if (not order_skills or route_skills.intersection(order_skills)) else 1.0
    depot_mismatch = 0.0 if str(order.get("PreferredDepot")) == str(route.get("StartDepotName")) else 1.0
    rating_penalty = (5.5 - float(route.get("TechnicianRating", 4.0))) * 10.0
    baseline_route = str(order.get("BaselineRouteName", "UNASSIGNED"))
    route_name = str(route.get("Name"))
    churn_penalty = 0.0
    continuity_reward = 0.0
    if apply_continuity_bias:
        if baseline_route and baseline_route != "UNASSIGNED" and baseline_route != route_name:
            churn_penalty = 6.0
        elif baseline_route == route_name:
            continuity_reward = -8.0
    priority = float(order.get("Priority", 3))
    sla_flag = float(order.get("SLAFlag", 0))
    urgency_reward = (priority * 2.0) + (sla_flag * 8.0)
    return float(
        weights.get("rating", 0.4) * rating_penalty
        + weights.get("skill", 1.0) * skill_mismatch * 120.0
        + depot_mismatch * 4.0
        + resulting_load_ratio * 12.0
        + churn_penalty
        + continuity_reward
        - urgency_reward
    )


def _hybrid_route_anchor(
    route_row: pd.Series,
    locked_df: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    depot = depots_lookup.get(str(route_row.get("StartDepotName")), {})
    depot_lat = float(depot.get("Latitude", 0.0))
    depot_lon = float(depot.get("Longitude", 0.0))
    route_start = _safe_timestamp(route_row.get("EarliestStartTime"), _base_time_for_routes(pd.DataFrame([route_row])))
    anchor_lat = depot_lat
    anchor_lon = depot_lon
    anchor_time = route_start
    locked_count = 0
    locked_travel = 0.0
    locked_service = 0.0
    locked_late = 0.0
    if locked_df is not None and not locked_df.empty:
        ordered = locked_df.sort_values("Sequence").copy()
        last_locked = ordered.iloc[-1]
        anchor_lat = float(last_locked.get("Latitude", anchor_lat))
        anchor_lon = float(last_locked.get("Longitude", anchor_lon))
        anchor_time = _safe_timestamp(last_locked.get("DepartTime"), route_start)
        locked_count = int(len(ordered))
        locked_travel = float(pd.to_numeric(ordered.get("TravelMinutes"), errors="coerce").fillna(0).sum())
        locked_service = float(pd.to_numeric(ordered.get("ServiceMinutes"), errors="coerce").fillna(0).sum())
        locked_late = float(pd.to_numeric(ordered.get("LateMinutes"), errors="coerce").fillna(0).sum())
    return {
        "depot_lat": depot_lat,
        "depot_lon": depot_lon,
        "route_start": route_start,
        "anchor_lat": anchor_lat,
        "anchor_lon": anchor_lon,
        "anchor_time": anchor_time,
        "locked_count": locked_count,
        "locked_travel_minutes": locked_travel,
        "locked_service_minutes": locked_service,
        "locked_late_minutes": locked_late,
    }


def _hybrid_simulate_sequence(
    route_row: pd.Series,
    locked_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
    speed_kmph: float,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
) -> Dict[str, Any]:
    anchor = _hybrid_route_anchor(route_row, locked_df, depots_lookup)
    current_lat = float(anchor["anchor_lat"])
    current_lon = float(anchor["anchor_lon"])
    current_time = _safe_timestamp(anchor["anchor_time"], anchor["route_start"])

    total_travel = float(anchor["locked_travel_minutes"])
    total_service = float(anchor["locked_service_minutes"])
    total_late = float(anchor["locked_late_minutes"])
    stop_rows: List[Dict[str, Any]] = []
    hard_violation = False
    max_late_seen = 0.0

    if tail_df is None:
        tail_df = pd.DataFrame()

    for _, order in tail_df.iterrows():
        d_km = float(haversine_km(current_lat, current_lon, float(order["Latitude"]), float(order["Longitude"])))
        travel_min = d_km / max(float(speed_kmph), 1.0) * 60.0
        arrival = current_time + pd.to_timedelta(travel_min, unit="m")
        tw_start = pd.to_datetime(order.get("TimeWindowStart1"), errors="coerce")
        tw_end = pd.to_datetime(order.get("TimeWindowEnd1"), errors="coerce")
        start_service = arrival if pd.isna(tw_start) or arrival >= tw_start else tw_start
        lateness = 0.0
        if pd.notna(tw_end) and start_service > tw_end:
            lateness = (start_service - tw_end).total_seconds() / 60.0
        allowed_late = _hybrid_hard_late_cap(order, hard_late_cap_minutes)
        if lateness > allowed_late + 1e-9:
            hard_violation = True
        max_late_seen = max(max_late_seen, lateness)
        service_minutes = float(order.get("ServiceTime", 45))
        depart = start_service + pd.to_timedelta(service_minutes, unit="m")

        total_travel += travel_min
        total_service += service_minutes
        total_late += lateness
        current_lat = float(order["Latitude"])
        current_lon = float(order["Longitude"])
        current_time = depart
        stop_rows.append({
            "OrderId": str(order["OrderId"]),
            "OrderName": order.get("Name"),
            "Latitude": float(order["Latitude"]),
            "Longitude": float(order["Longitude"]),
            "ArrivalTime": arrival,
            "ServiceStart": start_service,
            "DepartTime": depart,
            "LateMinutes": lateness,
            "TravelMinutes": travel_min,
            "ServiceMinutes": service_minutes,
            "Priority": order.get("Priority"),
            "GoalProfile": order.get("GoalProfile"),
            "SpecialtyNames": order.get("SpecialtyNames"),
        })

    if np.isfinite(current_lat) and np.isfinite(current_lon) and np.isfinite(anchor["depot_lat"]) and np.isfinite(anchor["depot_lon"]):
        return_km = float(haversine_km(current_lat, current_lon, anchor["depot_lat"], anchor["depot_lon"]))
        return_min = return_km / max(float(speed_kmph), 1.0) * 60.0
    else:
        return_km = 0.0
        return_min = 0.0
    end_time = current_time + pd.to_timedelta(return_min, unit="m")
    shift_minutes = max(0.0, (end_time - anchor["route_start"]).total_seconds() / 60.0)
    overtime = max(0.0, shift_minutes - float(route_row.get("OvertimeStartTime", 540)))
    hard_shift_limit = float(route_row.get("MaxTotalTime", 720)) + float(shift_buffer_minutes)
    if shift_minutes > hard_shift_limit + 1e-9:
        hard_violation = True

    total_distance = (total_travel + return_min) * max(float(speed_kmph), 1.0) / 60.0
    return {
        "stop_rows": stop_rows,
        "end_time": end_time,
        "shift_minutes": shift_minutes,
        "overtime_minutes": overtime,
        "total_travel_minutes": float(total_travel + return_min),
        "total_service_minutes": float(total_service),
        "total_late_minutes": float(total_late),
        "total_distance_km": float(total_distance),
        "hard_violation": bool(hard_violation),
        "max_late_minutes": float(max_late_seen),
        "strict_feasible": bool((not hard_violation) and float(total_late) <= 1e-9 and float(overtime) <= 1e-9),
        "anchor": anchor,
    }


def _hybrid_metrics_cost(
    metrics: Dict[str, Any],
    *,
    travel_weight: float = 1.35,
    distance_weight: float = 0.20,
    late_weight: float = 12.0,
    overtime_weight: float = 8.0,
    routes_used_weight: float = 0.0,
) -> float:
    hard_penalty = 100000.0 if bool(metrics.get("hard_violation", False)) else 0.0
    late = float(metrics.get("total_late_minutes", 0.0))
    overtime = float(metrics.get("overtime_minutes", 0.0))
    travel = float(metrics.get("total_travel_minutes", 0.0))
    distance = float(metrics.get("total_distance_km", 0.0))
    routes_used = float(metrics.get("routes_used", 0.0))
    return float(
        hard_penalty
        + late * late_weight
        + overtime * overtime_weight
        + travel * travel_weight
        + distance * distance_weight
        + routes_used * routes_used_weight
    )


def _hybrid_metrics_rank(metrics: Dict[str, Any], *, require_zero_late_overtime: bool = True) -> Tuple[float, ...]:
    hard = 1.0 if bool(metrics.get("hard_violation", False)) else 0.0
    late = float(metrics.get("total_late_minutes", 0.0))
    overtime = float(metrics.get("overtime_minutes", 0.0))
    if require_zero_late_overtime:
        return (hard, 1.0 if late > 1e-9 else 0.0, 1.0 if overtime > 1e-9 else 0.0, late, overtime)
    return (hard, late, overtime)


def _hybrid_is_better_metrics(
    candidate_metrics: Dict[str, Any],
    current_metrics: Dict[str, Any],
    *,
    travel_weight: float,
    distance_weight: float,
    require_zero_late_overtime: bool = True,
) -> bool:
    cand_rank = _hybrid_metrics_rank(candidate_metrics, require_zero_late_overtime=require_zero_late_overtime)
    curr_rank = _hybrid_metrics_rank(current_metrics, require_zero_late_overtime=require_zero_late_overtime)
    if cand_rank < curr_rank:
        return True
    if cand_rank > curr_rank:
        return False
    cand_cost = _hybrid_metrics_cost(candidate_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
    curr_cost = _hybrid_metrics_cost(current_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
    if cand_cost + 1e-9 < curr_cost:
        return True
    if curr_cost + 1e-9 < cand_cost:
        return False
    cand_travel = float(candidate_metrics.get("total_travel_minutes", 0.0))
    curr_travel = float(current_metrics.get("total_travel_minutes", 0.0))
    if cand_travel + 1e-9 < curr_travel:
        return True
    if curr_travel + 1e-9 < cand_travel:
        return False
    cand_distance = float(candidate_metrics.get("total_distance_km", 0.0))
    curr_distance = float(current_metrics.get("total_distance_km", 0.0))
    return cand_distance + 1e-9 < curr_distance


def _hybrid_best_position(
    route_row: pd.Series,
    locked_df: pd.DataFrame,
    seq_ids: List[str],
    order_lookup: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
    speed_kmph: float,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
    travel_weight: float,
    distance_weight: float,
    require_zero_late_overtime: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    if len(seq_ids) <= 1:
        tail_df = order_lookup.loc[seq_ids].copy() if seq_ids else pd.DataFrame(columns=order_lookup.columns)
        return seq_ids, _hybrid_simulate_sequence(route_row, locked_df, tail_df, depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)

    best_seq = list(seq_ids)
    best_metrics = _hybrid_simulate_sequence(route_row, locked_df, order_lookup.loc[best_seq].copy(), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)

    for idx, oid in enumerate(seq_ids):
        remaining = [x for i, x in enumerate(seq_ids) if i != idx]
        best_local_seq = None
        best_local_metrics = None
        best_local_cost = None
        for pos in range(len(remaining) + 1):
            candidate = remaining[:]
            candidate.insert(pos, oid)
            cand_metrics = _hybrid_simulate_sequence(route_row, locked_df, order_lookup.loc[candidate].copy(), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
            cand_cost = _hybrid_metrics_cost(cand_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
            if best_local_metrics is None or _hybrid_is_better_metrics(
                cand_metrics,
                best_local_metrics,
                travel_weight=travel_weight,
                distance_weight=distance_weight,
                require_zero_late_overtime=require_zero_late_overtime,
            ):
                best_local_seq = candidate
                best_local_metrics = cand_metrics
                best_local_cost = cand_cost
        if best_local_metrics is not None and _hybrid_is_better_metrics(
            best_local_metrics,
            best_metrics,
            travel_weight=travel_weight,
            distance_weight=distance_weight,
            require_zero_late_overtime=require_zero_late_overtime,
        ):
            best_seq = list(best_local_seq)
            best_metrics = best_local_metrics
            best_cost = float(best_local_cost)
            seq_ids = list(best_seq)
    return best_seq, best_metrics


def _hybrid_total_state_cost(
    route_states: Dict[str, Dict[str, Any]],
    *,
    travel_weight: float,
    distance_weight: float,
) -> float:
    total = 0.0
    active_routes = 0.0
    for state in route_states.values():
        stop_count = int(state.get("locked_count", 0)) + len(state.get("open_sequence", []))
        if stop_count > 0:
            active_routes += 1.0
        total += _hybrid_metrics_cost(
            state.get("metrics", {}),
            travel_weight=travel_weight,
            distance_weight=distance_weight,
        )
    return float(total + active_routes * 6.0)


def _hybrid_compact_sparse_routes(
    route_states: Dict[str, Dict[str, Any]],
    order_lookup: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
    speed_kmph: float,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
    apply_continuity_bias: bool,
    travel_weight: float,
    distance_weight: float,
    route_activation_penalty: float,
    compact_passes: int,
    compact_route_limit: int,
    global_relocate_passes: int = 0,
    global_swap_passes: int = 0,
    require_zero_late_overtime: bool = True,
) -> None:
    # global_relocate_passes/global_swap_passes are accepted here for call compatibility;
    # the actual global exchange phase is executed separately in _hybrid_global_exchange_improve().
    _ = (global_relocate_passes, global_swap_passes, apply_continuity_bias, route_activation_penalty)
    if compact_passes <= 0 or compact_route_limit <= 0:
        return

    def clone_states(states: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        cloned: Dict[str, Dict[str, Any]] = {}
        for route_name, state in states.items():
            cloned[route_name] = {
                "route_row": state["route_row"],
                "locked_df": state["locked_df"].copy(),
                "open_sequence": list(state.get("open_sequence", [])),
                "metrics": dict(state.get("metrics", {})),
                "anchor_lat": state.get("anchor_lat"),
                "anchor_lon": state.get("anchor_lon"),
                "anchor_time": state.get("anchor_time"),
                "locked_count": int(state.get("locked_count", 0)),
                "assigned_open_count": int(state.get("assigned_open_count", 0)),
            }
        return cloned

    def recompute_state(state: Dict[str, Any]) -> None:
        seq = list(state.get("open_sequence", []))
        tail_df = order_lookup.loc[seq].copy() if seq else pd.DataFrame(columns=order_lookup.columns)
        metrics = _hybrid_simulate_sequence(
            state["route_row"],
            state["locked_df"],
            tail_df,
            depots_lookup,
            speed_kmph,
            hard_late_cap_minutes,
            shift_buffer_minutes,
        )
        state["metrics"] = metrics
        state["assigned_open_count"] = len(seq)
        if seq:
            last_row = tail_df.iloc[-1]
            state["anchor_lat"] = float(last_row["Latitude"])
            state["anchor_lon"] = float(last_row["Longitude"])
        else:
            anchor = metrics["anchor"]
            state["anchor_lat"] = anchor["anchor_lat"]
            state["anchor_lon"] = anchor["anchor_lon"]
            state["anchor_time"] = anchor["anchor_time"]

    def best_target_insertion(states: Dict[str, Dict[str, Any]], order_id: str, source_route: str) -> Optional[Tuple[str, List[str], Dict[str, Any], float]]:
        order = order_lookup.loc[order_id]
        best = None
        for target_name, target_state in states.items():
            if target_name == source_route:
                continue
            route_row = target_state["route_row"]
            capacity = int(route_row.get("OriginalMaxOrderCount", route_row.get("MaxOrderCount", 0)))
            if target_state["locked_count"] + len(target_state["open_sequence"]) >= capacity:
                continue
            current_cost = _hybrid_metrics_cost(target_state["metrics"], travel_weight=travel_weight, distance_weight=distance_weight)
            activation_penalty = route_activation_penalty if (target_state["locked_count"] == 0 and len(target_state["open_sequence"]) == 0) else 0.0
            best_local = None
            for pos in range(len(target_state["open_sequence"]) + 1):
                cand_seq = list(target_state["open_sequence"])
                cand_seq.insert(pos, order_id)
                cand_metrics = _hybrid_simulate_sequence(
                    route_row,
                    target_state["locked_df"],
                    order_lookup.loc[cand_seq].copy(),
                    depots_lookup,
                    speed_kmph,
                    hard_late_cap_minutes,
                    shift_buffer_minutes,
                )
                if cand_metrics["hard_violation"]:
                    continue
                if require_zero_late_overtime and (float(cand_metrics.get("total_late_minutes", 0.0)) > 1e-9 or float(cand_metrics.get("overtime_minutes", 0.0)) > 1e-9):
                    continue
                cand_cost = _hybrid_metrics_cost(cand_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
                delta = cand_cost - current_cost + activation_penalty
                if best_local is None or delta + 1e-9 < best_local[-1]:
                    best_local = (target_name, cand_seq, cand_metrics, float(delta))
            if best_local is not None and (best is None or best_local[-1] + 1e-9 < best[-1]):
                best = best_local
        return best

    for _ in range(int(compact_passes)):
        base_total_cost = _hybrid_total_state_cost(route_states, travel_weight=travel_weight, distance_weight=distance_weight)
        candidates = []
        for route_name, state in route_states.items():
            open_count = len(state.get("open_sequence", []))
            if open_count <= 0 or int(state.get("locked_count", 0)) > 0:
                continue
            metrics = state.get("metrics", {})
            travel = float(metrics.get("total_travel_minutes", 0.0))
            travel_per_stop = travel / max(open_count, 1)
            candidates.append((open_count, -travel_per_stop, -travel, route_name))
        candidates.sort()
        improved = False
        for _, _, _, source_route in candidates[: max(1, int(compact_route_limit))]:
            working = clone_states(route_states)
            source_state = working[source_route]
            source_orders = list(source_state.get("open_sequence", []))
            if not source_orders:
                continue
            source_state["open_sequence"] = []
            recompute_state(source_state)
            feasible = True
            for oid in sorted(source_orders, key=lambda x: _urgency_value(order_lookup.loc[x]), reverse=True):
                move = best_target_insertion(working, oid, source_route)
                if move is None:
                    feasible = False
                    break
                target_name, cand_seq, cand_metrics, _ = move
                working[target_name]["open_sequence"] = cand_seq
                working[target_name]["metrics"] = cand_metrics
                working[target_name]["assigned_open_count"] = len(cand_seq)
            if not feasible:
                continue
            if require_zero_late_overtime and any(
                float(st.get("metrics", {}).get("total_late_minutes", 0.0)) > 1e-9 or float(st.get("metrics", {}).get("overtime_minutes", 0.0)) > 1e-9 or bool(st.get("metrics", {}).get("hard_violation", False))
                for st in working.values()
            ):
                continue
            new_total_cost = _hybrid_total_state_cost(working, travel_weight=travel_weight, distance_weight=distance_weight)
            old_travel = sum(float(st.get("metrics", {}).get("total_travel_minutes", 0.0)) for st in route_states.values())
            new_travel = sum(float(st.get("metrics", {}).get("total_travel_minutes", 0.0)) for st in working.values())
            old_distance = sum(float(st.get("metrics", {}).get("total_distance_km", 0.0)) for st in route_states.values())
            new_distance = sum(float(st.get("metrics", {}).get("total_distance_km", 0.0)) for st in working.values())
            if new_total_cost + 1e-6 < base_total_cost or new_travel + 1e-6 < old_travel or new_distance + 1e-6 < old_distance:
                route_states.clear()
                route_states.update(working)
                improved = True
                break
        if not improved:
            break



def _hybrid_global_exchange_improve(
    route_states: Dict[str, Dict[str, Any]],
    order_lookup: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
    speed_kmph: float,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
    travel_weight: float,
    distance_weight: float,
    relocate_passes: int,
    swap_passes: int,
    require_zero_late_overtime: bool = True,
) -> None:
    if relocate_passes <= 0 and swap_passes <= 0:
        return

    def clone_states(states: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        cloned: Dict[str, Dict[str, Any]] = {}
        for route_name, state in states.items():
            cloned[route_name] = {
                "route_row": state["route_row"],
                "locked_df": state["locked_df"].copy(),
                "open_sequence": list(state.get("open_sequence", [])),
                "metrics": dict(state.get("metrics", {})),
                "anchor_lat": state.get("anchor_lat"),
                "anchor_lon": state.get("anchor_lon"),
                "anchor_time": state.get("anchor_time"),
                "locked_count": int(state.get("locked_count", 0)),
                "assigned_open_count": int(state.get("assigned_open_count", 0)),
            }
        return cloned

    def recompute_state(state: Dict[str, Any]) -> None:
        seq = list(state.get("open_sequence", []))
        tail_df = order_lookup.loc[seq].copy() if seq else pd.DataFrame(columns=order_lookup.columns)
        metrics = _hybrid_simulate_sequence(
            state["route_row"],
            state["locked_df"],
            tail_df,
            depots_lookup,
            speed_kmph,
            hard_late_cap_minutes,
            shift_buffer_minutes,
        )
        state["metrics"] = metrics
        state["assigned_open_count"] = len(seq)
        if seq:
            last_row = tail_df.iloc[-1]
            state["anchor_lat"] = float(last_row["Latitude"])
            state["anchor_lon"] = float(last_row["Longitude"])
        else:
            anchor = metrics["anchor"]
            state["anchor_lat"] = anchor["anchor_lat"]
            state["anchor_lon"] = anchor["anchor_lon"]
            state["anchor_time"] = anchor["anchor_time"]

    def route_rank(states: Dict[str, Dict[str, Any]]) -> List[str]:
        scored = []
        for route_name, state in states.items():
            open_seq = list(state.get("open_sequence", []))
            if not open_seq:
                continue
            metrics = state.get("metrics", {})
            travel = float(metrics.get("total_travel_minutes", 0.0))
            distance = float(metrics.get("total_distance_km", 0.0))
            per_stop = travel / max(1, len(open_seq))
            scored.append((-per_stop, -travel, -distance, len(open_seq), route_name))
        scored.sort()
        return [name for *_rest, name in scored]

    def feasible_metrics(metrics: Dict[str, Any]) -> bool:
        if bool(metrics.get("hard_violation", False)):
            return False
        if require_zero_late_overtime and (float(metrics.get("total_late_minutes", 0.0)) > 1e-9 or float(metrics.get("overtime_minutes", 0.0)) > 1e-9):
            return False
        return True

    def total_cost(states: Dict[str, Dict[str, Any]]) -> float:
        return _hybrid_total_state_cost(states, travel_weight=travel_weight, distance_weight=distance_weight)

    for _ in range(int(relocate_passes)):
        improved = False
        active_routes = route_rank(route_states)
        base_cost = total_cost(route_states)
        best_move = None
        best_cost = base_cost
        for source_name in active_routes[: max(6, min(16, len(active_routes)))]:
            source_state = route_states[source_name]
            source_seq = list(source_state.get("open_sequence", []))
            if not source_seq:
                continue
            # high-detour orders first: farthest from their current predecessor
            order_priority = []
            prev_lat = float(source_state.get("metrics", {}).get("anchor", {}).get("depot_lat", 0.0))
            prev_lon = float(source_state.get("metrics", {}).get("anchor", {}).get("depot_lon", 0.0))
            for oid in source_seq:
                row = order_lookup.loc[oid]
                jump = float(haversine_km(prev_lat, prev_lon, float(row["Latitude"]), float(row["Longitude"])))
                order_priority.append((-jump, oid))
                prev_lat = float(row["Latitude"])
                prev_lon = float(row["Longitude"])
            order_priority.sort()
            for _negjump, oid in order_priority[: max(3, min(10, len(order_priority)))]:
                for target_name in active_routes:
                    if target_name == source_name:
                        continue
                    target_state = route_states[target_name]
                    target_route = target_state["route_row"]
                    capacity = int(target_route.get("OriginalMaxOrderCount", target_route.get("MaxOrderCount", 0)))
                    if target_state["locked_count"] + len(target_state.get("open_sequence", [])) >= capacity:
                        continue
                    working = clone_states(route_states)
                    ws = working[source_name]
                    wt = working[target_name]
                    src_seq = list(ws.get("open_sequence", []))
                    if oid not in src_seq:
                        continue
                    src_seq.remove(oid)
                    ws["open_sequence"] = src_seq
                    recompute_state(ws)
                    if not feasible_metrics(ws["metrics"]):
                        continue
                    best_target_seq = None
                    best_target_metrics = None
                    best_target_cost = None
                    tgt_seq_base = list(wt.get("open_sequence", []))
                    for pos in range(len(tgt_seq_base) + 1):
                        cand_seq = tgt_seq_base[:]
                        cand_seq.insert(pos, oid)
                        cand_metrics = _hybrid_simulate_sequence(
                            wt["route_row"],
                            wt["locked_df"],
                            order_lookup.loc[cand_seq].copy(),
                            depots_lookup,
                            speed_kmph,
                            hard_late_cap_minutes,
                            shift_buffer_minutes,
                        )
                        if not feasible_metrics(cand_metrics):
                            continue
                        cand_cost = _hybrid_metrics_cost(cand_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
                        if best_target_cost is None or cand_cost + 1e-9 < best_target_cost:
                            best_target_seq = cand_seq
                            best_target_metrics = cand_metrics
                            best_target_cost = cand_cost
                    if best_target_seq is None:
                        continue
                    wt["open_sequence"] = best_target_seq
                    wt["metrics"] = best_target_metrics
                    wt["assigned_open_count"] = len(best_target_seq)
                    new_cost = total_cost(working)
                    if new_cost + 1e-6 < best_cost:
                        best_cost = new_cost
                        best_move = working
                # keep search bounded
        if best_move is not None:
            route_states.clear()
            route_states.update(best_move)
            improved = True
        if not improved:
            break

    for _ in range(int(swap_passes)):
        improved = False
        active_routes = route_rank(route_states)
        base_cost = total_cost(route_states)
        best_swap = None
        best_cost = base_cost
        limited_routes = active_routes[: max(6, min(12, len(active_routes)))]
        for i, left_name in enumerate(limited_routes):
            for right_name in limited_routes[i + 1:]:
                left_state = route_states[left_name]
                right_state = route_states[right_name]
                left_seq = list(left_state.get("open_sequence", []))
                right_seq = list(right_state.get("open_sequence", []))
                if not left_seq or not right_seq:
                    continue
                left_try = left_seq[: max(2, min(6, len(left_seq)))]
                right_try = right_seq[: max(2, min(6, len(right_seq)))]
                for loid in left_try:
                    for roid in right_try:
                        working = clone_states(route_states)
                        wl = working[left_name]
                        wr = working[right_name]
                        lseq = list(wl.get("open_sequence", []))
                        rseq = list(wr.get("open_sequence", []))
                        li = lseq.index(loid)
                        ri = rseq.index(roid)
                        lseq[li] = roid
                        rseq[ri] = loid
                        left_best_seq, left_best_metrics = _hybrid_best_position(
                            wl["route_row"],
                            wl["locked_df"],
                            lseq,
                            order_lookup,
                            depots_lookup,
                            speed_kmph,
                            hard_late_cap_minutes,
                            shift_buffer_minutes,
                            travel_weight,
                            distance_weight,
                            require_zero_late_overtime=require_zero_late_overtime,
                        )
                        right_best_seq, right_best_metrics = _hybrid_best_position(
                            wr["route_row"],
                            wr["locked_df"],
                            rseq,
                            order_lookup,
                            depots_lookup,
                            speed_kmph,
                            hard_late_cap_minutes,
                            shift_buffer_minutes,
                            travel_weight,
                            distance_weight,
                            require_zero_late_overtime=require_zero_late_overtime,
                        )
                        if not feasible_metrics(left_best_metrics) or not feasible_metrics(right_best_metrics):
                            continue
                        wl["open_sequence"] = left_best_seq
                        wl["metrics"] = left_best_metrics
                        wl["assigned_open_count"] = len(left_best_seq)
                        wr["open_sequence"] = right_best_seq
                        wr["metrics"] = right_best_metrics
                        wr["assigned_open_count"] = len(right_best_seq)
                        new_cost = total_cost(working)
                        if new_cost + 1e-6 < best_cost:
                            best_cost = new_cost
                            best_swap = working
        if best_swap is not None:
            route_states.clear()
            route_states.update(best_swap)
            improved = True
        if not improved:
            break

def _hybrid_local_route_search(
    route_row: pd.Series,
    locked_df: pd.DataFrame,
    seq_ids: List[str],
    order_lookup: pd.DataFrame,
    depots_lookup: Dict[str, Dict[str, Any]],
    speed_kmph: float,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
    passes: int,
    no_improve_rounds: int = 2,
    travel_weight: float = 1.35,
    distance_weight: float = 0.20,
    require_zero_late_overtime: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    seq_ids = [str(x) for x in seq_ids if str(x) in order_lookup.index]
    if len(seq_ids) < 2 or passes <= 0:
        tail_df = order_lookup.loc[seq_ids].copy() if seq_ids else pd.DataFrame(columns=order_lookup.columns)
        return seq_ids, _hybrid_simulate_sequence(route_row, locked_df, tail_df, depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)

    current_seq = list(seq_ids)
    current_metrics = _hybrid_simulate_sequence(route_row, locked_df, order_lookup.loc[current_seq].copy(), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
    current_cost = _hybrid_metrics_cost(current_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
    no_improve_count = 0

    for _ in range(int(passes)):
        improved = False

        best_seq, best_metrics = _hybrid_best_position(
            route_row,
            locked_df,
            current_seq,
            order_lookup,
            depots_lookup,
            speed_kmph,
            hard_late_cap_minutes,
            shift_buffer_minutes,
            travel_weight,
            distance_weight,
            require_zero_late_overtime=require_zero_late_overtime,
        )
        best_cost = _hybrid_metrics_cost(best_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
        if _hybrid_is_better_metrics(
            best_metrics,
            current_metrics,
            travel_weight=travel_weight,
            distance_weight=distance_weight,
            require_zero_late_overtime=require_zero_late_overtime,
        ):
            current_seq = list(best_seq)
            current_metrics = best_metrics
            current_cost = best_cost
            improved = True

        if len(current_seq) >= 3:
            for i in range(len(current_seq) - 1):
                for j in range(i + 1, len(current_seq)):
                    candidate = current_seq[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    cand_metrics = _hybrid_simulate_sequence(route_row, locked_df, order_lookup.loc[candidate].copy(), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
                    cand_cost = _hybrid_metrics_cost(cand_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
                    if _hybrid_is_better_metrics(
                        cand_metrics,
                        current_metrics,
                        travel_weight=travel_weight,
                        distance_weight=distance_weight,
                        require_zero_late_overtime=require_zero_late_overtime,
                    ):
                        current_seq = candidate
                        current_metrics = cand_metrics
                        current_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break

        if len(current_seq) >= 4:
            for i in range(len(current_seq) - 2):
                for j in range(i + 2, len(current_seq)):
                    candidate = current_seq[:i] + list(reversed(current_seq[i:j + 1])) + current_seq[j + 1:]
                    cand_metrics = _hybrid_simulate_sequence(route_row, locked_df, order_lookup.loc[candidate].copy(), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
                    cand_cost = _hybrid_metrics_cost(cand_metrics, travel_weight=travel_weight, distance_weight=distance_weight)
                    if _hybrid_is_better_metrics(
                        cand_metrics,
                        current_metrics,
                        travel_weight=travel_weight,
                        distance_weight=distance_weight,
                        require_zero_late_overtime=require_zero_late_overtime,
                    ):
                        current_seq = candidate
                        current_metrics = cand_metrics
                        current_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max(1, int(no_improve_rounds)):
                break
        else:
            no_improve_count = 0
    return current_seq, current_metrics


def _hybrid_assign_open_orders(
    open_orders_df: pd.DataFrame,
    scenario_routes: pd.DataFrame,
    locked_prefixes: Dict[str, pd.DataFrame],
    policy_agent: PolicyAgent,
    depots_df: pd.DataFrame,
    speed_kmph: float,
    candidate_routes: int,
    frontier_size: int,
    local_search_passes: int,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
    apply_continuity_bias: bool,
    no_improve_rounds: int = 2,
    skip_clean_route_polish: bool = True,
    polish_top_k: int = 3,
    travel_weight: float = 1.35,
    distance_weight: float = 0.20,
    route_activation_penalty: float = 18.0,
    compact_passes: int = 1,
    compact_route_limit: int = 4,
    global_relocate_passes: int = 0,
    global_swap_passes: int = 0,
    require_zero_late_overtime: bool = True,
) -> tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    depots_lookup = depots_df.set_index("Name")[["Latitude", "Longitude"]].to_dict("index") if depots_df is not None and not depots_df.empty else {}
    scenario_routes = scenario_routes.copy().reset_index(drop=True)
    scenario_routes["OriginalMaxOrderCount"] = scenario_routes["MaxOrderCount"]

    order_pool = open_orders_df.copy()
    order_pool["OrderId"] = order_pool["OrderId"].astype(str)
    order_lookup = order_pool.set_index("OrderId", drop=False)

    route_states: Dict[str, Dict[str, Any]] = {}
    for _, route in scenario_routes.iterrows():
        route_name = str(route.get("Name"))
        locked = locked_prefixes.get(route_name, pd.DataFrame()).copy()
        metrics = _hybrid_simulate_sequence(route, locked, pd.DataFrame(columns=order_pool.columns), depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
        anchor = metrics["anchor"]
        route_states[route_name] = {
            "route_row": route,
            "locked_df": locked,
            "open_sequence": [],
            "metrics": metrics,
            "anchor_lat": anchor["anchor_lat"],
            "anchor_lon": anchor["anchor_lon"],
            "anchor_time": anchor["anchor_time"],
            "locked_count": anchor["locked_count"],
            "assigned_open_count": 0,
        }

    pending_ids = order_pool["OrderId"].astype(str).tolist()
    assignments_by_route: Dict[str, List[str]] = {str(r.get("Name")): [] for _, r in scenario_routes.iterrows()}
    assigned_route_name: Dict[str, str] = {}
    assignment_reason: Dict[str, str] = {}

    def insert_candidate(order_id: str, route_name: str, position: int) -> Dict[str, Any]:
        state = route_states[route_name]
        route_row = state["route_row"]
        current_seq = list(state["open_sequence"])
        current_metrics = state["metrics"]
        candidate_seq = current_seq[:]
        candidate_seq.insert(position, order_id)
        tail_df = order_lookup.loc[candidate_seq].copy()
        cand_metrics = _hybrid_simulate_sequence(route_row, state["locked_df"], tail_df, depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
        order = order_lookup.loc[order_id]
        weights = policy_agent.weights_for_order(order)
        capacity = max(1.0, float(route_row.get("OriginalMaxOrderCount", route_row.get("MaxOrderCount", 1))))
        resulting_load = float(state["locked_count"] + len(candidate_seq)) / capacity
        fixed_penalty = _hybrid_route_fixed_penalty(order, route_row, weights, resulting_load, apply_continuity_bias)
        delta_distance = cand_metrics["total_distance_km"] - current_metrics["total_distance_km"]
        delta_travel = cand_metrics["total_travel_minutes"] - current_metrics["total_travel_minutes"]
        delta_late = cand_metrics["total_late_minutes"] - current_metrics["total_late_minutes"]
        delta_overtime = cand_metrics["overtime_minutes"] - current_metrics["overtime_minutes"]
        hard_penalty = 0.0 if not cand_metrics["hard_violation"] else 100000.0
        strict_penalty = 0.0
        strict_ok = (not cand_metrics["hard_violation"]) and float(cand_metrics.get("total_late_minutes", 0.0)) <= 1e-9 and float(cand_metrics.get("overtime_minutes", 0.0)) <= 1e-9
        if require_zero_late_overtime and not strict_ok:
            strict_penalty = 250000.0 + float(cand_metrics.get("total_late_minutes", 0.0)) * 2000.0 + float(cand_metrics.get("overtime_minutes", 0.0)) * 2000.0
        route_activation = route_activation_penalty if (state["locked_count"] == 0 and len(current_seq) == 0) else 0.0
        score = float(
            weights.get("travel", 1.0) * (delta_travel * travel_weight + delta_distance * distance_weight)
            + weights.get("sla", 1.0) * delta_late * 10.0
            + weights.get("overtime", 1.0) * delta_overtime * 8.0
            + fixed_penalty
            + route_activation
            + strict_penalty
            + hard_penalty
        )
        return {
            "route_name": route_name,
            "position": position,
            "score": score,
            "hard_ok": not cand_metrics["hard_violation"],
            "strict_ok": strict_ok,
            "metrics": cand_metrics,
            "sequence": candidate_seq,
        }

    while pending_ids:
        frontier = sorted(pending_ids, key=lambda oid: _urgency_value(order_lookup.loc[oid]), reverse=True)[: max(1, int(frontier_size))]
        chosen: Optional[Dict[str, Any]] = None
        chosen_regret = -1e18
        dead_ids: List[str] = []

        for order_id in frontier:
            order = order_lookup.loc[order_id]
            candidates = _hybrid_filter_candidate_routes(order, scenario_routes, route_states, policy_agent.strict_skill_bias, candidate_routes)
            evaluated: List[Dict[str, Any]] = []
            seen_routes: set[str] = set()
            def _eval_candidates(candidates_df: pd.DataFrame) -> None:
                for _, route in candidates_df.iterrows():
                    route_name = str(route.get("Name"))
                    if route_name in seen_routes:
                        continue
                    seen_routes.add(route_name)
                    state = route_states[route_name]
                    capacity = int(route.get("OriginalMaxOrderCount", route.get("MaxOrderCount", 0)))
                    if state["locked_count"] + len(state["open_sequence"]) >= capacity:
                        continue
                    max_positions = len(state["open_sequence"]) + 1
                    for pos in range(max_positions):
                        evaluated.append(insert_candidate(order_id, route_name, pos))
            _eval_candidates(candidates)
            strict_ok = [opt for opt in evaluated if opt.get("strict_ok", False)]
            if require_zero_late_overtime and not strict_ok and len(seen_routes) < len(scenario_routes):
                _eval_candidates(scenario_routes)
                strict_ok = [opt for opt in evaluated if opt.get("strict_ok", False)]
            if not evaluated:
                dead_ids.append(order_id)
                continue
            hard_ok = [opt for opt in evaluated if opt["hard_ok"]]
            if require_zero_late_overtime and strict_ok:
                pool = strict_ok
            else:
                pool = hard_ok if hard_ok else evaluated
            pool.sort(key=lambda opt: opt["score"])
            best = pool[0]
            second = pool[1]["score"] if len(pool) > 1 else best["score"] + 20.0
            regret = (second - best["score"]) + _urgency_value(order)
            if regret > chosen_regret or (abs(regret - chosen_regret) < 1e-9 and (chosen is None or best["score"] < chosen["score"])):
                chosen = {**best, "order_id": order_id}
                chosen_regret = regret

        if chosen is None:
            for oid in dead_ids + pending_ids:
                assigned_route_name.setdefault(str(oid), "UNASSIGNED")
                assignment_reason.setdefault(str(oid), "No feasible aggressive hybrid route")
            break

        oid = str(chosen["order_id"])
        route_name = str(chosen["route_name"])
        pending_ids.remove(oid)
        state = route_states[route_name]
        state["open_sequence"] = list(chosen["sequence"])
        state["metrics"] = chosen["metrics"]
        state["assigned_open_count"] = len(state["open_sequence"])
        if state["open_sequence"]:
            tail_df = order_lookup.loc[state["open_sequence"]].copy()
            last_row = tail_df.iloc[-1]
            state["anchor_lat"] = float(last_row["Latitude"])
            state["anchor_lon"] = float(last_row["Longitude"])
        assignments_by_route[route_name] = list(state["open_sequence"])
        assigned_route_name[oid] = route_name
        assignment_reason[oid] = "Aggressive hybrid best-insertion assignment"

    route_travel_rank = sorted(
        route_states.items(),
        key=lambda item: float(item[1].get("metrics", {}).get("total_travel_minutes", 0.0)),
        reverse=True,
    )
    top_travel_routes = {name for name, _ in route_travel_rank[: max(0, int(polish_top_k))]}

    _hybrid_compact_sparse_routes(
        route_states=route_states,
        order_lookup=order_lookup,
        depots_lookup=depots_lookup,
        speed_kmph=speed_kmph,
        hard_late_cap_minutes=hard_late_cap_minutes,
        shift_buffer_minutes=shift_buffer_minutes,
        apply_continuity_bias=apply_continuity_bias,
        travel_weight=travel_weight,
        distance_weight=distance_weight,
        route_activation_penalty=route_activation_penalty,
        compact_passes=compact_passes,
        compact_route_limit=compact_route_limit,
        global_relocate_passes=global_relocate_passes,
        global_swap_passes=global_swap_passes,
        require_zero_late_overtime=require_zero_late_overtime,
    )

    _hybrid_global_exchange_improve(
        route_states=route_states,
        order_lookup=order_lookup,
        depots_lookup=depots_lookup,
        speed_kmph=speed_kmph,
        hard_late_cap_minutes=hard_late_cap_minutes,
        shift_buffer_minutes=shift_buffer_minutes,
        travel_weight=travel_weight,
        distance_weight=distance_weight,
        relocate_passes=global_relocate_passes,
        swap_passes=global_swap_passes,
        require_zero_late_overtime=require_zero_late_overtime,
    )

    for route_name, state in route_states.items():
        seq_ids = list(state["open_sequence"])
        if not seq_ids:
            continue
        metrics_before = state.get("metrics", {})
        should_polish = True
        if skip_clean_route_polish:
            has_late = float(metrics_before.get("total_late_minutes", 0.0)) > 0.0
            has_overtime = float(metrics_before.get("overtime_minutes", 0.0)) > 0.0
            is_top_travel = route_name in top_travel_routes
            should_polish = has_late or has_overtime or is_top_travel
        if should_polish:
            seq_ids, metrics = _hybrid_local_route_search(
                state["route_row"],
                state["locked_df"],
                seq_ids,
                order_lookup,
                depots_lookup,
                speed_kmph,
                hard_late_cap_minutes,
                shift_buffer_minutes,
                passes=local_search_passes,
                no_improve_rounds=no_improve_rounds,
                travel_weight=travel_weight,
                distance_weight=distance_weight,
                require_zero_late_overtime=require_zero_late_overtime,
            )
        else:
            metrics = metrics_before
        state["open_sequence"] = seq_ids
        state["metrics"] = metrics
        assignments_by_route[route_name] = list(seq_ids)
        for oid in seq_ids:
            assigned_route_name[str(oid)] = route_name
            assignment_reason[str(oid)] = "Aggressive hybrid polished assignment" if should_polish else "Aggressive hybrid assignment"

    return assignments_by_route, assigned_route_name, assignment_reason, route_states


def _build_hybrid_result(
    scenario_name: str,
    area: str,
    scenario_orders: pd.DataFrame,
    scenario_routes: pd.DataFrame,
    depots_df: pd.DataFrame,
    locked_prefixes: Dict[str, pd.DataFrame],
    fully_locked_routes: set[str],
    baseline_result: Dict[str, pd.DataFrame],
    assignments_by_route: Dict[str, List[str]],
    assigned_route_name: Dict[str, str],
    assignment_reason: Dict[str, str],
    speed_kmph: float,
    use_two_opt: bool,
    hard_late_cap_minutes: float,
    shift_buffer_minutes: float,
) -> Dict[str, pd.DataFrame]:
    depots_lookup = depots_df.set_index("Name")[["Latitude", "Longitude"]].to_dict("index") if depots_df is not None and not depots_df.empty else {}
    order_lookup = scenario_orders.copy()
    order_lookup["OrderId"] = order_lookup["OrderId"].astype(str)
    order_lookup = order_lookup.set_index("OrderId", drop=False)
    route_lookup = scenario_routes.copy()
    route_lookup["Name"] = route_lookup["Name"].astype(str)
    route_lookup = route_lookup.set_index("Name", drop=False)

    baseline_routes = baseline_result["routes"].copy()
    if not baseline_routes.empty:
        baseline_routes["RouteName"] = baseline_routes["RouteName"].astype(str)
        baseline_routes = baseline_routes.set_index("RouteName", drop=False)
    baseline_stops = baseline_result["stops"].copy()
    if not baseline_stops.empty:
        baseline_stops["RouteName"] = baseline_stops["RouteName"].astype(str)
        baseline_stops["OrderId"] = baseline_stops["OrderId"].astype(str)

    route_rows: List[Dict[str, Any]] = []
    stop_rows: List[Dict[str, Any]] = []

    for route_name, route_row in route_lookup.sort_index().iterrows():
        locked = locked_prefixes.get(route_name, pd.DataFrame()).copy()
        open_ids = [oid for oid in assignments_by_route.get(route_name, []) if oid in order_lookup.index]

        if route_name in fully_locked_routes and not open_ids and route_name in baseline_routes.index:
            base_route = baseline_routes.loc[route_name].to_dict()
            base_route["Scenario"] = scenario_name
            base_route["Area"] = area
            route_rows.append(base_route)
            base_stops = baseline_stops.loc[baseline_stops["RouteName"] == route_name].copy()
            if not base_stops.empty:
                base_stops["Scenario"] = scenario_name
                base_stops["Area"] = area
                stop_rows.extend(base_stops.to_dict("records"))
            continue

        if locked is not None and not locked.empty:
            locked = locked.sort_values("Sequence").copy()
            locked["Scenario"] = scenario_name
            locked["Area"] = area
            stop_rows.extend(locked.to_dict("records"))
            seq_counter = int(locked["Sequence"].max())
        else:
            seq_counter = 0

        tail_df = order_lookup.loc[open_ids].copy() if open_ids else pd.DataFrame(columns=order_lookup.columns)
        metrics = _hybrid_simulate_sequence(route_row, locked, tail_df, depots_lookup, speed_kmph, hard_late_cap_minutes, shift_buffer_minutes)
        tail_rows = metrics["stop_rows"]
        for row in tail_rows:
            seq_counter += 1
            stop_rows.append({
                "Scenario": scenario_name,
                "Area": area,
                "RouteName": route_name,
                "Sequence": seq_counter,
                **row,
            })

        if seq_counter == 0:
            continue

        route_rows.append({
            "Scenario": scenario_name,
            "Area": area,
            "RouteName": route_name,
            "StartDepotName": route_row.get("StartDepotName"),
            "AssignedOrders": seq_counter,
            "StartTime": _safe_timestamp(route_row.get("EarliestStartTime"), _base_time_for_routes(pd.DataFrame([route_row]))),
            "EndTime": metrics["end_time"],
            "TotalDistanceKm": metrics["total_distance_km"],
            "TotalTravelMinutes": metrics["total_travel_minutes"],
            "TotalServiceMinutes": metrics["total_service_minutes"],
            "LateMinutes": metrics["total_late_minutes"],
            "OvertimeMinutes": metrics["overtime_minutes"],
            "TechnicianRating": route_row.get("TechnicianRating"),
            "SpecialtyNames": route_row.get("SpecialtyNames"),
        })

    routes_out = pd.DataFrame(route_rows)
    stops_out = pd.DataFrame(stop_rows)

    orders_out = scenario_orders.copy()
    orders_out["OrderId"] = orders_out["OrderId"].astype(str)
    orders_out["Scenario"] = scenario_name
    orders_out["Area"] = area
    orders_out["AssignedRouteName"] = orders_out["OrderId"].map(assigned_route_name).fillna("UNASSIGNED")
    orders_out["Assigned"] = (orders_out["AssignedRouteName"] != "UNASSIGNED").astype(int)
    orders_out["Reason"] = orders_out["OrderId"].map(assignment_reason).fillna("Unassigned after aggressive hybrid optimization")

    summary = pd.DataFrame([{
        "Scenario": scenario_name,
        "Area": area,
        "AssignedOrders": int(orders_out["Assigned"].sum()),
        "UnassignedOrders": int((orders_out["Assigned"] == 0).sum()),
        "RoutesUsed": int(routes_out["RouteName"].nunique()) if not routes_out.empty else 0,
        "TotalDistanceKm": float(routes_out["TotalDistanceKm"].sum()) if not routes_out.empty else 0.0,
        "LateMinutes": float(routes_out["LateMinutes"].sum()) if not routes_out.empty else 0.0,
        "OvertimeMinutes": float(routes_out["OvertimeMinutes"].sum()) if not routes_out.empty else 0.0,
        "TotalTravelMinutes": float(routes_out["TotalTravelMinutes"].sum()) if not routes_out.empty else 0.0,
    }])

    return {
        "scenario_summary": summary,
        "orders": orders_out,
        "routes": routes_out,
        "stops": stops_out,
    }


def _hybrid_reason_maps(
    scenario_orders: pd.DataFrame,
    locked_order_ids: set[str],
    locked_prefixes: Dict[str, pd.DataFrame],
    assignments_by_route: Dict[str, List[str]],
    dynamic_reason: str,
) -> tuple[Dict[str, str], Dict[str, str]]:
    assigned_route_name: Dict[str, str] = {}
    assignment_reason: Dict[str, str] = {}
    for route_name, locked in locked_prefixes.items():
        if locked is not None and not locked.empty:
            for oid in locked["OrderId"].astype(str).tolist():
                assigned_route_name[oid] = route_name
                assignment_reason[oid] = "Locked stable prefix from baseline plan"
    for route_name, order_ids in assignments_by_route.items():
        for oid in order_ids:
            assigned_route_name[str(oid)] = route_name
            assignment_reason[str(oid)] = dynamic_reason
    scenario_ids = set(scenario_orders["OrderId"].astype(str))
    for oid in scenario_ids:
        assigned_route_name.setdefault(str(oid), "UNASSIGNED")
        assignment_reason.setdefault(str(oid), "Unassigned after hybrid rolling repair")
    return assigned_route_name, assignment_reason



def _run_hybrid_solver_once(
    scenario_key: str,
    area: str,
    scenario_orders: pd.DataFrame,
    scenario_routes: pd.DataFrame,
    policy: PolicyAgent,
    depots_df: pd.DataFrame,
    speed_kmph: float,
    use_two_opt: bool,
    locked_prefixes: Optional[Dict[str, pd.DataFrame]] = None,
    fully_locked_routes: Optional[set[str]] = None,
    baseline_result: Optional[Dict[str, pd.DataFrame]] = None,
    candidate_routes: int = 5,
    frontier_size: int = 12,
    local_search_passes: int = 1,
    hard_late_cap_minutes: float = 30.0,
    shift_buffer_minutes: float = 15.0,
    apply_continuity_bias: bool = False,
    no_improve_rounds: int = 2,
    skip_clean_route_polish: bool = True,
    polish_top_k: int = 3,
    travel_weight: float = 1.35,
    distance_weight: float = 0.20,
    route_activation_penalty: float = 18.0,
    compact_passes: int = 1,
    compact_route_limit: int = 4,
    global_relocate_passes: int = 0,
    global_swap_passes: int = 0,
    require_zero_late_overtime: bool = True,
) -> Dict[str, pd.DataFrame]:
    locked_prefixes = locked_prefixes or {}
    fully_locked_routes = fully_locked_routes or set()
    scenario_orders = scenario_orders.copy().reset_index(drop=True)
    scenario_routes = scenario_routes.copy().reset_index(drop=True)
    if "OrderId" not in scenario_orders.columns and "Name" in scenario_orders.columns:
        scenario_orders["OrderId"] = scenario_orders["Name"].astype(str)
    scenario_orders["OrderId"] = scenario_orders["OrderId"].astype(str)

    if baseline_result is not None and "orders" in baseline_result and not baseline_result["orders"].empty:
        base_map = baseline_result["orders"][["OrderId", "AssignedRouteName"]].copy()
        base_map["OrderId"] = base_map["OrderId"].astype(str)
        base_map["AssignedRouteName"] = base_map["AssignedRouteName"].astype(str)
        scenario_orders = scenario_orders.merge(base_map, on="OrderId", how="left", suffixes=("", "_baseline"))
        scenario_orders["BaselineRouteName"] = scenario_orders["AssignedRouteName"].fillna(scenario_orders.get("AssignedRouteName_baseline", "UNASSIGNED"))
        if "AssignedRouteName_baseline" in scenario_orders.columns:
            scenario_orders = scenario_orders.drop(columns=["AssignedRouteName_baseline"], errors="ignore")
        scenario_orders = scenario_orders.drop(columns=["AssignedRouteName"], errors="ignore")
    else:
        scenario_orders["BaselineRouteName"] = "UNASSIGNED"

    locked_order_ids = set()
    for locked in locked_prefixes.values():
        if locked is not None and not locked.empty:
            locked_order_ids.update(locked["OrderId"].astype(str).tolist())

    open_orders = scenario_orders.loc[~scenario_orders["OrderId"].isin(list(locked_order_ids))].copy().reset_index(drop=True)
    assignments_by_route, assigned_open_map, open_reason_map, _ = _hybrid_assign_open_orders(
        open_orders_df=open_orders,
        scenario_routes=scenario_routes,
        locked_prefixes=locked_prefixes,
        policy_agent=policy,
        depots_df=depots_df,
        speed_kmph=speed_kmph,
        candidate_routes=candidate_routes,
        frontier_size=frontier_size,
        local_search_passes=local_search_passes,
        hard_late_cap_minutes=hard_late_cap_minutes,
        shift_buffer_minutes=shift_buffer_minutes,
        apply_continuity_bias=apply_continuity_bias,
        no_improve_rounds=no_improve_rounds,
        skip_clean_route_polish=skip_clean_route_polish,
        polish_top_k=polish_top_k,
        travel_weight=travel_weight,
        distance_weight=distance_weight,
        route_activation_penalty=route_activation_penalty,
        compact_passes=compact_passes,
        compact_route_limit=compact_route_limit,
        require_zero_late_overtime=require_zero_late_overtime,
    )
    assigned_route_name, assignment_reason = _hybrid_reason_maps(
        scenario_orders=scenario_orders,
        locked_order_ids=locked_order_ids,
        locked_prefixes=locked_prefixes,
        assignments_by_route=assignments_by_route,
        dynamic_reason="Aggressive hybrid rolling repair assignment" if baseline_result is not None else "Aggressive hybrid execution-aware assignment",
    )
    assigned_route_name.update(assigned_open_map)
    assignment_reason.update(open_reason_map)

    return _build_hybrid_result(
        scenario_name=scenario_key,
        area=area,
        scenario_orders=scenario_orders,
        scenario_routes=scenario_routes,
        depots_df=depots_df,
        locked_prefixes=locked_prefixes,
        fully_locked_routes=fully_locked_routes,
        baseline_result=baseline_result or {"routes": pd.DataFrame(), "stops": pd.DataFrame()},
        assignments_by_route=assignments_by_route,
        assigned_route_name=assigned_route_name,
        assignment_reason=assignment_reason,
        speed_kmph=speed_kmph,
        use_two_opt=use_two_opt,
        hard_late_cap_minutes=hard_late_cap_minutes,
        shift_buffer_minutes=shift_buffer_minutes,
    )


def run_hybrid_execution_aware_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=config.max_orders,
        max_routes=config.max_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )
    subset_orders = subset_orders.loc[subset_orders["Latitude"].notna() & subset_orders["Longitude"].notna()].copy().reset_index(drop=True)
    subset_routes = subset_routes.copy().reset_index(drop=True)

    policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()

    baseline_result = _run_hybrid_solver_once(
        scenario_key="BASE",
        area="BASE",
        scenario_orders=subset_orders,
        scenario_routes=subset_routes,
        policy=policy,
        depots_df=inputs["depots"],
        speed_kmph=config.speed_kmph,
        use_two_opt=config.use_two_opt,
        locked_prefixes={},
        fully_locked_routes=set(),
        baseline_result=None,
        candidate_routes=config.hybrid_candidate_routes,
        frontier_size=config.hybrid_frontier_size,
        local_search_passes=config.hybrid_local_search_passes,
        hard_late_cap_minutes=config.hybrid_max_hard_late_minutes,
        shift_buffer_minutes=config.hybrid_shift_buffer_minutes,
        apply_continuity_bias=False,
        no_improve_rounds=config.hybrid_no_improve_rounds,
        skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
        polish_top_k=config.hybrid_polish_top_k,
        travel_weight=config.hybrid_travel_focus_weight,
        distance_weight=config.hybrid_distance_focus_weight,
        route_activation_penalty=config.hybrid_route_activation_penalty,
        compact_passes=config.hybrid_compact_passes,
        compact_route_limit=config.hybrid_compact_route_limit,
        global_relocate_passes=config.hybrid_global_relocate_passes,
        global_swap_passes=config.hybrid_global_swap_passes,
        require_zero_late_overtime=config.hybrid_require_zero_late_overtime,
    )

    scenario_results: List[Dict[str, Any]] = []
    timings: List[Dict[str, Any]] = []
    actions_rows: List[Dict[str, Any]] = []
    total = len(DEFAULT_SCENARIOS)

    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"Hybrid Rolling — {scenario_key}")
        started = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        scenario_orders = scenario_orders.loc[scenario_orders["Latitude"].notna() & scenario_orders["Longitude"].notna()].copy().reset_index(drop=True)
        scenario_routes = scenario_routes.copy().reset_index(drop=True)

        if area == "Goal-Based Scheduling":
            goal_policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
            original_weights_for_order = goal_policy.weights_for_order
            if override_weights:
                goal_policy.weights_for_order = lambda order_row, override=override_weights, _orig=original_weights_for_order: _orig(order_row, override=override)
            result = _run_hybrid_solver_once(
                scenario_key=scenario_key,
                area=area,
                scenario_orders=scenario_orders,
                scenario_routes=scenario_routes,
                policy=goal_policy,
                depots_df=inputs["depots"],
                speed_kmph=config.speed_kmph,
                use_two_opt=config.use_two_opt,
                locked_prefixes={},
                fully_locked_routes=set(),
                baseline_result=None,
                candidate_routes=max(config.hybrid_candidate_routes, 6),
                frontier_size=max(config.hybrid_frontier_size, 16),
                local_search_passes=max(config.hybrid_local_search_passes, 1),
                hard_late_cap_minutes=max(10.0, config.hybrid_max_hard_late_minutes * 0.6),
                shift_buffer_minutes=max(5.0, config.hybrid_shift_buffer_minutes * 0.5),
                apply_continuity_bias=False,
                no_improve_rounds=config.hybrid_no_improve_rounds,
                skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
                polish_top_k=config.hybrid_polish_top_k,
            )
        elif scenario_key == "BASE":
            result = baseline_result.copy()
            result = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in result.items()}
            result["scenario_summary"] = result["scenario_summary"].copy()
            result["scenario_summary"]["Scenario"] = "BASE"
            result["scenario_summary"]["Area"] = "BASE"
            for key in ["orders", "routes", "stops"]:
                if key in result and isinstance(result[key], pd.DataFrame) and not result[key].empty:
                    result[key] = result[key].copy()
                    result[key]["Scenario"] = "BASE"
                    result[key]["Area"] = "BASE"
        else:
            repair_context = _prepare_locked_prefixes(
                scenario_key=scenario_key,
                subset_orders=subset_orders,
                subset_routes=subset_routes,
                scenario_orders=scenario_orders,
                scenario_routes=scenario_routes,
                baseline_result=baseline_result,
                depots_df=inputs["depots"],
            )
            result = _run_hybrid_solver_once(
                scenario_key=scenario_key,
                area=area,
                scenario_orders=scenario_orders,
                scenario_routes=scenario_routes,
                policy=policy,
                depots_df=inputs["depots"],
                speed_kmph=config.speed_kmph,
                use_two_opt=config.use_two_opt,
                locked_prefixes=repair_context["locked_prefixes"],
                fully_locked_routes=repair_context["fully_locked_routes"],
                baseline_result=baseline_result,
                candidate_routes=config.hybrid_candidate_routes,
                frontier_size=config.hybrid_frontier_size,
                local_search_passes=max(1, config.hybrid_local_search_passes - 1),
                hard_late_cap_minutes=config.hybrid_max_hard_late_minutes,
                shift_buffer_minutes=config.hybrid_shift_buffer_minutes,
                apply_continuity_bias=True,
                no_improve_rounds=config.hybrid_no_improve_rounds,
                skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
                polish_top_k=config.hybrid_polish_top_k,
            )
            result["scenario_summary"]["LockedOrders"] = len(repair_context["locked_order_ids"])
            result["scenario_summary"]["UnlockedOrders"] = int(len(scenario_orders) - len(repair_context["locked_order_ids"]))

        result["scenario_summary"]["ScenarioLabel"] = scenario_label
        result["scenario_summary"]["Backend"] = "Hybrid Execution-Aware Rolling VRP Solver"
        scenario_results.append(result)
        timings.append({
            "Backend": "Hybrid Execution-Aware Rolling VRP Solver",
            "Scenario": scenario_key,
            "ScenarioLabel": scenario_label,
            "RuntimeSeconds": time.perf_counter() - started,
        })
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({
                "Backend": "Hybrid Execution-Aware Rolling VRP Solver",
                "Scenario": scenario_key,
                "Action": action,
            })

    bundle = _build_common_bundle(
        "Hybrid Execution-Aware Rolling VRP Solver",
        inputs,
        subset_orders,
        subset_routes,
        scenario_results,
        timings,
        actions_rows,
        config,
    )
    bundle["run_meta"]["repair_mode"] = "freeze_stable_prefix_then_regret_repair"
    bundle["run_meta"]["baseline_backend"] = "hybrid_execution_aware"
    return bundle


def _adaptive_result_score(result: Dict[str, pd.DataFrame]) -> tuple[float, ...]:
    summary = result.get("scenario_summary", pd.DataFrame())
    if summary is None or summary.empty:
        return (1e9,)
    row = summary.iloc[0]
    unassigned = float(row.get("UnassignedOrders", 0.0))
    late = float(row.get("LateMinutes", 0.0))
    overtime = float(row.get("OvertimeMinutes", 0.0))
    travel = float(row.get("TotalTravelMinutes", 0.0))
    distance = float(row.get("TotalDistanceKm", 0.0))
    routes = float(row.get("RoutesUsed", 0.0))
    return (unassigned, 1.0 if late > 0 else 0.0, 1.0 if overtime > 0 else 0.0, late, overtime, travel, distance, routes)


def _copy_result_frames(result: Dict[str, Any]) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def run_adaptive_execution_aware_metaheuristic_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=config.max_orders,
        max_routes=config.max_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )
    subset_orders = subset_orders.loc[subset_orders["Latitude"].notna() & subset_orders["Longitude"].notna()].copy().reset_index(drop=True)
    subset_routes = subset_routes.copy().reset_index(drop=True)

    policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()

    baseline_result = _run_hybrid_solver_once(
        scenario_key="BASE",
        area="BASE",
        scenario_orders=subset_orders,
        scenario_routes=subset_routes,
        policy=policy,
        depots_df=inputs["depots"],
        speed_kmph=config.speed_kmph,
        use_two_opt=config.use_two_opt,
        locked_prefixes={},
        fully_locked_routes=set(),
        baseline_result=None,
        candidate_routes=max(config.hybrid_candidate_routes, 6),
        frontier_size=max(config.hybrid_frontier_size, 16),
        local_search_passes=max(config.hybrid_local_search_passes, 1),
        hard_late_cap_minutes=max(8.0, min(config.hybrid_max_hard_late_minutes, 20.0)),
        shift_buffer_minutes=max(5.0, min(config.hybrid_shift_buffer_minutes, 10.0)),
        apply_continuity_bias=False,
        no_improve_rounds=config.hybrid_no_improve_rounds,
        skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
        polish_top_k=config.hybrid_polish_top_k,
        travel_weight=config.hybrid_travel_focus_weight,
        distance_weight=config.hybrid_distance_focus_weight,
        route_activation_penalty=config.hybrid_route_activation_penalty,
        compact_passes=config.hybrid_compact_passes,
        compact_route_limit=config.hybrid_compact_route_limit,
        global_relocate_passes=config.hybrid_global_relocate_passes,
        global_swap_passes=config.hybrid_global_swap_passes,
        require_zero_late_overtime=config.hybrid_require_zero_late_overtime,
    )

    scenario_results: List[Dict[str, Any]] = []
    timings: List[Dict[str, Any]] = []
    actions_rows: List[Dict[str, Any]] = []
    total = len(DEFAULT_SCENARIOS)

    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"Adaptive Metaheuristic — {scenario_key}")
        started = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        scenario_orders = scenario_orders.loc[scenario_orders["Latitude"].notna() & scenario_orders["Longitude"].notna()].copy().reset_index(drop=True)
        scenario_routes = scenario_routes.copy().reset_index(drop=True)

        candidates: List[Dict[str, pd.DataFrame]] = []

        candidate_specs = [
            {
                "candidate_routes": max(config.hybrid_candidate_routes, 5),
                "frontier_size": max(config.hybrid_frontier_size, 12),
                "local_search_passes": max(config.hybrid_local_search_passes, 1),
                "hard_late_cap_minutes": max(10.0, min(config.hybrid_max_hard_late_minutes, 20.0)),
                "shift_buffer_minutes": max(5.0, min(config.hybrid_shift_buffer_minutes, 10.0)),
            },
            {
                "candidate_routes": max(config.hybrid_candidate_routes + 1, 7),
                "frontier_size": max(config.hybrid_frontier_size + 4, 16),
                "local_search_passes": max(config.hybrid_local_search_passes, 1),
                "hard_late_cap_minutes": max(8.0, min(config.hybrid_max_hard_late_minutes, 16.0)),
                "shift_buffer_minutes": max(4.0, min(config.hybrid_shift_buffer_minutes, 8.0)),
            },
        ]

        if area == "Goal-Based Scheduling":
            goal_policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
            original_weights_for_order = goal_policy.weights_for_order
            if override_weights:
                goal_policy.weights_for_order = lambda order_row, override=override_weights, _orig=original_weights_for_order: _orig(order_row, override=override)
            for spec in candidate_specs:
                candidates.append(_run_hybrid_solver_once(
                    scenario_key=scenario_key,
                    area=area,
                    scenario_orders=scenario_orders,
                    scenario_routes=scenario_routes,
                    policy=goal_policy,
                    depots_df=inputs["depots"],
                    speed_kmph=config.speed_kmph,
                    use_two_opt=config.use_two_opt,
                    locked_prefixes={},
                    fully_locked_routes=set(),
                    baseline_result=None,
                    candidate_routes=spec["candidate_routes"],
                    frontier_size=spec["frontier_size"],
                    local_search_passes=spec["local_search_passes"],
                    hard_late_cap_minutes=spec["hard_late_cap_minutes"],
                    shift_buffer_minutes=spec["shift_buffer_minutes"],
                    apply_continuity_bias=False,
                    no_improve_rounds=config.hybrid_no_improve_rounds,
                    skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
                    polish_top_k=config.hybrid_polish_top_k,
                ))
        elif scenario_key == "BASE":
            candidates.append(_copy_result_frames(baseline_result))
            candidates[-1]["scenario_summary"] = candidates[-1]["scenario_summary"].copy()
            candidates[-1]["scenario_summary"]["Scenario"] = "BASE"
            candidates[-1]["scenario_summary"]["Area"] = "BASE"
            for key in ["orders", "routes", "stops"]:
                if key in candidates[-1] and isinstance(candidates[-1][key], pd.DataFrame) and not candidates[-1][key].empty:
                    candidates[-1][key] = candidates[-1][key].copy()
                    candidates[-1][key]["Scenario"] = "BASE"
                    candidates[-1][key]["Area"] = "BASE"
        else:
            repair_context = _prepare_locked_prefixes(
                scenario_key=scenario_key,
                subset_orders=subset_orders,
                subset_routes=subset_routes,
                scenario_orders=scenario_orders,
                scenario_routes=scenario_routes,
                baseline_result=baseline_result,
                depots_df=inputs["depots"],
            )
            # rolling candidate with continuity
            for spec in candidate_specs[:2]:
                candidates.append(_run_hybrid_solver_once(
                    scenario_key=scenario_key,
                    area=area,
                    scenario_orders=scenario_orders,
                    scenario_routes=scenario_routes,
                    policy=policy,
                    depots_df=inputs["depots"],
                    speed_kmph=config.speed_kmph,
                    use_two_opt=config.use_two_opt,
                    locked_prefixes=repair_context["locked_prefixes"],
                    fully_locked_routes=repair_context["fully_locked_routes"],
                    baseline_result=baseline_result,
                    candidate_routes=spec["candidate_routes"],
                    frontier_size=spec["frontier_size"],
                    local_search_passes=max(2, spec["local_search_passes"]),
                    hard_late_cap_minutes=spec["hard_late_cap_minutes"],
                    shift_buffer_minutes=spec["shift_buffer_minutes"],
                    apply_continuity_bias=True,
                    no_improve_rounds=config.hybrid_no_improve_rounds,
                    skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
                    polish_top_k=config.hybrid_polish_top_k,
                ))
            # more global candidate without continuity/locks to escape local minima
            for spec in candidate_specs[1:]:
                candidates.append(_run_hybrid_solver_once(
                    scenario_key=scenario_key,
                    area=area,
                    scenario_orders=scenario_orders,
                    scenario_routes=scenario_routes,
                    policy=policy,
                    depots_df=inputs["depots"],
                    speed_kmph=config.speed_kmph,
                    use_two_opt=config.use_two_opt,
                    locked_prefixes={},
                    fully_locked_routes=set(),
                    baseline_result=None,
                    candidate_routes=spec["candidate_routes"],
                    frontier_size=spec["frontier_size"],
                    local_search_passes=max(3, spec["local_search_passes"]),
                    hard_late_cap_minutes=spec["hard_late_cap_minutes"],
                    shift_buffer_minutes=spec["shift_buffer_minutes"],
                    apply_continuity_bias=False,
                    no_improve_rounds=config.hybrid_no_improve_rounds,
                    skip_clean_route_polish=config.hybrid_skip_clean_route_polish,
                    polish_top_k=config.hybrid_polish_top_k,
                ))

        result = min(candidates, key=_adaptive_result_score)
        result = _copy_result_frames(result)
        result["scenario_summary"] = result["scenario_summary"].copy()
        result["scenario_summary"]["ScenarioLabel"] = scenario_label
        result["scenario_summary"]["Backend"] = "Adaptive Execution-Aware Metaheuristic Solver"
        scenario_results.append(result)
        timings.append({
            "Backend": "Adaptive Execution-Aware Metaheuristic Solver",
            "Scenario": scenario_key,
            "ScenarioLabel": scenario_label,
            "RuntimeSeconds": time.perf_counter() - started,
        })
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({
                "Backend": "Adaptive Execution-Aware Metaheuristic Solver",
                "Scenario": scenario_key,
                "Action": action,
            })

    bundle = _build_common_bundle(
        "Adaptive Execution-Aware Metaheuristic Solver",
        inputs,
        subset_orders,
        subset_routes,
        scenario_results,
        timings,
        actions_rows,
        config,
    )
    bundle["run_meta"]["repair_mode"] = "adaptive_metaheuristic_multi_candidate_search"
    bundle["run_meta"]["baseline_backend"] = "adaptive_execution_aware_metaheuristic"
    return bundle



def _scalar_float(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return float(default)


def _scalar_int(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return int(default)
    except Exception:
        pass
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _scenario_base_time(orders_df: pd.DataFrame, routes_df: pd.DataFrame) -> pd.Timestamp:
    candidates: List[pd.Timestamp] = []
    for col in ["TimeWindowStart1", "TimeWindowEnd1"]:
        if col in orders_df.columns:
            ser = pd.to_datetime(orders_df[col], errors="coerce").dropna()
            if not ser.empty:
                candidates.append(pd.Timestamp(ser.min()))
    for col in ["EarliestStartTime", "LatestStartTime"]:
        if col in routes_df.columns:
            ser = pd.to_datetime(routes_df[col], errors="coerce").dropna()
            if not ser.empty:
                candidates.append(pd.Timestamp(ser.min()))
    if candidates:
        return min(candidates)
    return pd.Timestamp("2026-04-06 07:00:00")


def _minute_offset(value: Any, base_ts: pd.Timestamp, default: int = 0) -> int:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return int(default)
    return max(0, int(round((pd.Timestamp(ts) - base_ts).total_seconds() / 60.0)))


def _priority_prize(order_row: pd.Series, override_weights: Optional[Dict[str, float]] = None) -> int:
    priority = _scalar_int(order_row.get("Priority"), 3)
    sla_flag = _scalar_int(order_row.get("SLAFlag"), 0)
    base = 2000 + priority * 400 + sla_flag * 1200
    if override_weights:
        base *= 1.0 + 0.10 * max(override_weights.get("sla", 0.0), 0.0)
        base *= 1.0 + 0.04 * max(override_weights.get("travel", 0.0), 0.0)
    return max(500, int(round(base)))


def run_pyvrp_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    try:
        import pyvrp
        from pyvrp.stop import MaxRuntime
    except Exception as exc:
        raise RuntimeError(
            "PyVRP backend needs the `pyvrp` package in the same environment. Install it with `pip install pyvrp`."
        ) from exc

    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=config.max_orders,
        max_routes=config.max_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )

    subset_orders = subset_orders.loc[
        subset_orders["Latitude"].notna() & subset_orders["Longitude"].notna()
    ].copy().reset_index(drop=True)
    subset_routes = subset_routes.copy().reset_index(drop=True)

    if subset_routes.empty:
        raise RuntimeError("PyVRP backend has no route rows in the selected scope.")
    if subset_orders.empty:
        raise RuntimeError("PyVRP backend has no geocoded order rows in the selected scope.")

    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()

    depots_df = inputs["depots"].copy()
    depots_df["Name"] = depots_df["Name"].astype(str)

    scenario_results = []
    timings = []
    actions_rows = []

    total = len(DEFAULT_SCENARIOS)

    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"PyVRP — {scenario_key}")

        started = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        scenario_orders = scenario_orders.loc[
            scenario_orders["Latitude"].notna() & scenario_orders["Longitude"].notna()
        ].copy().reset_index(drop=True)
        scenario_routes = scenario_routes.copy().reset_index(drop=True)

        if scenario_orders.empty or scenario_routes.empty:
            empty_orders = scenario_orders.copy()
            if "OrderId" not in empty_orders.columns and "Name" in empty_orders.columns:
                empty_orders["OrderId"] = empty_orders["Name"].astype(str)
            empty_orders["Scenario"] = scenario_key
            empty_orders["Area"] = area
            empty_orders["Assigned"] = 0
            empty_orders["AssignedRouteName"] = "UNASSIGNED"
            summary = pd.DataFrame([{
                "Scenario": scenario_key,
                "Area": area,
                "AssignedOrders": 0,
                "UnassignedOrders": len(empty_orders),
                "RoutesUsed": 0,
                "TotalDistanceKm": 0.0,
                "LateMinutes": 0.0,
                "OvertimeMinutes": 0.0,
                "TotalTravelMinutes": 0.0,
                "ScenarioLabel": scenario_label,
                "Backend": "PyVRP",
            }])
            scenario_results.append(
                {"scenario_summary": summary, "routes": pd.DataFrame(), "stops": pd.DataFrame(), "orders": empty_orders}
            )
            timings.append({"Backend": "PyVRP", "Scenario": scenario_key, "ScenarioLabel": scenario_label, "RuntimeSeconds": time.perf_counter() - started})
            for action in explainer.actions_for(scenario_key):
                actions_rows.append({"Backend": "PyVRP", "Scenario": scenario_key, "Action": action})
            continue

        base_ts = _scenario_base_time(scenario_orders, scenario_routes)
        model = pyvrp.Model()

        depot_names = set(scenario_routes["StartDepotName"].astype(str))
        if "EndDepotName" in scenario_routes.columns:
            depot_names.update(scenario_routes["EndDepotName"].dropna().astype(str).tolist())
        if not depot_names:
            depot_names.update(depots_df["Name"].astype(str).tolist())

        depot_objects: Dict[str, Any] = {}
        depot_coords: Dict[str, Tuple[float, float]] = {}

        for depot_name in sorted(depot_names):
            match = depots_df.loc[depots_df["Name"] == str(depot_name)].head(1)
            if not match.empty:
                dep_lat = _scalar_float(match.iloc[0].get("Latitude"), float(scenario_orders["Latitude"].mean()))
                dep_lon = _scalar_float(match.iloc[0].get("Longitude"), float(scenario_orders["Longitude"].mean()))
            else:
                dep_lat = float(scenario_orders["Latitude"].mean())
                dep_lon = float(scenario_orders["Longitude"].mean())

            depot_obj = model.add_depot(float(dep_lon), float(dep_lat), name=str(depot_name))
            depot_objects[str(depot_name)] = depot_obj
            depot_coords[str(depot_name)] = (float(dep_lat), float(dep_lon))

        if not depot_objects:
            raise RuntimeError("PyVRP backend could not build depot definitions.")

        first_depot_name = next(iter(depot_objects.keys()))

        client_records: List[Dict[str, Any]] = []
        for _, order in scenario_orders.iterrows():
            order_id = str(order.get("OrderId") or order.get("Name") or f"order_{len(client_records) + 1}")
            lat = _scalar_float(order.get("Latitude"), 0.0)
            lon = _scalar_float(order.get("Longitude"), 0.0)
            service_minutes = max(1, _scalar_int(order.get("ServiceTime"), 45))
            tw_early = _minute_offset(order.get("TimeWindowStart1"), base_ts, default=0)
            tw_late_default = tw_early + max(service_minutes + 60, 240)
            tw_late = _minute_offset(order.get("TimeWindowEnd1"), base_ts, default=tw_late_default)
            if tw_late < tw_early:
                tw_late = tw_early + max(service_minutes, 30)

            model.add_client(
                float(lon),
                float(lat),
                delivery=1,
                service_duration=service_minutes,
                tw_early=tw_early,
                tw_late=tw_late,
                prize=_priority_prize(order, override_weights),
                required=True,
                name=order_id,
            )
            client_records.append(
                {
                    "OrderId": order_id,
                    "OrderName": str(order.get("Name", order_id)),
                    "Latitude": lat,
                    "Longitude": lon,
                    "ServiceMinutes": float(service_minutes),
                    "Priority": order.get("Priority"),
                    "GoalProfile": order.get("GoalProfile"),
                    "SpecialtyNames": order.get("SpecialtyNames"),
                    "TimeWindowEndMinute": tw_late,
                }
            )

        travel_weight = 1.0 if not override_weights else max(_scalar_float(override_weights.get("travel"), 1.0), 0.1)
        rating_weight = 0.4 if not override_weights else max(_scalar_float(override_weights.get("rating"), 0.4), 0.1)
        sla_weight = 0.6 if not override_weights else max(_scalar_float(override_weights.get("sla"), 0.6), 0.1)
        overtime_weight = 0.5 if not override_weights else max(_scalar_float(override_weights.get("overtime"), 0.5), 0.1)

        vehicle_type_names: List[str] = []
        vehicle_type_depots: List[str] = []
        vehicle_type_rows: List[Dict[str, Any]] = []

        for _, route in scenario_routes.iterrows():
            route_name = str(route.get("Name", f"Route_{len(vehicle_type_names) + 1}"))
            start_depot_name = str(route.get("StartDepotName") or first_depot_name)
            end_depot_name = str(route.get("EndDepotName") or start_depot_name)
            if start_depot_name not in depot_objects:
                start_depot_name = first_depot_name
            if end_depot_name not in depot_objects:
                end_depot_name = start_depot_name

            capacity = max(1, _scalar_int(route.get("MaxOrderCount"), 40))
            shift_duration = max(60, _scalar_int(route.get("MaxTotalTime"), 720))
            overtime_start = max(0, _scalar_int(route.get("OvertimeStartTime"), shift_duration))
            max_overtime = max(0, shift_duration - overtime_start)

            start_early = _minute_offset(route.get("EarliestStartTime"), base_ts, default=0)
            start_late_default = start_early + max(60, min(shift_duration, 240))
            start_late = _minute_offset(route.get("LatestStartTime"), base_ts, default=start_late_default)
            if start_late < start_early:
                start_late = start_early

            rating = _scalar_float(route.get("TechnicianRating"), 4.0)
            fixed_cost = max(0, int(round((5.5 - rating) * 75.0 * rating_weight)))
            unit_distance_cost = max(1, int(round(100 * travel_weight)))
            unit_duration_cost = max(1, int(round(50 * sla_weight)))
            unit_overtime_cost = max(1, int(round(100 * overtime_weight)))

            model.add_vehicle_type(
                num_available=1,
                capacity=capacity,
                start_depot=depot_objects[start_depot_name],
                end_depot=depot_objects[end_depot_name],
                fixed_cost=fixed_cost,
                tw_early=start_early,
                tw_late=start_late,
                shift_duration=shift_duration,
                unit_distance_cost=unit_distance_cost,
                unit_duration_cost=unit_duration_cost,
                max_overtime=max_overtime,
                unit_overtime_cost=unit_overtime_cost,
                name=route_name,
            )

            vehicle_type_names.append(route_name)
            vehicle_type_depots.append(start_depot_name)
            vehicle_type_rows.append(route.to_dict())

        for frm in model.locations:
            for to in model.locations:
                if frm is to:
                    model.add_edge(frm, to, distance=0, duration=0)
                    continue

                dist_km = float(haversine_km(frm.y, frm.x, to.y, to.x))
                duration_min = dist_km / max(float(config.speed_kmph), 1.0) * 60.0
                model.add_edge(
                    frm,
                    to,
                    distance=max(1, int(round(dist_km * 1000.0))),
                    duration=max(1, int(round(duration_min))),
                )

        result = model.solve(
            stop=MaxRuntime(int(config.pyvrp_max_runtime_seconds)),
            seed=int(config.random_seed),
            collect_stats=False,
            display=False,
        )
        best = result.best

        assigned_route_map: Dict[str, str] = {}
        route_rows: List[Dict[str, Any]] = []
        stop_rows: List[Dict[str, Any]] = []

        client_lookup = {str(rec["OrderId"]): rec for rec in client_records}

        for py_route_idx, route_plan in enumerate(best.routes()):
            vehicle_type_idx = _scalar_int(route_plan.vehicle_type(), py_route_idx)
            if vehicle_type_idx >= len(vehicle_type_names):
                route_name = f"PyVRP_Route_{py_route_idx + 1}"
                route_src = {}
            else:
                route_name = vehicle_type_names[vehicle_type_idx]
                route_src = vehicle_type_rows[vehicle_type_idx]

            start_depot_idx = _scalar_int(route_plan.start_depot(), -1)
            end_depot_idx = _scalar_int(route_plan.end_depot(), start_depot_idx)
            start_depot_name = first_depot_name
            if 0 <= start_depot_idx < len(model.locations):
                start_loc = model.locations[start_depot_idx]
                start_depot_name = str(getattr(start_loc, "name", first_depot_name) or first_depot_name)

            schedule = list(route_plan.schedule()) if hasattr(route_plan, "schedule") else []
            if not schedule:
                visits = list(route_plan.visits()) if hasattr(route_plan, "visits") else []
                cur_idx = start_depot_idx
                cur_time = _scalar_int(route_plan.start_time(), 0)
                for client_idx in visits:
                    client_idx = _scalar_int(client_idx, -1)
                    if client_idx < 0 or client_idx >= len(client_records):
                        continue
                    rec = client_records[client_idx]
                    travel_min = int(round(float(haversine_km(model.locations[cur_idx].y, model.locations[cur_idx].x, rec["Latitude"], rec["Longitude"])) / max(float(config.speed_kmph), 1.0) * 60.0)) if 0 <= cur_idx < len(model.locations) else 0
                    start_service = cur_time + travel_min
                    end_service = start_service + int(round(rec["ServiceMinutes"]))

                    class _Visit:
                        def __init__(self, location, start_service, end_service):
                            self.location = location
                            self.start_service = start_service
                            self.end_service = end_service
                            self.service_duration = end_service - start_service
                            self.wait_duration = 0
                            self.time_warp = 0
                    schedule.append(_Visit(client_idx, start_service, end_service))
                    cur_idx = client_idx
                    cur_time = end_service

            prev_visit = None
            prev_lat = None
            prev_lon = None
            service_total = 0.0
            seq = 0

            for visit in schedule:
                loc_idx = _scalar_int(getattr(visit, "location", -1), -1)
                if loc_idx < 0 or loc_idx >= len(model.locations):
                    prev_visit = visit
                    continue

                loc = model.locations[loc_idx]
                loc_name = str(getattr(loc, "name", loc_idx))
                lat = float(getattr(loc, "y", np.nan))
                lon = float(getattr(loc, "x", np.nan))

                is_depot_visit = loc_idx == start_depot_idx or loc_idx == end_depot_idx

                arrival_min = max(0.0, float(getattr(visit, "start_service", 0)) - float(getattr(visit, "wait_duration", 0)))
                service_start_min = float(getattr(visit, "start_service", arrival_min))
                depart_min = float(getattr(visit, "end_service", service_start_min))

                if prev_visit is None:
                    travel_min = max(0.0, arrival_min - float(_scalar_int(route_plan.start_time(), 0)))
                else:
                    travel_min = max(0.0, arrival_min - float(getattr(prev_visit, "end_service", getattr(prev_visit, "start_service", 0))))
                    if prev_lat is not None and prev_lon is not None and np.isfinite(lat) and np.isfinite(lon):
                        # keep geometry-derived travel for map rows consistent with coordinates
                        travel_min = max(0.0, float(haversine_km(prev_lat, prev_lon, lat, lon)) / max(float(config.speed_kmph), 1.0) * 60.0)

                if is_depot_visit:
                    prev_visit = visit
                    prev_lat, prev_lon = lat, lon
                    continue

                rec = client_lookup.get(loc_name)
                if rec is None:
                    rec = {
                        "OrderId": loc_name,
                        "OrderName": loc_name,
                        "Latitude": lat,
                        "Longitude": lon,
                        "ServiceMinutes": float(getattr(visit, "service_duration", 0.0)),
                        "Priority": None,
                        "GoalProfile": None,
                        "SpecialtyNames": None,
                    }

                seq += 1
                lateness = float(getattr(visit, "time_warp", 0.0))
                assigned_route_map[str(rec["OrderId"])] = route_name
                stop_rows.append(
                    {
                        "Scenario": scenario_key,
                        "Area": area,
                        "RouteName": route_name,
                        "Sequence": seq,
                        "OrderId": str(rec["OrderId"]),
                        "OrderName": rec.get("OrderName", rec["OrderId"]),
                        "Latitude": float(rec.get("Latitude", lat)),
                        "Longitude": float(rec.get("Longitude", lon)),
                        "ArrivalTime": base_ts + pd.to_timedelta(arrival_min, unit="m"),
                        "ServiceStart": base_ts + pd.to_timedelta(service_start_min, unit="m"),
                        "DepartTime": base_ts + pd.to_timedelta(depart_min, unit="m"),
                        "LateMinutes": lateness,
                        "TravelMinutes": travel_min,
                        "ServiceMinutes": float(rec.get("ServiceMinutes", getattr(visit, "service_duration", 0.0))),
                        "Priority": rec.get("Priority"),
                        "GoalProfile": rec.get("GoalProfile"),
                        "SpecialtyNames": rec.get("SpecialtyNames"),
                    }
                )

                prev_visit = visit
                prev_lat, prev_lon = lat, lon
                service_total += float(rec.get("ServiceMinutes", getattr(visit, "service_duration", 0.0)))

            route_rows.append(
                {
                    "Scenario": scenario_key,
                    "Area": area,
                    "RouteName": route_name,
                    "StartDepotName": start_depot_name,
                    "AssignedOrders": seq,
                    "StartTime": base_ts + pd.to_timedelta(_scalar_int(route_plan.start_time(), 0), unit="m"),
                    "EndTime": base_ts + pd.to_timedelta(_scalar_int(route_plan.end_time(), _scalar_int(route_plan.start_time(), 0)), unit="m"),
                    "TotalDistanceKm": float(_scalar_float(route_plan.distance(), 0.0)) / 1000.0,
                    "TotalTravelMinutes": float(_scalar_float(route_plan.travel_duration(), 0.0)),
                    "TotalServiceMinutes": service_total,
                    "LateMinutes": float(_scalar_float(route_plan.time_warp(), 0.0)),
                    "OvertimeMinutes": float(_scalar_float(route_plan.overtime(), 0.0)),
                    "TechnicianRating": route_src.get("TechnicianRating"),
                    "SpecialtyNames": route_src.get("SpecialtyNames"),
                }
            )

        routes_out = pd.DataFrame(route_rows)
        stops_out = pd.DataFrame(stop_rows)

        orders_out = scenario_orders.copy()
        if "OrderId" not in orders_out.columns and "Name" in orders_out.columns:
            orders_out["OrderId"] = orders_out["Name"].astype(str)
        orders_out["Scenario"] = scenario_key
        orders_out["Area"] = area
        orders_out["AssignedRouteName"] = orders_out["OrderId"].astype(str).map(assigned_route_map).fillna("UNASSIGNED")
        orders_out["Assigned"] = (orders_out["AssignedRouteName"] != "UNASSIGNED").astype(int)
        orders_out["Reason"] = np.where(
            orders_out["Assigned"] == 1,
            "PyVRP metaheuristic assignment",
            "Dropped or unassigned by PyVRP",
        )

        summary = pd.DataFrame([{
            "Scenario": scenario_key,
            "Area": area,
            "AssignedOrders": int(orders_out["Assigned"].sum()),
            "UnassignedOrders": int((orders_out["Assigned"] == 0).sum()),
            "RoutesUsed": int(routes_out["RouteName"].nunique()) if not routes_out.empty else 0,
            "TotalDistanceKm": float(routes_out["TotalDistanceKm"].sum()) if not routes_out.empty else 0.0,
            "LateMinutes": float(routes_out["LateMinutes"].sum()) if not routes_out.empty else 0.0,
            "OvertimeMinutes": float(routes_out["OvertimeMinutes"].sum()) if not routes_out.empty else 0.0,
            "TotalTravelMinutes": float(routes_out["TotalTravelMinutes"].sum()) if not routes_out.empty else 0.0,
            "ScenarioLabel": scenario_label,
            "Backend": "PyVRP",
        }])

        scenario_results.append(
            {"scenario_summary": summary, "routes": routes_out, "stops": stops_out, "orders": orders_out}
        )
        timings.append({"Backend": "PyVRP", "Scenario": scenario_key, "ScenarioLabel": scenario_label, "RuntimeSeconds": time.perf_counter() - started})
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({"Backend": "PyVRP", "Scenario": scenario_key, "Action": action})

    bundle = _build_common_bundle("PyVRP", inputs, subset_orders, subset_routes, scenario_results, timings, actions_rows, config)
    bundle["run_meta"]["pyvrp_runtime_seconds"] = int(config.pyvrp_max_runtime_seconds)
    bundle["run_meta"]["pyvrp_allow_drop_orders"] = False
    return bundle


def _ortools_allowed_vehicle_ids(routes_df: pd.DataFrame, order_row: pd.Series, strict_skill_bias: bool) -> List[int]:
    allowed = list(range(len(routes_df)))

    preferred_depot = order_row.get("PreferredDepot")
    if pd.notna(preferred_depot):
        same_depot = [
            idx
            for idx, (_, route) in enumerate(routes_df.iterrows())
            if str(route.get("StartDepotName", "")).strip() == str(preferred_depot).strip()
        ]
        if same_depot:
            allowed = same_depot

    order_skills = _skill_set(order_row.get("SpecialtyNames") or order_row.get("WorkType"))
    if order_skills:
        skill_matches = [
            idx
            for idx, (_, route) in enumerate(routes_df.iterrows())
            if _skill_set(route.get("SpecialtyNames")).intersection(order_skills)
        ]
        if skill_matches and strict_skill_bias:
            narrowed = [idx for idx in allowed if idx in skill_matches]
            allowed = narrowed or skill_matches

    return sorted(set(int(x) for x in allowed))


def run_ortools_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except Exception as exc:
        raise RuntimeError(
            "OR-Tools backend needs the `ortools` package in the same environment. Install it with `pip install ortools`."
        ) from exc

    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=config.max_orders,
        max_routes=config.max_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )

    subset_orders = subset_orders.loc[
        subset_orders["Latitude"].notna() & subset_orders["Longitude"].notna()
    ].copy().reset_index(drop=True)
    subset_routes = subset_routes.copy().reset_index(drop=True)

    if subset_routes.empty:
        raise RuntimeError("OR-Tools backend has no route rows in the selected scope.")
    if subset_orders.empty:
        raise RuntimeError("OR-Tools backend has no geocoded order rows in the selected scope.")

    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()
    depots_df = inputs["depots"].copy()
    depots_df["Name"] = depots_df["Name"].astype(str)

    scenario_results = []
    timings = []
    actions_rows = []

    total = len(DEFAULT_SCENARIOS)

    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"OR-Tools — {scenario_key}")

        started = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        scenario_orders = scenario_orders.loc[
            scenario_orders["Latitude"].notna() & scenario_orders["Longitude"].notna()
        ].copy().reset_index(drop=True)
        scenario_routes = scenario_routes.copy().reset_index(drop=True)

        if scenario_orders.empty or scenario_routes.empty:
            empty_orders = scenario_orders.copy()
            if "OrderId" not in empty_orders.columns and "Name" in empty_orders.columns:
                empty_orders["OrderId"] = empty_orders["Name"].astype(str)
            empty_orders["Scenario"] = scenario_key
            empty_orders["Area"] = area
            empty_orders["Assigned"] = 0
            empty_orders["AssignedRouteName"] = "UNASSIGNED"
            empty_orders["Reason"] = "No OR-Tools solution scope"
            summary = pd.DataFrame([{
                "Scenario": scenario_key,
                "Area": area,
                "AssignedOrders": 0,
                "UnassignedOrders": len(empty_orders),
                "RoutesUsed": 0,
                "TotalDistanceKm": 0.0,
                "LateMinutes": 0.0,
                "OvertimeMinutes": 0.0,
                "TotalTravelMinutes": 0.0,
                "ScenarioLabel": scenario_label,
                "Backend": "OR-Tools",
            }])
            scenario_results.append({
                "scenario_summary": summary,
                "routes": pd.DataFrame(),
                "stops": pd.DataFrame(),
                "orders": empty_orders,
            })
            timings.append({
                "Backend": "OR-Tools",
                "Scenario": scenario_key,
                "ScenarioLabel": scenario_label,
                "RuntimeSeconds": time.perf_counter() - started,
            })
            for action in explainer.actions_for(scenario_key):
                actions_rows.append({"Backend": "OR-Tools", "Scenario": scenario_key, "Action": action})
            continue

        base_ts = _scenario_base_time(scenario_orders, scenario_routes)
        travel_weight = 1.0 if not override_weights else max(_scalar_float(override_weights.get("travel"), 1.0), 0.1)
        rating_weight = 0.4 if not override_weights else max(_scalar_float(override_weights.get("rating"), 0.4), 0.1)
        sla_weight = 0.6 if not override_weights else max(_scalar_float(override_weights.get("sla"), 0.6), 0.1)
        overtime_weight = 0.5 if not override_weights else max(_scalar_float(override_weights.get("overtime"), 0.5), 0.1)

        depot_names = set(scenario_routes["StartDepotName"].astype(str))
        if "EndDepotName" in scenario_routes.columns:
            depot_names.update(scenario_routes["EndDepotName"].dropna().astype(str).tolist())
        if not depot_names:
            depot_names.update(depots_df["Name"].astype(str).tolist())

        depot_nodes: List[Dict[str, Any]] = []
        depot_node_index: Dict[str, int] = {}

        for depot_name in sorted(depot_names):
            match = depots_df.loc[depots_df["Name"] == str(depot_name)].head(1)
            if not match.empty:
                dep_lat = _scalar_float(match.iloc[0].get("Latitude"), float(scenario_orders["Latitude"].mean()))
                dep_lon = _scalar_float(match.iloc[0].get("Longitude"), float(scenario_orders["Longitude"].mean()))
            else:
                dep_lat = float(scenario_orders["Latitude"].mean())
                dep_lon = float(scenario_orders["Longitude"].mean())
            depot_node_index[str(depot_name)] = len(depot_nodes)
            depot_nodes.append({
                "kind": "depot",
                "name": str(depot_name),
                "lat": dep_lat,
                "lon": dep_lon,
                "service": 0,
                "tw_early": 0,
                "tw_late": 0,
            })

        if not depot_nodes:
            raise RuntimeError("OR-Tools backend could not build depot definitions.")

        first_depot_name = next(iter(depot_node_index.keys()))
        route_rows_src = scenario_routes.to_dict("records")

        client_records: List[Dict[str, Any]] = []
        nodes: List[Dict[str, Any]] = list(depot_nodes)
        order_node_index: Dict[str, int] = {}

        for _, order in scenario_orders.iterrows():
            order_id = str(order.get("OrderId") or order.get("Name") or f"order_{len(client_records) + 1}")
            lat = _scalar_float(order.get("Latitude"), 0.0)
            lon = _scalar_float(order.get("Longitude"), 0.0)
            service_minutes = max(1, _scalar_int(order.get("ServiceTime"), 45))
            tw_early = _minute_offset(order.get("TimeWindowStart1"), base_ts, default=0)
            tw_late_default = tw_early + max(service_minutes + 60, 240)
            tw_late = _minute_offset(order.get("TimeWindowEnd1"), base_ts, default=tw_late_default)
            if tw_late < tw_early:
                tw_late = tw_early + max(service_minutes, 30)

            node_idx = len(nodes)
            order_node_index[order_id] = node_idx
            rec = {
                "kind": "order",
                "OrderId": order_id,
                "OrderName": str(order.get("Name", order_id)),
                "lat": lat,
                "lon": lon,
                "service": service_minutes,
                "tw_early": tw_early,
                "tw_late": tw_late,
                "Priority": order.get("Priority"),
                "GoalProfile": order.get("GoalProfile"),
                "SpecialtyNames": order.get("SpecialtyNames"),
                "SLAFlag": order.get("SLAFlag"),
                "PreferredDepot": order.get("PreferredDepot"),
                "MaxViolationTime1": order.get("MaxViolationTime1"),
                "WorkType": order.get("WorkType"),
            }
            client_records.append(rec)
            nodes.append(rec)

        num_nodes = len(nodes)
        dist_km_matrix = [[0.0] * num_nodes for _ in range(num_nodes)]
        travel_min_matrix = [[0] * num_nodes for _ in range(num_nodes)]

        for from_idx, from_node in enumerate(nodes):
            for to_idx, to_node in enumerate(nodes):
                if from_idx == to_idx:
                    continue
                dist_km = float(haversine_km(from_node["lat"], from_node["lon"], to_node["lat"], to_node["lon"]))
                travel_min = max(1, int(round(dist_km / max(float(config.speed_kmph), 1.0) * 60.0)))
                dist_km_matrix[from_idx][to_idx] = dist_km
                travel_min_matrix[from_idx][to_idx] = travel_min

        starts: List[int] = []
        ends: List[int] = []
        vehicle_capacities: List[int] = []
        vehicle_start_earlies: List[int] = []
        vehicle_start_lates: List[int] = []
        vehicle_shift_limits: List[int] = []
        vehicle_overtime_limits: List[int] = []
        vehicle_names: List[str] = []

        max_time_candidate = 24 * 60
        for route in route_rows_src:
            start_depot_name = str(route.get("StartDepotName") or first_depot_name)
            end_depot_name = str(route.get("EndDepotName") or start_depot_name)
            if start_depot_name not in depot_node_index:
                start_depot_name = first_depot_name
            if end_depot_name not in depot_node_index:
                end_depot_name = start_depot_name

            starts.append(int(depot_node_index[start_depot_name]))
            ends.append(int(depot_node_index[end_depot_name]))
            vehicle_capacities.append(max(1, _scalar_int(route.get("MaxOrderCount"), 40)))

            shift_duration = max(60, _scalar_int(route.get("MaxTotalTime"), 720))
            overtime_start = max(0, _scalar_int(route.get("OvertimeStartTime"), 540))
            start_early = _minute_offset(route.get("EarliestStartTime"), base_ts, default=0)
            start_late_default = start_early + max(60, min(shift_duration, 240))
            start_late = _minute_offset(route.get("LatestStartTime"), base_ts, default=start_late_default)
            if start_late < start_early:
                start_late = start_early

            vehicle_start_earlies.append(start_early)
            vehicle_start_lates.append(start_late)
            vehicle_shift_limits.append(shift_duration)
            vehicle_overtime_limits.append(overtime_start)
            vehicle_names.append(str(route.get("Name", f"Route_{len(vehicle_names) + 1}")))

            max_time_candidate = max(max_time_candidate, start_late + shift_duration + 720)

        for rec in client_records:
            max_time_candidate = max(max_time_candidate, int(rec["tw_late"]) + 720)

        manager = pywrapcp.RoutingIndexManager(num_nodes, len(route_rows_src), starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        def _travel_cost_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return max(0, int(round(travel_min_matrix[from_node][to_node] * travel_weight)))

        def _time_callback(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            service_time = int(nodes[from_node].get("service", 0))
            return int(service_time + travel_min_matrix[from_node][to_node])

        def _demand_callback(from_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            return 1 if nodes[from_node].get("kind") == "order" else 0

        transit_callback_index = routing.RegisterTransitCallback(_travel_cost_callback)
        time_callback_index = routing.RegisterTransitCallback(_time_callback)
        demand_callback_index = routing.RegisterUnaryTransitCallback(_demand_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            vehicle_capacities,
            True,
            "Load",
        )

        waiting_slack = max(120, min(1440, max_time_candidate))
        routing.AddDimension(
            time_callback_index,
            waiting_slack,
            int(max_time_candidate),
            False,
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        for vehicle_id, route in enumerate(route_rows_src):
            start_index = routing.Start(vehicle_id)
            end_index = routing.End(vehicle_id)
            start_early = int(vehicle_start_earlies[vehicle_id])
            start_late = int(vehicle_start_lates[vehicle_id])
            shift_duration = int(vehicle_shift_limits[vehicle_id])
            overtime_start = int(vehicle_overtime_limits[vehicle_id])

            time_dimension.CumulVar(start_index).SetRange(start_early, start_late)
            time_dimension.CumulVar(end_index).SetRange(start_early, start_early + shift_duration)
            overtime_coeff = max(1, int(round(8 * overtime_weight)))
            time_dimension.SetCumulVarSoftUpperBound(end_index, start_early + overtime_start, overtime_coeff)

            rating = _scalar_float(route.get("TechnicianRating"), 4.0)
            fixed_cost = max(0, int(round((5.5 - rating) * 75.0 * rating_weight)))
            routing.SetFixedCostOfVehicle(fixed_cost, vehicle_id)
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(start_index))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(end_index))

        for _, order in scenario_orders.iterrows():
            order_id = str(order.get("OrderId") or order.get("Name"))
            node_idx = order_node_index.get(order_id)
            if node_idx is None:
                continue
            index = manager.NodeToIndex(node_idx)
            if index < 0:
                continue

            tw_early = int(nodes[node_idx]["tw_early"])
            tw_late = int(nodes[node_idx]["tw_late"])
            soft_violation = max(180, _scalar_int(order.get("MaxViolationTime1"), 0))
            time_dimension.CumulVar(index).SetRange(tw_early, tw_late + soft_violation)

            priority = _scalar_int(order.get("Priority"), 3)
            sla_flag = _scalar_int(order.get("SLAFlag"), 0)
            lateness_cost = max(1, int(round((6 + 4 * priority + 10 * sla_flag) * sla_weight)))
            time_dimension.SetCumulVarSoftUpperBound(index, tw_late, lateness_cost)

            allowed_vehicle_ids = [int(v) for v in _ortools_allowed_vehicle_ids(scenario_routes, order, config.strict_skill_bias)]
            if allowed_vehicle_ids and len(allowed_vehicle_ids) < len(route_rows_src):
                routing.SetAllowedVehiclesForIndex(allowed_vehicle_ids, index)

            if bool(config.ortools_allow_drop_orders):
                routing.AddDisjunction([index], int(_priority_prize(order, override_weights)))

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = int(config.ortools_max_runtime_seconds)
        search_parameters.log_search = False

        solution = routing.SolveWithParameters(search_parameters)

        if solution is None:
            orders_out = scenario_orders.copy()
            if "OrderId" not in orders_out.columns and "Name" in orders_out.columns:
                orders_out["OrderId"] = orders_out["Name"].astype(str)
            orders_out["Scenario"] = scenario_key
            orders_out["Area"] = area
            orders_out["AssignedRouteName"] = "UNASSIGNED"
            orders_out["Assigned"] = 0
            orders_out["Reason"] = "No feasible OR-Tools solution"
            summary = pd.DataFrame([{
                "Scenario": scenario_key,
                "Area": area,
                "AssignedOrders": 0,
                "UnassignedOrders": int(len(orders_out)),
                "RoutesUsed": 0,
                "TotalDistanceKm": 0.0,
                "LateMinutes": 0.0,
                "OvertimeMinutes": 0.0,
                "TotalTravelMinutes": 0.0,
                "ScenarioLabel": scenario_label,
                "Backend": "OR-Tools",
            }])
            scenario_results.append({
                "scenario_summary": summary,
                "routes": pd.DataFrame(),
                "stops": pd.DataFrame(),
                "orders": orders_out,
            })
            timings.append({
                "Backend": "OR-Tools",
                "Scenario": scenario_key,
                "ScenarioLabel": scenario_label,
                "RuntimeSeconds": time.perf_counter() - started,
            })
            for action in explainer.actions_for(scenario_key):
                actions_rows.append({"Backend": "OR-Tools", "Scenario": scenario_key, "Action": action})
            continue

        assigned_route_map: Dict[str, str] = {}
        route_rows: List[Dict[str, Any]] = []
        stop_rows: List[Dict[str, Any]] = []

        for vehicle_id, route in enumerate(route_rows_src):
            if not routing.IsVehicleUsed(solution, vehicle_id):
                continue

            route_name = vehicle_names[vehicle_id]
            start_index = routing.Start(vehicle_id)
            end_index = routing.End(vehicle_id)
            start_node = manager.IndexToNode(start_index)
            start_depot_name = str(nodes[start_node].get("name", first_depot_name))

            prev_index = start_index
            prev_node = manager.IndexToNode(prev_index)
            prev_visit_index = None
            seq = 0
            total_distance_km = 0.0
            total_travel_minutes = 0.0
            total_service_minutes = 0.0
            total_late_minutes = 0.0

            index = solution.Value(routing.NextVar(start_index))
            while True:
                current_node = manager.IndexToNode(index)
                leg_distance_km = float(dist_km_matrix[prev_node][current_node])
                leg_travel_minutes = float(travel_min_matrix[prev_node][current_node])
                total_distance_km += leg_distance_km
                total_travel_minutes += leg_travel_minutes

                current_meta = nodes[current_node]
                if current_meta.get("kind") == "order":
                    service_start_min = float(solution.Min(time_dimension.CumulVar(index)))
                    service_minutes = float(current_meta.get("service", 0.0))
                    depart_min = service_start_min + service_minutes
                    tw_late = float(current_meta.get("tw_late", service_start_min))
                    lateness = max(0.0, service_start_min - tw_late)

                    seq += 1
                    order_id = str(current_meta["OrderId"])
                    assigned_route_map[order_id] = route_name
                    stop_rows.append(
                        {
                            "Scenario": scenario_key,
                            "Area": area,
                            "RouteName": route_name,
                            "Sequence": seq,
                            "OrderId": order_id,
                            "OrderName": current_meta.get("OrderName", order_id),
                            "Latitude": float(current_meta.get("lat", np.nan)),
                            "Longitude": float(current_meta.get("lon", np.nan)),
                            "ArrivalTime": base_ts + pd.to_timedelta(service_start_min, unit="m"),
                            "ServiceStart": base_ts + pd.to_timedelta(service_start_min, unit="m"),
                            "DepartTime": base_ts + pd.to_timedelta(depart_min, unit="m"),
                            "LateMinutes": lateness,
                            "TravelMinutes": leg_travel_minutes,
                            "ServiceMinutes": service_minutes,
                            "Priority": current_meta.get("Priority"),
                            "GoalProfile": current_meta.get("GoalProfile"),
                            "SpecialtyNames": current_meta.get("SpecialtyNames"),
                        }
                    )
                    total_service_minutes += service_minutes
                    total_late_minutes += lateness

                if routing.IsEnd(index):
                    break

                prev_index = index
                prev_node = current_node
                index = solution.Value(routing.NextVar(index))

            if seq == 0:
                continue

            start_time = base_ts + pd.to_timedelta(float(solution.Min(time_dimension.CumulVar(start_index))), unit="m")
            end_time = base_ts + pd.to_timedelta(float(solution.Min(time_dimension.CumulVar(end_index))), unit="m")
            shift_minutes = float(solution.Min(time_dimension.CumulVar(end_index)) - solution.Min(time_dimension.CumulVar(start_index)))
            overtime = max(0.0, shift_minutes - float(vehicle_overtime_limits[vehicle_id]))

            route_rows.append(
                {
                    "Scenario": scenario_key,
                    "Area": area,
                    "RouteName": route_name,
                    "StartDepotName": start_depot_name,
                    "AssignedOrders": seq,
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "TotalDistanceKm": total_distance_km,
                    "TotalTravelMinutes": total_travel_minutes,
                    "TotalServiceMinutes": total_service_minutes,
                    "LateMinutes": total_late_minutes,
                    "OvertimeMinutes": overtime,
                    "TechnicianRating": route.get("TechnicianRating"),
                    "SpecialtyNames": route.get("SpecialtyNames"),
                }
            )

        routes_out = pd.DataFrame(route_rows)
        stops_out = pd.DataFrame(stop_rows)

        orders_out = scenario_orders.copy()
        if "OrderId" not in orders_out.columns and "Name" in orders_out.columns:
            orders_out["OrderId"] = orders_out["Name"].astype(str)
        orders_out["Scenario"] = scenario_key
        orders_out["Area"] = area
        orders_out["AssignedRouteName"] = orders_out["OrderId"].astype(str).map(assigned_route_map).fillna("UNASSIGNED")
        orders_out["Assigned"] = (orders_out["AssignedRouteName"] != "UNASSIGNED").astype(int)
        orders_out["Reason"] = np.where(
            orders_out["Assigned"] == 1,
            "OR-Tools routing assignment",
            "Dropped or unassigned by OR-Tools",
        )

        summary = pd.DataFrame([{
            "Scenario": scenario_key,
            "Area": area,
            "AssignedOrders": int(orders_out["Assigned"].sum()),
            "UnassignedOrders": int((orders_out["Assigned"] == 0).sum()),
            "RoutesUsed": int(routes_out["RouteName"].nunique()) if not routes_out.empty else 0,
            "TotalDistanceKm": float(routes_out["TotalDistanceKm"].sum()) if not routes_out.empty else 0.0,
            "LateMinutes": float(routes_out["LateMinutes"].sum()) if not routes_out.empty else 0.0,
            "OvertimeMinutes": float(routes_out["OvertimeMinutes"].sum()) if not routes_out.empty else 0.0,
            "TotalTravelMinutes": float(routes_out["TotalTravelMinutes"].sum()) if not routes_out.empty else 0.0,
            "ScenarioLabel": scenario_label,
            "Backend": "OR-Tools",
        }])

        scenario_results.append({
            "scenario_summary": summary,
            "routes": routes_out,
            "stops": stops_out,
            "orders": orders_out,
        })
        timings.append({
            "Backend": "OR-Tools",
            "Scenario": scenario_key,
            "ScenarioLabel": scenario_label,
            "RuntimeSeconds": time.perf_counter() - started,
        })
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({"Backend": "OR-Tools", "Scenario": scenario_key, "Action": action})

    bundle = _build_common_bundle("OR-Tools", inputs, subset_orders, subset_routes, scenario_results, timings, actions_rows, config)
    bundle["run_meta"]["ortools_runtime_seconds"] = int(config.ortools_max_runtime_seconds)
    bundle["run_meta"]["ortools_allow_drop_orders"] = bool(config.ortools_allow_drop_orders)
    return bundle


def _build_common_bundle(
    backend_name: str,
    inputs: Dict[str, pd.DataFrame],
    subset_orders: pd.DataFrame,
    subset_routes: pd.DataFrame,
    scenario_results: List[Dict[str, Any]],
    timings: List[Dict[str, Any]],
    actions_rows: List[Dict[str, Any]],
    config: BackendConfig,
) -> Dict[str, Any]:
    summary_df = pd.concat([x["scenario_summary"] for x in scenario_results], ignore_index=True)
    routes_df = pd.concat([x["routes"] for x in scenario_results], ignore_index=True) if scenario_results else pd.DataFrame()
    stops_df = pd.concat([x["stops"] for x in scenario_results], ignore_index=True) if scenario_results else pd.DataFrame()
    orders_df = pd.concat([x["orders"] for x in scenario_results], ignore_index=True) if scenario_results else pd.DataFrame()
    actions_df = pd.DataFrame(actions_rows)
    timings_df = pd.DataFrame(timings)
    coverage_df = map_catalog_to_requirements().copy()
    coverage_df["Backend"] = backend_name
    run_meta = {
        "backend": backend_name,
        "subset_orders": len(subset_orders),
        "subset_routes": len(subset_routes),
        "uploaded_orders": len(inputs["orders"]),
        "uploaded_routes": len(inputs["routes"]),
        "uploaded_depots": len(inputs["depots"]),
        "config": config.__dict__,
    }
    return {
        "backend": backend_name,
        "scenario_summary": summary_df,
        "scenario_actions": actions_df,
        "route_output_all": routes_df,
        "stop_output_all": stops_df,
        "order_output_all": orders_df,
        "scenario_timings": timings_df,
        "requirement_checklist": coverage_df,
        "run_meta": run_meta,
    }


def run_current_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    from base_engine import run_all_scenarios
    bundle = run_all_scenarios(
        inputs=inputs,
        max_orders=config.max_orders,
        max_routes=config.max_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        speed_kmph=config.speed_kmph,
        fast_mode=config.fast_mode,
        sample_by_priority=config.sample_by_priority,
        use_agents=True,
        use_two_opt=config.use_two_opt,
        strict_skill_bias=config.strict_skill_bias,
        progress_cb=progress_cb,
    )
    bundle["backend"] = "Impacted-Subset Greedy + 2-opt"
    bundle["requirement_checklist"]["Backend"] = "Impacted-Subset Greedy + 2-opt"
    bundle["run_meta"]["backend"] = "Impacted-Subset Greedy + 2-opt"
    return bundle


def run_osrm_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    effective_orders, effective_routes = _effective_osrm_scope(config)
    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=effective_orders,
        max_routes=effective_routes,
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )
    policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
    optimizer = OSRMAssignmentAgent(
        inputs["depots"],
        osrm_client=OSRMClient(config.osrm_base_url, config.osrm_profile, timeout=int(config.osrm_timeout)),
        speed_kmph=config.speed_kmph,
        use_two_opt=config.use_two_opt,
    )
    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()

    scenario_results = []
    timings = []
    actions_rows = []

    total = len(DEFAULT_SCENARIOS)
    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"OSRM — {scenario_key}")
        started = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        result = optimizer.assign_and_sequence(
            orders_df=scenario_orders,
            routes_df=scenario_routes,
            policy_agent=policy,
            scenario_name=scenario_key,
            area=area,
            override_weights=override_weights,
        )
        result["scenario_summary"]["ScenarioLabel"] = scenario_label
        result["scenario_summary"]["Backend"] = "OSRM"
        scenario_results.append(result)
        runtime = time.perf_counter() - started
        timings.append({"Backend": "OSRM", "Scenario": scenario_key, "ScenarioLabel": scenario_label, "RuntimeSeconds": runtime})
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({"Backend": "OSRM", "Scenario": scenario_key, "Action": action})
    bundle = _build_common_bundle("OSRM", inputs, subset_orders, subset_routes, scenario_results, timings, actions_rows, config)
    bundle["run_meta"]["osrm_effective_orders"] = len(subset_orders)
    bundle["run_meta"]["osrm_effective_routes"] = len(subset_routes)
    bundle["run_meta"]["osrm_public_demo"] = _is_public_osrm(config.osrm_base_url)
    return bundle


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for k, v in defaults.items():
        if k not in out.columns:
            out[k] = v
    return out


def _prepare_orders_routes_for_esri(orders_df: pd.DataFrame, routes_df: pd.DataFrame, depots_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = _ensure_columns(orders_df, {
        "Address": orders_df.get("Name", pd.Series([""] * len(orders_df))),
        "Priority": 2,
        "ServiceTime": 15.0,
        "AssignmentRule": np.nan,
        "RouteName": "",
        "Sequence": np.nan,
        "SpecialtyNames": "",
        "TimeWindowStart1": pd.NaT,
        "TimeWindowEnd1": pd.NaT,
        "MaxViolationTime1": np.nan,
        "Description": "",
    }).copy()
    orders["TimeWindowStart1"] = pd.to_datetime(orders["TimeWindowStart1"], errors="coerce")
    orders["TimeWindowEnd1"] = pd.to_datetime(orders["TimeWindowEnd1"], errors="coerce")
    orders["MaxViolationTime1"] = pd.to_numeric(orders["MaxViolationTime1"], errors="coerce")
    orders["Priority"] = pd.to_numeric(orders["Priority"], errors="coerce").fillna(2).astype(int)
    orders["ServiceTime"] = pd.to_numeric(orders["ServiceTime"], errors="coerce").fillna(15)
    orders = orders.loc[orders["Latitude"].notna() & orders["Longitude"].notna()].copy()

    routes = _ensure_columns(routes_df, {
        "CostPerUnitTime": 1.0,
        "CostPerUnitDistance": 0.35,
        "CostPerUnitOvertime": 4.5,
        "OvertimeStartTime": 540,
        "MaxTotalTime": 720,
        "MaxOrderCount": 40,
        "StartDepotName": routes_df.get("StartDepotName", ""),
        "EndDepotName": routes_df.get("StartDepotName", ""),
        "SpecialtyNames": "",
        "TechnicianRating": 4.0,
    }).copy()
    routes["EarliestStartTime"] = pd.to_datetime(routes["EarliestStartTime"], errors="coerce")
    routes["LatestStartTime"] = pd.to_datetime(routes["LatestStartTime"], errors="coerce")
    for col in ["CostPerUnitTime", "CostPerUnitDistance", "CostPerUnitOvertime", "OvertimeStartTime", "MaxTotalTime", "MaxOrderCount", "TechnicianRating"]:
        routes[col] = pd.to_numeric(routes[col], errors="coerce")

    depots = depots_df.copy()
    depots = depots.loc[depots["Latitude"].notna() & depots["Longitude"].notna()].copy()
    return orders.reset_index(drop=True), routes.reset_index(drop=True), depots.reset_index(drop=True)


def _featureset_to_df(fs) -> pd.DataFrame:
    try:
        if hasattr(fs, "sdf"):
            return fs.sdf.copy()
    except Exception:
        pass
    try:
        if hasattr(fs, "to_dict"):
            payload = fs.to_dict()
            feats = payload.get("features", [])
            rows = []
            for feat in feats:
                attrs = feat.get("attributes", {}).copy()
                geom = feat.get("geometry")
                if geom:
                    attrs["SHAPE"] = geom
                rows.append(attrs)
            return pd.DataFrame(rows)
    except Exception:
        pass
    try:
        feats = getattr(fs, "features", [])
        rows = []
        for feat in feats:
            attrs = getattr(feat, "attributes", {}) or {}
            geom = getattr(feat, "geometry", None)
            row = dict(attrs)
            if geom:
                row["SHAPE"] = geom
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def _extract_esri_output(result: Any, key: str):
    if result is None:
        return None
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        if key in result:
            return result[key]
        short_key = key.replace("out_", "")
        return result.get(short_key)
    if isinstance(result, tuple):
        for item in result:
            val = _extract_esri_output(item, key)
            if val is not None:
                return val
    return None


def _normalize_esri_stop_route_names(stops_df: pd.DataFrame, routes_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stops = stops_df.copy()
    routes = routes_df.copy()

    if "Name" in routes.columns and "RouteName" not in routes.columns:
        routes["RouteName"] = routes["Name"].astype(str)

    if "RouteName" not in stops.columns:
        for cand in ["RouteName", "Route", "RouteID", "RouteName_1"]:
            if cand in stops.columns:
                stops["RouteName"] = stops[cand].astype(str)
                break

    if "RouteName" not in stops.columns and "RouteName" in routes.columns and len(routes) == 1:
        stops["RouteName"] = str(routes.iloc[0]["RouteName"])

    if "StopType" in stops.columns:
        st = pd.to_numeric(stops["StopType"], errors="coerce")
        stops = stops.loc[st.fillna(0).isin([0, 1])].copy()

    return stops.reset_index(drop=True), routes.reset_index(drop=True)


def _series_from(df: pd.DataFrame, value: Any, default: Any = np.nan) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reset_index(drop=True)
    if isinstance(value, np.ndarray):
        return pd.Series(value)
    if isinstance(value, list):
        return pd.Series(value)
    if value is None:
        return pd.Series([default] * len(df), index=df.index)
    return pd.Series([value] * len(df), index=df.index)


def _num_series(df: pd.DataFrame, value: Any, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(_series_from(df, value, default), errors="coerce").fillna(default)


def _dt_series(df: pd.DataFrame, value: Any) -> pd.Series:
    raw = _series_from(df, value, pd.NaT)
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_mask = numeric.notna()

    if numeric_mask.any():
        abs_numeric = numeric.abs()

        epoch_ms_mask = numeric_mask & (abs_numeric >= 1e11)
        if epoch_ms_mask.any():
            out.loc[epoch_ms_mask] = pd.to_datetime(numeric.loc[epoch_ms_mask], unit="ms", errors="coerce")

        epoch_s_mask = numeric_mask & (abs_numeric >= 1e9) & (abs_numeric < 1e11)
        if epoch_s_mask.any():
            out.loc[epoch_s_mask] = pd.to_datetime(numeric.loc[epoch_s_mask], unit="s", errors="coerce")

        excel_day_mask = numeric_mask & (abs_numeric >= 2e4) & (abs_numeric < 1e6)
        if excel_day_mask.any():
            out.loc[excel_day_mask] = pd.to_datetime(
                numeric.loc[excel_day_mask],
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )

    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        out.loc[non_numeric_mask] = pd.to_datetime(raw.loc[non_numeric_mask], errors="coerce")

    return out


def run_esri_backend(inputs: Dict[str, pd.DataFrame], config: BackendConfig, progress_cb: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    try:
        import arcgis
        from arcgis.gis import GIS
        from arcgis.network.analysis import solve_vehicle_routing_problem
    except Exception as exc:
        raise RuntimeError("ArcGIS backend needs the `arcgis` Python package in the same environment.") from exc

    if not config.esri_portal_url:
        raise RuntimeError("ArcGIS backend needs portal URL, username, and password from the sidebar or environment variables.")

    gis = GIS(config.esri_portal_url, config.esri_username, config.esri_password, set_active=True)

    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=min(config.max_orders, 2000),
        max_routes=min(config.max_routes, 100),
        depot_focus=config.depot_focus,
        random_seed=config.random_seed,
        sample_by_priority=config.sample_by_priority,
    )

    policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=config.strict_skill_bias)
    scenario_agent = ScenarioAgent(random_seed=config.random_seed)
    explainer = ExplainAgent()

    scenario_results = []
    timings = []
    actions_rows = []

    total = len(DEFAULT_SCENARIOS)
    for idx, (scenario_key, scenario_label, area) in enumerate(DEFAULT_SCENARIOS, start=1):
        if progress_cb:
            progress_cb(idx, total, f"Esri — {scenario_key}")
        started = time.perf_counter()

        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        if override_weights:
            travel_w = float(override_weights.get("travel", 0.0))
            overtime_w = float(override_weights.get("overtime", 0.0))
            scenario_routes = scenario_routes.copy()
            if "CostPerUnitDistance" in scenario_routes.columns:
                scenario_routes["CostPerUnitDistance"] = pd.to_numeric(scenario_routes["CostPerUnitDistance"], errors="coerce").fillna(0.35) * (1 + 0.4 * travel_w)
            if "CostPerUnitOvertime" in scenario_routes.columns:
                scenario_routes["CostPerUnitOvertime"] = pd.to_numeric(scenario_routes["CostPerUnitOvertime"], errors="coerce").fillna(4.5) * (1 + 0.6 * overtime_w)

        orders_p, routes_p, depots_p = _prepare_orders_routes_for_esri(scenario_orders, scenario_routes, inputs["depots"])

        depot_names = set(depots_p["Name"].astype(str).str.strip())
        bad_start = routes_p.loc[~routes_p["StartDepotName"].astype(str).str.strip().isin(depot_names)]
        bad_end = routes_p.loc[~routes_p["EndDepotName"].astype(str).str.strip().isin(depot_names)]
        if not bad_start.empty or not bad_end.empty:
            raise RuntimeError(
                f"Esri depot mapping error for {scenario_key}. "
                f"Bad StartDepotName rows: {len(bad_start)}, Bad EndDepotName rows: {len(bad_end)}"
            )

        orders_sdf = pd.DataFrame.spatial.from_xy(orders_p.copy(), "Longitude", "Latitude")
        orders_sdf = orders_sdf.drop(columns=[c for c in ["Latitude", "Longitude"] if c in orders_sdf.columns], errors="ignore")
        orders_fset = orders_sdf.spatial.to_featureset()

        depots_sdf = pd.DataFrame.spatial.from_xy(depots_p.copy(), "Longitude", "Latitude")
        depots_sdf = depots_sdf.drop(columns=[c for c in ["Latitude", "Longitude"] if c in depots_sdf.columns], errors="ignore")
        depots_fset = depots_sdf.spatial.to_featureset()

        routes_fset = arcgis.features.FeatureSet.from_dataframe(routes_p.copy())

        result = solve_vehicle_routing_problem(
            orders=orders_fset,
            depots=depots_fset,
            routes=routes_fset,
            save_route_data=True,
            populate_directions=True,
            populate_stop_shapes=True,
            travel_mode=config.esri_travel_mode,
            default_date=pd.Timestamp("2026-04-06 07:00:00").to_pydatetime(),
            gis=gis,
        )

        routes_out = _featureset_to_df(_extract_esri_output(result, "out_routes"))
        stops_out = _featureset_to_df(_extract_esri_output(result, "out_stops"))
        _ = _featureset_to_df(_extract_esri_output(result, "out_unassigned_stops"))

        stops_out, routes_out = _normalize_esri_stop_route_names(stops_out, routes_out)
        if routes_out.empty and stops_out.empty:
            debug_msg = {
                "result_type": str(type(result)),
                "has_out_routes": hasattr(result, "out_routes"),
                "has_out_stops": hasattr(result, "out_stops"),
                "has_out_unassigned_stops": hasattr(result, "out_unassigned_stops"),
                "dict_keys": list(result.keys()) if isinstance(result, dict) else [],
                "routes_cols": list(routes_out.columns),
                "stops_cols": list(stops_out.columns),
                "routes_rows": len(routes_out),
                "stops_rows": len(stops_out),
            }
            raise RuntimeError(f"Esri empty output for {scenario_key}. Debug: {debug_msg}")

        if "Latitude" not in stops_out.columns or "Longitude" not in stops_out.columns:
            if "SHAPE" in stops_out.columns:
                def _xy(shape):
                    if isinstance(shape, dict):
                        return pd.Series({"Longitude": shape.get("x"), "Latitude": shape.get("y")})
                    return pd.Series({"Longitude": np.nan, "Latitude": np.nan})
                xy_df = stops_out["SHAPE"].apply(_xy)
                stops_out[["Longitude", "Latitude"]] = xy_df

        if "Sequence" not in stops_out.columns:
            for cand in ["VisitOrder", "VisitSequence", "SequenceID"]:
                if cand in stops_out.columns:
                    stops_out["Sequence"] = stops_out[cand]
                    break
        if "Sequence" not in stops_out.columns:
            stops_out["Sequence"] = range(1, len(stops_out) + 1)

        if "RouteName" not in routes_out.columns and "Name" in routes_out.columns:
            routes_out["RouteName"] = routes_out["Name"].astype(str)
        if "RouteName" not in stops_out.columns and "Name" in stops_out.columns:
            stops_out["RouteName"] = stops_out["Name"].astype(str)
        if "OrderId" not in stops_out.columns:
            for cand in ["OrderName", "Name"]:
                if cand in stops_out.columns:
                    stops_out["OrderId"] = stops_out[cand].astype(str)
                    break
        if "OrderId" not in stops_out.columns:
            stops_out["OrderId"] = pd.Series([f"STOP_{i+1}" for i in range(len(stops_out))])

        routes_norm = pd.DataFrame({
            "Scenario": [scenario_key] * len(routes_out),
            "Area": [area] * len(routes_out),
            "RouteName": _series_from(routes_out, routes_out.get("RouteName", routes_out.get("Name", None)), "").astype(str),
            "StartDepotName": _series_from(routes_out, routes_out.get("StartDepotName", ""), "").astype(str),
            "AssignedOrders": _num_series(routes_out, routes_out.get("OrderCount", routes_out.get("OrdersCount", None)), 0).astype(int),
            "StartTime": _dt_series(routes_out, routes_out.get("StartTime", None)),
            "EndTime": _dt_series(routes_out, routes_out.get("EndTime", None)),
            "TotalDistanceKm": _num_series(routes_out, routes_out.get("TotalDistance", routes_out.get("TotalDistanceKm", None)), 0.0),
            "TotalTravelMinutes": _num_series(routes_out, routes_out.get("TotalTravelTime", routes_out.get("TotalTravelMinutes", None)), 0.0),
            "TotalServiceMinutes": _num_series(routes_out, routes_out.get("TotalServiceTime", routes_out.get("TotalServiceMinutes", None)), 0.0),
            "LateMinutes": _num_series(routes_out, routes_out.get("TotalViolationTime", routes_out.get("LateMinutes", None)), 0.0),
            "OvertimeMinutes": _num_series(routes_out, routes_out.get("TotalOvertime", routes_out.get("OvertimeMinutes", None)), 0.0),
            "TechnicianRating": pd.Series([np.nan] * len(routes_out)),
            "SpecialtyNames": pd.Series([""] * len(routes_out)),
        })

        stops_norm = pd.DataFrame({
            "Scenario": [scenario_key] * len(stops_out),
            "Area": [area] * len(stops_out),
            "RouteName": _series_from(stops_out, stops_out.get("RouteName", ""), "").astype(str),
            "Sequence": _num_series(stops_out, stops_out.get("Sequence", None), 0).astype(int),
            "OrderId": _series_from(stops_out, stops_out.get("OrderId", stops_out.get("Name", None)), "").astype(str),
            "OrderName": _series_from(stops_out, stops_out.get("Name", ""), "").astype(str),
            "Latitude": _num_series(stops_out, stops_out.get("Latitude", None), np.nan),
            "Longitude": _num_series(stops_out, stops_out.get("Longitude", None), np.nan),
            "ArrivalTime": _dt_series(stops_out, stops_out.get("ArriveTime", stops_out.get("ArrivalTime", None))),
            "ServiceStart": _dt_series(stops_out, stops_out.get("ServiceStart", stops_out.get("DepartTime", None))),
            "DepartTime": _dt_series(stops_out, stops_out.get("DepartTime", None)),
            "LateMinutes": _num_series(stops_out, stops_out.get("ViolationTime", None), 0.0),
            "TravelMinutes": _num_series(stops_out, stops_out.get("WaitTime", None), 0.0),
            "ServiceMinutes": _num_series(stops_out, stops_out.get("ServiceTime", None), 0.0),
            "Priority": pd.Series([np.nan] * len(stops_out)),
            "GoalProfile": pd.Series([""] * len(stops_out)),
            "SpecialtyNames": pd.Series([""] * len(stops_out)),
        })

        orders_out = scenario_orders.copy()
        orders_out["Scenario"] = scenario_key
        orders_out["Area"] = area
        if "Name" in orders_out.columns and "OrderId" not in orders_out.columns:
            orders_out["OrderId"] = orders_out["Name"].astype(str)
        assigned_ids = set(stops_norm["OrderId"].astype(str))
        orders_out["Assigned"] = orders_out["OrderId"].astype(str).isin(assigned_ids).astype(int)
        route_map = stops_norm.drop_duplicates("OrderId").set_index("OrderId")["RouteName"].to_dict() if not stops_norm.empty else {}
        orders_out["AssignedRouteName"] = orders_out["OrderId"].astype(str).map(route_map).fillna("UNASSIGNED")

        summary = pd.DataFrame([{
            "Scenario": scenario_key,
            "Area": area,
            "AssignedOrders": int(orders_out["Assigned"].sum()),
            "UnassignedOrders": int((orders_out["Assigned"] == 0).sum()),
            "RoutesUsed": int(routes_norm["RouteName"].nunique()) if not routes_norm.empty else 0,
            "TotalDistanceKm": float(routes_norm["TotalDistanceKm"].sum()) if not routes_norm.empty else 0.0,
            "LateMinutes": float(routes_norm["LateMinutes"].sum()) if not routes_norm.empty else 0.0,
            "OvertimeMinutes": float(routes_norm["OvertimeMinutes"].sum()) if not routes_norm.empty else 0.0,
            "TotalTravelMinutes": float(routes_norm["TotalTravelMinutes"].sum()) if not routes_norm.empty else 0.0,
            "ScenarioLabel": scenario_label,
            "Backend": "Esri",
        }])

        scenario_results.append({"scenario_summary": summary, "routes": routes_norm, "stops": stops_norm, "orders": orders_out})
        timings.append({"Backend": "Esri", "Scenario": scenario_key, "ScenarioLabel": scenario_label, "RuntimeSeconds": time.perf_counter() - started})
        for action in explainer.actions_for(scenario_key):
            actions_rows.append({"Backend": "Esri", "Scenario": scenario_key, "Action": action})

    return _build_common_bundle("Esri", inputs, subset_orders, subset_routes, scenario_results, timings, actions_rows, config)




def _default_scenario_meta() -> Dict[str, Dict[str, str]]:
    return {
        key: {"ScenarioLabel": label, "Area": area}
        for key, label, area in DEFAULT_SCENARIOS
    }


def _normalize_uploaded_scenario_key(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    s = raw.upper().replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s in {"BASE", "BASE SCENARIO"}:
        return "BASE"
    s = re.sub(r"^SCENARIO\s+", "", s)
    match = re.match(r"^(EAS|RT|GB)\s*(\d+)$", s)
    if match:
        return f"{match.group(1)}_{int(match.group(2))}"
    return raw.strip().replace(" ", "_")


def _ffill_scenario_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Scenario", "ScenarioTitle", "Category"]:
        if col in out.columns:
            out[col] = out[col].ffill()
    return out


# -------------------------------
# Saved artifact helpers
# -------------------------------


def _sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "item"


def _backend_artifact_dir(root_dir: str | Path, backend_name: str) -> Path:
    return Path(root_dir) / _sanitize_name(backend_name)


def build_backend_workbook(bundle: Dict[str, Any]) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        bundle.get("scenario_summary", pd.DataFrame()).to_excel(writer, sheet_name="scenario_summary", index=False)
        bundle.get("scenario_actions", pd.DataFrame()).to_excel(writer, sheet_name="scenario_actions", index=False)
        bundle.get("route_output_all", pd.DataFrame()).to_excel(writer, sheet_name="route_output_all", index=False)
        bundle.get("stop_output_all", pd.DataFrame()).to_excel(writer, sheet_name="stop_output_all", index=False)
        bundle.get("order_output_all", pd.DataFrame()).to_excel(writer, sheet_name="order_output_all", index=False)
        bundle.get("scenario_timings", pd.DataFrame()).to_excel(writer, sheet_name="scenario_timings", index=False)
        bundle.get("requirement_checklist", pd.DataFrame()).to_excel(writer, sheet_name="requirement_checklist", index=False)
        pd.DataFrame([bundle.get("run_meta", {})]).to_excel(writer, sheet_name="run_meta", index=False)
    out.seek(0)
    return out.getvalue()


def load_saved_backend_artifacts(root_dir: str | Path, backend_name: str) -> Dict[str, Any]:
    artifact_dir = _backend_artifact_dir(root_dir, backend_name)
    workbook_path = artifact_dir / "vrp_scenario_results_full.xlsx"
    if not workbook_path.exists():
        raise FileNotFoundError(f"Saved workbook was not found for {backend_name}: {workbook_path}")

    xls = pd.ExcelFile(workbook_path)
    required = {"scenario_summary", "route_output_all", "stop_output_all", "scenario_timings"}
    missing = sorted(required.difference(set(xls.sheet_names)))
    if missing:
        raise RuntimeError(f"Saved workbook for {backend_name} is missing required sheets: {', '.join(missing)}")

    def _read(name: str) -> pd.DataFrame:
        return pd.read_excel(xls, sheet_name=name) if name in xls.sheet_names else pd.DataFrame()

    run_meta = {}
    if "run_meta" in xls.sheet_names:
        meta_df = pd.read_excel(xls, sheet_name="run_meta")
        if not meta_df.empty:
            run_meta = meta_df.iloc[0].to_dict()
    run_meta.setdefault("backend", backend_name)
    run_meta.setdefault("source", "saved_notebook_artifacts")
    run_meta["artifact_dir"] = str(artifact_dir)

    return {
        "backend": backend_name,
        "scenario_summary": _read("scenario_summary"),
        "scenario_actions": _read("scenario_actions"),
        "route_output_all": _read("route_output_all"),
        "stop_output_all": _read("stop_output_all"),
        "order_output_all": _read("order_output_all"),
        "scenario_timings": _read("scenario_timings"),
        "requirement_checklist": _read("requirement_checklist"),
        "run_meta": run_meta,
    }


def _save_png_route_map(
    stops_df: pd.DataFrame,
    depots_df: pd.DataFrame,
    scenario_name: str,
    png_path: Path,
    route_name: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    scenario_stops = stops_df.loc[stops_df["Scenario"].astype(str) == str(scenario_name)].copy()
    if route_name is not None:
        scenario_stops = scenario_stops.loc[scenario_stops["RouteName"].astype(str) == str(route_name)].copy()

    if scenario_stops.empty:
        plt.text(0.5, 0.5, "No stops available", ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close()
        return

    colors = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ]

    route_names = scenario_stops["RouteName"].dropna().astype(str).unique().tolist()
    for idx, route in enumerate(route_names):
        route_stops = scenario_stops.loc[scenario_stops["RouteName"].astype(str) == route].copy()
        if "Sequence" in route_stops.columns:
            route_stops = route_stops.sort_values("Sequence")
        color = colors[idx % len(colors)]
        ax.plot(
            route_stops["Longitude"].astype(float),
            route_stops["Latitude"].astype(float),
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=route,
            color=color,
        )
        if not depots_df.empty and "StartDepotName" in route_stops.columns:
            non_null = route_stops["StartDepotName"].dropna()
            depot_name = str(non_null.iloc[0]) if not non_null.empty else None
            if depot_name is not None and "Name" in depots_df.columns:
                depot_slice = depots_df.loc[depots_df["Name"].astype(str) == depot_name].head(1)
                if not depot_slice.empty:
                    dep_lon = float(depot_slice.iloc[0]["Longitude"])
                    dep_lat = float(depot_slice.iloc[0]["Latitude"])
                    ax.scatter([dep_lon], [dep_lat], marker="s", s=80, color=color, edgecolor="black")
                    first_stop = route_stops.iloc[0]
                    ax.plot([dep_lon, float(first_stop["Longitude"])], [dep_lat, float(first_stop["Latitude"])], linestyle="--", linewidth=1, color=color)

    ax.set_title(f"{scenario_name} — {route_name if route_name else 'All routes'}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    if len(route_names) <= 15:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_backend_artifacts(
    root_dir: str | Path,
    backend_name: str,
    bundle: Dict[str, Any],
    inputs: Dict[str, pd.DataFrame],
) -> Path:
    artifact_dir = _backend_artifact_dir(root_dir, backend_name)
    maps_html_dir = artifact_dir / "maps_html"
    maps_png_dir = artifact_dir / "maps_png"
    maps_html_dir.mkdir(parents=True, exist_ok=True)
    maps_png_dir.mkdir(parents=True, exist_ok=True)

    workbook_path = artifact_dir / "vrp_scenario_results_full.xlsx"
    workbook_path.write_bytes(build_backend_workbook(bundle))

    scenario_summary = bundle.get("scenario_summary", pd.DataFrame())
    stops_df = bundle.get("stop_output_all", pd.DataFrame())
    if not scenario_summary.empty and not stops_df.empty and "Scenario" in scenario_summary.columns:
        for scenario_name in scenario_summary["Scenario"].dropna().astype(str).tolist():
            scenario_stops = stops_df.loc[stops_df["Scenario"].astype(str) == str(scenario_name)].copy()
            if scenario_stops.empty:
                continue

            all_name = f"{_sanitize_name(scenario_name)}__all_routes"
            map_obj = render_route_map(
                stops_df=stops_df,
                depots_df=inputs.get("depots", pd.DataFrame()),
                scenario_name=scenario_name,
                route_name=None,
            )
            map_obj.save(str(maps_html_dir / f"{all_name}.html"))
            _save_png_route_map(
                stops_df=stops_df,
                depots_df=inputs.get("depots", pd.DataFrame()),
                scenario_name=scenario_name,
                png_path=maps_png_dir / f"{all_name}.png",
                route_name=None,
            )

            if "RouteName" in scenario_stops.columns:
                for route_name in scenario_stops["RouteName"].dropna().astype(str).unique().tolist():
                    safe_name = f"{_sanitize_name(scenario_name)}__{_sanitize_name(route_name)}"
                    route_map = render_route_map(
                        stops_df=stops_df,
                        depots_df=inputs.get("depots", pd.DataFrame()),
                        scenario_name=scenario_name,
                        route_name=route_name,
                    )
                    route_map.save(str(maps_html_dir / f"{safe_name}.html"))
                    _save_png_route_map(
                        stops_df=stops_df,
                        depots_df=inputs.get("depots", pd.DataFrame()),
                        scenario_name=scenario_name,
                        png_path=maps_png_dir / f"{safe_name}.png",
                        route_name=route_name,
                    )

    run_meta = dict(bundle.get("run_meta", {}))
    run_meta["artifact_dir"] = str(artifact_dir)
    run_meta["saved_workbook"] = str(workbook_path)
    run_meta["maps_html_dir"] = str(maps_html_dir)
    run_meta["maps_png_dir"] = str(maps_png_dir)
    (artifact_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")

    return artifact_dir


def _read_excel_sheet_if_present(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    if sheet_name not in xls.sheet_names:
        return pd.DataFrame()
    return pd.read_excel(xls, sheet_name=sheet_name)


def load_uploaded_esri_workbook(
    workbook: Any,
    inputs: Dict[str, pd.DataFrame],
    config: BackendConfig,
    backend_name: str = "Esri (Uploaded)",
) -> Dict[str, Any]:
    xls = pd.ExcelFile(workbook)
    required = {"Scenario_Summary", "Route_Output_All", "Stop_Output_All"}
    missing = sorted(required.difference(set(xls.sheet_names)))
    if missing:
        raise RuntimeError(
            "Uploaded Esri workbook is missing required sheets: "
            + ", ".join(missing)
        )

    scenario_meta = _default_scenario_meta()

    summary_raw = _ffill_scenario_columns(_read_excel_sheet_if_present(xls, "Scenario_Summary"))
    routes_raw = _ffill_scenario_columns(_read_excel_sheet_if_present(xls, "Route_Output_All"))
    stops_raw = _ffill_scenario_columns(_read_excel_sheet_if_present(xls, "Stop_Output_All"))
    actions_raw = _ffill_scenario_columns(_read_excel_sheet_if_present(xls, "Scenario_Actions"))
    timings_raw = _read_excel_sheet_if_present(xls, "Timing_Log")
    requirements_raw = _read_excel_sheet_if_present(xls, "Requirement_Checklist")
    orders_input_raw = _ffill_scenario_columns(_read_excel_sheet_if_present(xls, "Orders_Input_All"))

    for df in [summary_raw, routes_raw, stops_raw, actions_raw, orders_input_raw]:
        if not df.empty and "Scenario" in df.columns:
            df["Scenario"] = df["Scenario"].apply(_normalize_uploaded_scenario_key)

    if routes_raw.empty:
        raise RuntimeError("Uploaded Esri workbook has no route rows.")
    if stops_raw.empty:
        raise RuntimeError("Uploaded Esri workbook has no stop rows.")

    routes_raw = routes_raw.loc[routes_raw.get("Scenario").notna()].copy()
    stops_raw = stops_raw.loc[stops_raw.get("Scenario").notna()].copy()
    if not summary_raw.empty and "Scenario" in summary_raw.columns:
        summary_raw = summary_raw.loc[summary_raw["Scenario"].notna()].copy()
    if not actions_raw.empty and "Scenario" in actions_raw.columns:
        actions_raw = actions_raw.loc[actions_raw["Scenario"].notna()].copy()
    if not orders_input_raw.empty and "Scenario" in orders_input_raw.columns:
        orders_input_raw = orders_input_raw.loc[orders_input_raw["Scenario"].notna()].copy()

    if "StopType" in stops_raw.columns:
        stop_type = pd.to_numeric(stops_raw["StopType"], errors="coerce")
        stops_raw = stops_raw.loc[stop_type.fillna(0).isin([0, 1])].copy()

    def _area_for(s: Any) -> str:
        key = _normalize_uploaded_scenario_key(s)
        if key in scenario_meta:
            return scenario_meta[key]["Area"]
        return "Uploaded Esri"

    routes_norm = pd.DataFrame({
        "Scenario": routes_raw["Scenario"].astype(str),
        "Area": routes_raw["Scenario"].apply(_area_for),
        "RouteName": _series_from(routes_raw, routes_raw.get("RouteName", routes_raw.get("Name", None)), "").astype(str),
        "StartDepotName": _series_from(routes_raw, routes_raw.get("StartDepotName", ""), "").astype(str),
        "AssignedOrders": _num_series(routes_raw, routes_raw.get("OrderCount", routes_raw.get("OrdersCount", None)), 0).astype(int),
        "StartTime": _dt_series(routes_raw, routes_raw.get("StartTime", None)),
        "EndTime": _dt_series(routes_raw, routes_raw.get("EndTime", None)),
        "TotalDistanceKm": _num_series(routes_raw, routes_raw.get("TotalDistance", routes_raw.get("TotalDistanceKm", None)), 0.0),
        "TotalTravelMinutes": _num_series(routes_raw, routes_raw.get("TotalTravelTime", routes_raw.get("TotalTravelMinutes", None)), 0.0),
        "TotalServiceMinutes": _num_series(routes_raw, routes_raw.get("TotalOrderServiceTime", routes_raw.get("TotalServiceTime", routes_raw.get("TotalServiceMinutes", None))), 0.0),
        "LateMinutes": _num_series(routes_raw, routes_raw.get("TotalViolationTime", routes_raw.get("LateMinutes", None)), 0.0),
        "OvertimeMinutes": _num_series(routes_raw, routes_raw.get("TotalOvertime", routes_raw.get("OvertimeMinutes", None)), 0.0),
        "TechnicianRating": pd.Series([np.nan] * len(routes_raw)),
        "SpecialtyNames": pd.Series([""] * len(routes_raw)),
    })

    service_start_series = _dt_series(stops_raw, stops_raw.get("ServiceStart", stops_raw.get("ArriveTime", None)))
    arrival_series = _dt_series(stops_raw, stops_raw.get("ArriveTime", stops_raw.get("ArrivalTime", None)))
    depart_series = _dt_series(stops_raw, stops_raw.get("DepartTime", None))
    if depart_series.notna().any() and arrival_series.notna().any():
        inferred_service_minutes = (depart_series - arrival_series).dt.total_seconds().div(60.0).fillna(0.0)
    else:
        inferred_service_minutes = pd.Series([0.0] * len(stops_raw), index=stops_raw.index)

    stops_norm = pd.DataFrame({
        "Scenario": stops_raw["Scenario"].astype(str),
        "Area": stops_raw["Scenario"].apply(_area_for),
        "RouteName": _series_from(stops_raw, stops_raw.get("RouteName", stops_raw.get("Name", None)), "").astype(str),
        "Sequence": _num_series(stops_raw, stops_raw.get("Sequence", None), 0).astype(int),
        "OrderId": _series_from(stops_raw, stops_raw.get("OrderId", stops_raw.get("Name", None)), "").astype(str),
        "OrderName": _series_from(stops_raw, stops_raw.get("Name", stops_raw.get("OrderName", None)), "").astype(str),
        "Latitude": _num_series(stops_raw, stops_raw.get("Latitude", stops_raw.get("SnapY", None)), np.nan),
        "Longitude": _num_series(stops_raw, stops_raw.get("Longitude", stops_raw.get("SnapX", None)), np.nan),
        "ArrivalTime": arrival_series,
        "ServiceStart": service_start_series,
        "DepartTime": depart_series,
        "LateMinutes": _num_series(stops_raw, stops_raw.get("ViolationTime", stops_raw.get("LateMinutes", None)), 0.0),
        "TravelMinutes": _num_series(stops_raw, stops_raw.get("FromPrevTravelTime", stops_raw.get("TravelMinutes", None)), 0.0),
        "ServiceMinutes": inferred_service_minutes,
        "Priority": pd.Series([np.nan] * len(stops_raw)),
        "GoalProfile": pd.Series([""] * len(stops_raw)),
        "SpecialtyNames": pd.Series([""] * len(stops_raw)),
    })

    if not orders_input_raw.empty:
        orders_norm = orders_input_raw.copy()
        if "Name" in orders_norm.columns and "OrderId" not in orders_norm.columns:
            orders_norm["OrderId"] = orders_norm["Name"].astype(str)
        elif "OrderId" not in orders_norm.columns:
            orders_norm["OrderId"] = pd.Series([f"ORDER_{i+1}" for i in range(len(orders_norm))], index=orders_norm.index)

        orders_norm["Area"] = orders_norm["Scenario"].apply(_area_for)
        assigned_flags = []
        assigned_route_names = []
        route_lookup = (
            stops_norm.drop_duplicates(["Scenario", "OrderId"])
            .set_index(["Scenario", "OrderId"])["RouteName"]
            .to_dict()
        )
        for _, row in orders_norm.iterrows():
            scenario_key = str(row["Scenario"])
            order_id = str(row["OrderId"])
            route_name = route_lookup.get((scenario_key, order_id))
            assigned_route_names.append(route_name if route_name is not None else "UNASSIGNED")
            assigned_flags.append(int(route_name is not None))
        orders_norm["Assigned"] = assigned_flags
        orders_norm["AssignedRouteName"] = assigned_route_names
    else:
        orders_norm = (
            stops_norm[["Scenario", "Area", "OrderId", "OrderName", "RouteName"]]
            .drop_duplicates()
            .rename(columns={"RouteName": "AssignedRouteName"})
        )
        orders_norm["Assigned"] = 1

    if not summary_raw.empty:
        summary_base = summary_raw.copy()
        summary_base = summary_base.drop_duplicates(subset=["Scenario"]).reset_index(drop=True)
        summary_base["Area"] = summary_base["Scenario"].apply(_area_for)
        summary_base["ScenarioLabel"] = summary_base.get("ScenarioTitle", pd.Series([""] * len(summary_base))).fillna("")
        summary_base["ScenarioLabel"] = summary_base.apply(
            lambda r: r["ScenarioLabel"] if str(r["ScenarioLabel"]).strip() else scenario_meta.get(str(r["Scenario"]), {}).get("ScenarioLabel", str(r["Scenario"])),
            axis=1,
        )
        summary_norm = pd.DataFrame({
            "Scenario": summary_base["Scenario"].astype(str),
            "Area": summary_base["Area"].astype(str),
            "AssignedOrders": _num_series(summary_base, summary_base.get("OrdersServed", None), 0).astype(int),
            "UnassignedOrders": (
                _num_series(summary_base, summary_base.get("InputOrders", None), 0)
                - _num_series(summary_base, summary_base.get("OrdersServed", None), 0)
            ).clip(lower=0).astype(int),
            "RoutesUsed": _num_series(summary_base, summary_base.get("OutputRouteCount", None), 0).astype(int),
            "TotalDistanceKm": _num_series(summary_base, summary_base.get("TotalDistance", None), 0.0),
            "LateMinutes": pd.Series(
                [
                    float(routes_norm.loc[routes_norm["Scenario"] == str(s), "LateMinutes"].sum())
                    for s in summary_base["Scenario"].astype(str)
                ]
            ),
            "OvertimeMinutes": pd.Series(
                [
                    float(routes_norm.loc[routes_norm["Scenario"] == str(s), "OvertimeMinutes"].sum())
                    for s in summary_base["Scenario"].astype(str)
                ]
            ),
            "TotalTravelMinutes": _num_series(summary_base, summary_base.get("TotalTravelTime", summary_base.get("TotalTime", None)), 0.0),
            "ScenarioLabel": summary_base["ScenarioLabel"].astype(str),
            "Backend": backend_name,
        })
    else:
        summary_norm = (
            routes_norm.groupby(["Scenario", "Area"], as_index=False)
            .agg(
                RoutesUsed=("RouteName", "nunique"),
                TotalDistanceKm=("TotalDistanceKm", "sum"),
                LateMinutes=("LateMinutes", "sum"),
                OvertimeMinutes=("OvertimeMinutes", "sum"),
                TotalTravelMinutes=("TotalTravelMinutes", "sum"),
            )
        )
        assigned = stops_norm.groupby("Scenario")["OrderId"].nunique().rename("AssignedOrders")
        summary_norm = summary_norm.merge(assigned, on="Scenario", how="left")
        summary_norm["AssignedOrders"] = summary_norm["AssignedOrders"].fillna(0).astype(int)
        summary_norm["UnassignedOrders"] = 0
        summary_norm["ScenarioLabel"] = summary_norm["Scenario"].map(
            lambda s: scenario_meta.get(str(s), {}).get("ScenarioLabel", str(s))
        )
        summary_norm["Backend"] = backend_name

    desired_order = [key for key, _, _ in DEFAULT_SCENARIOS]
    order_rank = {key: idx for idx, key in enumerate(desired_order)}
    summary_norm["__order"] = summary_norm["Scenario"].map(lambda s: order_rank.get(str(s), 999))
    summary_norm = summary_norm.sort_values(["__order", "Scenario"]).drop(columns="__order").reset_index(drop=True)

    if not actions_raw.empty:
        actions_norm = pd.DataFrame({
            "Backend": backend_name,
            "Scenario": actions_raw["Scenario"].astype(str),
            "Action": _series_from(actions_raw, actions_raw.get("ActionDescription", actions_raw.get("Requirement", None)), "").astype(str),
        })
    else:
        explainer = ExplainAgent()
        actions_rows = []
        for scenario_key in summary_norm["Scenario"].astype(str).tolist():
            for action in explainer.actions_for(scenario_key):
                actions_rows.append({"Backend": backend_name, "Scenario": scenario_key, "Action": action})
        actions_norm = pd.DataFrame(actions_rows)

    if not timings_raw.empty:
        timings_base = timings_raw.copy()
        scenario_candidates = None
        for candidate in ["scenario_title", "Scenario", "ScenarioTitle"]:
            if candidate in timings_base.columns:
                scenario_candidates = timings_base[candidate]
                break
        if scenario_candidates is None:
            timings_base["Scenario"] = ""
        else:
            timings_base["Scenario"] = scenario_candidates.apply(_normalize_uploaded_scenario_key)
        timings_base["ScenarioLabel"] = timings_base["Scenario"].map(
            lambda s: scenario_meta.get(str(s), {}).get("ScenarioLabel", str(s))
        )
        runtime_col = None
        for candidate in ["cell_execution_seconds", "RuntimeSeconds"]:
            if candidate in timings_base.columns:
                runtime_col = candidate
                break
        if runtime_col is None:
            timings_base["RuntimeSeconds"] = 0.0
            runtime_col = "RuntimeSeconds"
        timings_norm = (
            timings_base.loc[timings_base["Scenario"].notna()]
            .groupby(["Scenario", "ScenarioLabel"], as_index=False)[runtime_col]
            .sum()
            .rename(columns={runtime_col: "RuntimeSeconds"})
        )
        timings_norm["Backend"] = backend_name
    else:
        timings_norm = summary_norm[["Scenario", "ScenarioLabel"]].copy()
        timings_norm["RuntimeSeconds"] = np.nan
        timings_norm["Backend"] = backend_name

    if not requirements_raw.empty:
        requirement_norm = requirements_raw.copy()
        requirement_norm["Backend"] = backend_name
    else:
        requirement_norm = map_catalog_to_requirements().copy()
        requirement_norm["Backend"] = backend_name

    run_meta = {
        "backend": backend_name,
        "source": "uploaded_esri_workbook",
        "workbook_sheets": list(xls.sheet_names),
        "subset_orders": int(orders_norm["Scenario"].notna().sum()) if "Scenario" in orders_norm.columns else int(len(orders_norm)),
        "subset_routes": int(routes_norm["RouteName"].nunique()) if not routes_norm.empty else 0,
        "uploaded_orders": int(len(inputs["orders"])),
        "uploaded_routes": int(len(inputs["routes"])),
        "uploaded_depots": int(len(inputs["depots"])),
        "config": config.__dict__,
    }

    return {
        "backend": backend_name,
        "scenario_summary": summary_norm.reset_index(drop=True),
        "scenario_actions": actions_norm.reset_index(drop=True),
        "route_output_all": routes_norm.reset_index(drop=True),
        "stop_output_all": stops_norm.reset_index(drop=True),
        "order_output_all": orders_norm.reset_index(drop=True),
        "scenario_timings": timings_norm.reset_index(drop=True),
        "requirement_checklist": requirement_norm.reset_index(drop=True),
        "run_meta": run_meta,
    }


def run_backend_suite(inputs: Dict[str, pd.DataFrame], config: BackendConfig, backends: List[str], progress_cb: Optional[Callable[[str, int, int, str], None]] = None) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        if backend == "Impacted-Subset Greedy + 2-opt":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_current_backend(inputs, config, progress_cb=cb)
        elif backend == "Hybrid Execution-Aware Rolling VRP Solver":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_hybrid_execution_aware_backend(inputs, config, progress_cb=cb)
        elif backend == "Adaptive Execution-Aware Metaheuristic Solver":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_adaptive_execution_aware_metaheuristic_backend(inputs, config, progress_cb=cb)
        elif backend == "OSRM":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_osrm_backend(inputs, config, progress_cb=cb)
        elif backend == "Esri":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_esri_backend(inputs, config, progress_cb=cb)
        elif backend == "PyVRP":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_pyvrp_backend(inputs, config, progress_cb=cb)
        elif backend == "OR-Tools":
            cb = None if progress_cb is None else lambda i, t, name, b=backend: progress_cb(b, i, t, name)
            results[backend] = run_ortools_backend(inputs, config, progress_cb=cb)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    return results


def build_compare_excel(results: Dict[str, Dict[str, Any]]) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for backend, bundle in results.items():
            prefix = backend.replace(" ", "_")[:20]
            bundle["scenario_summary"].to_excel(writer, sheet_name=f"{prefix}_summary"[:31], index=False)
            bundle["route_output_all"].to_excel(writer, sheet_name=f"{prefix}_routes"[:31], index=False)
            bundle["stop_output_all"].to_excel(writer, sheet_name=f"{prefix}_stops"[:31], index=False)
            bundle["scenario_timings"].to_excel(writer, sheet_name=f"{prefix}_timings"[:31], index=False)
    out.seek(0)
    return out.getvalue()
