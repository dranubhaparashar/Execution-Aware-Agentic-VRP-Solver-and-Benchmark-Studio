
from __future__ import annotations

import copy
import io
import math
import random
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import folium
import nbformat
import numpy as np
import pandas as pd

DEFAULT_SCENARIOS = [
    ("BASE", "Base scenario", "BASE"),
    ("EAS_1", "Job starts earlier or later than planned", "Execution-Aware Scheduling"),
    ("EAS_2", "Job finishes earlier or later", "Execution-Aware Scheduling"),
    ("EAS_3", "Technician skips scheduled job order", "Execution-Aware Scheduling"),
    ("EAS_4", "Automatically re-optimize downstream activities", "Execution-Aware Scheduling"),
    ("RT_1", "New or urgent activity", "Real-Time Scheduling"),
    ("RT_2", "Cancellation", "Real-Time Scheduling"),
    ("RT_3", "Duration changes", "Real-Time Scheduling"),
    ("RT_4", "Technician availability or shift changes", "Real-Time Scheduling"),
    ("RT_5", "Location hours changes", "Real-Time Scheduling"),
    ("RT_6", "Technician calls in sick", "Real-Time Scheduling"),
    ("RT_7", "All EAS scenarios included in real-time flow", "Real-Time Scheduling"),
    ("GB_1", "Minimize travel time", "Goal-Based Scheduling"),
    ("GB_2", "Prefer higher-rated technicians", "Goal-Based Scheduling"),
    ("GB_3", "Prefer specialized technicians", "Goal-Based Scheduling"),
    ("GB_4", "Maximize on-time completion / SLA adherence", "Goal-Based Scheduling"),
    ("GB_5", "Reduce overtime", "Goal-Based Scheduling"),
    ("GB_6", "Customer-defined soft-constraint weights", "Goal-Based Scheduling"),
]


def load_inputs_from_uploads(
    orders_file,
    routes_file,
    accounts_file,
    depots_file,
    goal_profiles_file,
) -> Dict[str, pd.DataFrame]:
    orders = pd.read_csv(orders_file)
    routes = pd.read_csv(routes_file)
    accounts = pd.read_csv(accounts_file)
    depots = pd.read_csv(depots_file)
    goal_profiles = pd.read_csv(goal_profiles_file)

    orders["Priority"] = pd.to_numeric(orders.get("Priority"), errors="coerce").fillna(3).astype(int)
    orders["ServiceTime"] = pd.to_numeric(orders.get("ServiceTime"), errors="coerce").fillna(45).astype(float)
    orders["MaxViolationTime1"] = pd.to_numeric(orders.get("MaxViolationTime1"), errors="coerce").fillna(0.0)
    orders["TimeWindowStart1"] = pd.to_datetime(orders.get("TimeWindowStart1"), errors="coerce")
    orders["TimeWindowEnd1"] = pd.to_datetime(orders.get("TimeWindowEnd1"), errors="coerce")
    orders["SLAFlag"] = pd.to_numeric(orders.get("SLAFlag"), errors="coerce").fillna(0).astype(int)
    orders["Latitude"] = pd.to_numeric(orders.get("Latitude"), errors="coerce")
    orders["Longitude"] = pd.to_numeric(orders.get("Longitude"), errors="coerce")
    orders["OrderId"] = orders.get("Name", pd.Series(range(1, len(orders) + 1), index=orders.index)).astype(str)

    routes["EarliestStartTime"] = pd.to_datetime(routes.get("EarliestStartTime"), errors="coerce")
    routes["LatestStartTime"] = pd.to_datetime(routes.get("LatestStartTime"), errors="coerce")
    routes["TechnicianRating"] = pd.to_numeric(routes.get("TechnicianRating"), errors="coerce").fillna(4.0)
    routes["MaxOrderCount"] = pd.to_numeric(routes.get("MaxOrderCount"), errors="coerce").fillna(40).astype(int)
    routes["OvertimeStartTime"] = pd.to_numeric(routes.get("OvertimeStartTime"), errors="coerce").fillna(540.0)
    routes["MaxTotalTime"] = pd.to_numeric(routes.get("MaxTotalTime"), errors="coerce").fillna(720.0)
    routes["Name"] = routes["Name"].astype(str)

    depots["Latitude"] = pd.to_numeric(depots.get("Latitude"), errors="coerce")
    depots["Longitude"] = pd.to_numeric(depots.get("Longitude"), errors="coerce")
    depots["Name"] = depots["Name"].astype(str)

    goal_profiles = goal_profiles.fillna(0.0)
    if "GoalProfile" not in orders.columns:
        orders = orders.merge(accounts, on="AccountId", how="left")
    else:
        orders = orders.merge(accounts, on="AccountId", how="left", suffixes=("", "_acct"))
        orders["GoalProfile"] = orders["GoalProfile"].fillna(orders.get("GoalProfile_acct"))

    return {
        "orders": orders,
        "routes": routes,
        "accounts": accounts,
        "depots": depots,
        "goal_profiles": goal_profiles,
    }


def extract_notebook_headings(uploaded_file) -> List[str]:
    nb = nbformat.read(io.BytesIO(uploaded_file.getvalue()), as_version=4)
    headings: List[str] = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            first = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
            if first.startswith("##"):
                headings.append(first.lstrip("# ").strip())
    return headings


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _skill_set(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}


def goal_weights(goal_profiles_df: pd.DataFrame, profile_name: Optional[str], override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    defaults = {
        "travel": 1.0,
        "rating": 0.4,
        "sla": 0.6,
        "overtime": 0.5,
        "skill": 0.9,
        "priority": 0.4,
    }
    if profile_name and profile_name in set(goal_profiles_df["GoalProfile"].astype(str)):
        row = goal_profiles_df.loc[goal_profiles_df["GoalProfile"].astype(str) == str(profile_name)].iloc[0]
        defaults = {
            "travel": float(row.get("minimize_travel_time", defaults["travel"])),
            "rating": float(row.get("prefer_higher_rated_technicians", defaults["rating"])),
            "sla": float(row.get("maximize_sla_adherence", defaults["sla"])),
            "overtime": float(row.get("reduce_overtime", defaults["overtime"])),
            "skill": max(float(row.get("prefer_higher_rated_technicians", 0.5)), 0.5),
            "priority": 0.4,
        }
    if override:
        defaults.update(override)
    return defaults


def build_impacted_subset(
    orders: pd.DataFrame,
    routes: pd.DataFrame,
    depots: pd.DataFrame,
    max_orders: int,
    max_routes: int,
    depot_focus: Optional[str],
    random_seed: int,
    sample_by_priority: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(random_seed)

    route_pool = routes.copy()
    if depot_focus:
        focused = route_pool.loc[route_pool["StartDepotName"].astype(str) == str(depot_focus)].copy()
        if len(focused) >= min(5, max_routes):
            route_pool = focused
    else:
        top_depot = route_pool["StartDepotName"].astype(str).value_counts().index[0]
        route_pool = route_pool.loc[route_pool["StartDepotName"].astype(str) == top_depot].copy()

    if len(route_pool) > max_routes:
        route_pool = route_pool.sample(max_routes, random_state=random_seed)

    allowed_depots = set(route_pool["StartDepotName"].astype(str))
    order_pool = orders.loc[orders["PreferredDepot"].astype(str).isin(allowed_depots)].copy()
    if order_pool.empty:
        order_pool = orders.copy()

    if sample_by_priority:
        order_pool = order_pool.sort_values(
            by=["SLAFlag", "Priority", "TimeWindowStart1"],
            ascending=[False, False, True],
            na_position="last",
        )

    if len(order_pool) > max_orders:
        head_n = int(max_orders * 0.65)
        sampled = order_pool.head(head_n)
        remaining = order_pool.iloc[head_n:]
        if not remaining.empty and len(sampled) < max_orders:
            tail = remaining.sample(max_orders - len(sampled), random_state=random_seed)
            order_pool = pd.concat([sampled, tail], ignore_index=True)
        else:
            order_pool = sampled

    return order_pool.reset_index(drop=True), route_pool.reset_index(drop=True)


class IntakeAgent:
    def __init__(self, inputs: Dict[str, pd.DataFrame]):
        self.inputs = inputs

    def summarize(self) -> Dict[str, Any]:
        return {
            "orders": len(self.inputs["orders"]),
            "routes": len(self.inputs["routes"]),
            "depots": len(self.inputs["depots"]),
            "accounts": len(self.inputs["accounts"]),
        }


class PolicyAgent:
    def __init__(self, goal_profiles: pd.DataFrame, strict_skill_bias: bool = True):
        self.goal_profiles = goal_profiles
        self.strict_skill_bias = strict_skill_bias

    def weights_for_order(self, order_row: pd.Series, override: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        gp = order_row.get("GoalProfile")
        weights = goal_weights(self.goal_profiles, gp, override=override)
        if self.strict_skill_bias:
            weights["skill"] = max(weights["skill"], 0.95)
        return weights


class AssignmentAgent:
    def __init__(self, depots_df: pd.DataFrame, speed_kmph: float = 35.0, use_two_opt: bool = True):
        self.speed_kmph = speed_kmph
        self.use_two_opt = use_two_opt
        self.depot_lookup = depots_df.set_index("Name")[["Latitude", "Longitude"]].to_dict("index")

    def _score_route(self, order: pd.Series, route: pd.Series, weights: Dict[str, float], route_state: Dict[str, Any]) -> float:
        depot_name = str(route.get("StartDepotName"))
        depot = self.depot_lookup.get(depot_name, {})
        d_km = float(
            haversine_km(
                order["Latitude"],
                order["Longitude"],
                depot.get("Latitude", order["Latitude"]),
                depot.get("Longitude", order["Longitude"]),
            )
        )
        rating_penalty = (5.5 - float(route.get("TechnicianRating", 4.0))) * 10.0
        load_ratio = route_state["assigned_count"] / max(float(route.get("MaxOrderCount", 1)), 1.0)
        overtime_risk = max(0.0, route_state["current_minutes"] + float(order.get("ServiceTime", 45)) - float(route.get("OvertimeStartTime", 540)))
        tw_end = order.get("TimeWindowEnd1")
        tw_start = order.get("TimeWindowStart1")
        eta_pressure = 0.0
        if pd.notna(tw_end) and pd.notna(route.get("EarliestStartTime")):
            travel_min = d_km / max(self.speed_kmph, 1.0) * 60.0
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

    def _improve_route_two_opt(self, df: pd.DataFrame, depot_lat: float, depot_lon: float) -> pd.DataFrame:
        if not self.use_two_opt or len(df) < 5 or len(df) > 12:
            return df
        coords = [(depot_lat, depot_lon)] + list(zip(df["Latitude"], df["Longitude"])) + [(depot_lat, depot_lon)]
        best_order = list(range(len(df)))
        best_distance = self._path_distance(coords)

        improved = True
        attempts = 0
        while improved and attempts < 2:
            improved = False
            attempts += 1
            for i in range(1, len(best_order) - 1):
                for j in range(i + 1, len(best_order)):
                    new_order = best_order[:]
                    new_order[i:j] = reversed(new_order[i:j])
                    new_coords = [(depot_lat, depot_lon)] + [(df.iloc[k]["Latitude"], df.iloc[k]["Longitude"]) for k in new_order] + [(depot_lat, depot_lon)]
                    new_distance = self._path_distance(new_coords)
                    if new_distance + 1e-6 < best_distance:
                        best_distance = new_distance
                        best_order = new_order
                        improved = True
            if improved:
                df = df.iloc[best_order].reset_index(drop=True)
        return df

    def _path_distance(self, coords: list[tuple[float, float]]) -> float:
        dist = 0.0
        for (la1, lo1), (la2, lo2) in zip(coords[:-1], coords[1:]):
            dist += float(haversine_km(la1, lo1, la2, lo2))
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
                    {
                        "Scenario": scenario_name,
                        "Area": area,
                        "OrderId": order["OrderId"],
                        "RouteName": None,
                        "Assigned": 0,
                        "Reason": "No feasible route",
                    }
                )
                continue

            feasible.sort(key=lambda x: x[0])
            best_route_name = feasible[0][1]
            route_states[best_route_name]["assigned_count"] += 1
            route_states[best_route_name]["current_minutes"] += float(order.get("ServiceTime", 45))
            route_states[best_route_name]["assigned_orders"].append(order["OrderId"])

            assignment_records.append(
                {
                    "Scenario": scenario_name,
                    "Area": area,
                    "OrderId": order["OrderId"],
                    "AssignedRouteName": best_route_name,
                    "Assigned": 1,
                    "Reason": "Greedy weighted assignment",
                }
            )

        assignment_df = pd.DataFrame(assignment_records)
        orders_enriched = orders_df.merge(
            assignment_df[["OrderId", "AssignedRouteName", "Assigned", "Reason"]],
            on="OrderId",
            how="left",
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
            depot_lat = depot.get("Latitude", assigned_orders["Latitude"].mean())
            depot_lon = depot.get("Longitude", assigned_orders["Longitude"].mean())

            assigned_orders = assigned_orders.sort_values(
                by=["TimeWindowStart1", "Priority"],
                ascending=[True, False],
                na_position="last",
            ).reset_index(drop=True)

            # nearest-neighbor pass
            ordered_idx = []
            remaining = assigned_orders.copy()
            current_lat, current_lon = depot_lat, depot_lon
            while not remaining.empty:
                dists = haversine_km(
                    remaining["Latitude"].astype(float).to_numpy(),
                    remaining["Longitude"].astype(float).to_numpy(),
                    current_lat,
                    current_lon,
                )
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
                d_km = float(haversine_km(current_lat, current_lon, order["Latitude"], order["Longitude"]))
                travel_min = d_km / max(self.speed_kmph, 1.0) * 60.0
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

            return_d = float(haversine_km(current_lat, current_lon, depot_lat, depot_lon))
            return_min = return_d / max(self.speed_kmph, 1.0) * 60.0
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

        summary = pd.DataFrame(
            [
                {
                    "Scenario": scenario_name,
                    "Area": area,
                    "AssignedOrders": assigned_orders,
                    "UnassignedOrders": unassigned,
                    "RoutesUsed": routes_used,
                    "TotalDistanceKm": total_distance,
                    "LateMinutes": total_late,
                    "OvertimeMinutes": total_overtime,
                    "TotalTravelMinutes": total_travel,
                }
            ]
        )

        return {
            "scenario_summary": summary,
            "orders": orders_enriched,
            "routes": routes_out,
            "stops": stops_out,
            "assignments": assignment_df,
        }


class ExplainAgent:
    def actions_for(self, scenario_key: str) -> List[str]:
        mapping = {
            "BASE": ["Build baseline on impacted subset selected from uploaded universe."],
            "EAS_1": ["Shift a high-impact order start window earlier/later.", "Re-run weighted assignment and resequencing."],
            "EAS_2": ["Change service duration of one active order.", "Re-sequence impacted route."],
            "EAS_3": ["Skip one scheduled job from a technician.", "Push skipped job back to reassignment pool."],
            "EAS_4": ["Combine execution change with downstream re-optimization."],
            "RT_1": ["Inject one urgent work order with tight window and high priority."],
            "RT_2": ["Cancel one assigned order and resequence remaining work."],
            "RT_3": ["Update service durations on selected work orders."],
            "RT_4": ["Delay a technician shift start or reduce availability."],
            "RT_5": ["Tighten location hours / time windows for impacted work."],
            "RT_6": ["Remove one technician as sick/unavailable and reassign load."],
            "RT_7": ["Combine all execution-aware events inside real-time flow."],
            "GB_1": ["Override weights to favor travel minimization."],
            "GB_2": ["Override weights to favor higher technician ratings."],
            "GB_3": ["Bias assignment more strongly toward specialty match."],
            "GB_4": ["Override weights to favor SLA adherence and time windows."],
            "GB_5": ["Override weights to reduce overtime risk."],
            "GB_6": ["Use customer/account goal profiles as-is from uploaded goal profile table."],
        }
        return mapping.get(scenario_key, ["Scenario action applied."])


class ScenarioAgent:
    def __init__(self, random_seed: int = 42):
        self.rng = random.Random(random_seed)

    def apply(
        self,
        scenario_key: str,
        orders: pd.DataFrame,
        routes: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, float]]]:
        orders = orders.copy()
        routes = routes.copy()
        override_weights = None

        def pick_order(n=1):
            selectable = orders.sort_values(["Priority", "SLAFlag"], ascending=[False, False])
            return selectable.head(n).index.tolist()

        def pick_route(n=1):
            return routes.sort_values("TechnicianRating", ascending=False).head(n).index.tolist()

        if scenario_key == "EAS_1":
            idx = pick_order(1)
            if idx:
                orders.loc[idx, "TimeWindowStart1"] = orders.loc[idx, "TimeWindowStart1"] + pd.to_timedelta(60, unit="m")
                orders.loc[idx, "TimeWindowEnd1"] = orders.loc[idx, "TimeWindowEnd1"] + pd.to_timedelta(60, unit="m")

        elif scenario_key == "EAS_2":
            idx = pick_order(1)
            if idx:
                orders.loc[idx, "ServiceTime"] = orders.loc[idx, "ServiceTime"] + 40

        elif scenario_key == "EAS_3":
            idx = pick_order(1)
            if idx:
                orders.loc[idx, "PreferredDepot"] = routes["StartDepotName"].sample(1, random_state=7).iloc[0]
                orders.loc[idx, "Priority"] = np.maximum(orders.loc[idx, "Priority"] + 1, 5)

        elif scenario_key == "EAS_4":
            idx = pick_order(2)
            if idx:
                orders.loc[idx[:1], "ServiceTime"] = orders.loc[idx[:1], "ServiceTime"] + 30
                orders.loc[idx[1:], "TimeWindowStart1"] = orders.loc[idx[1:], "TimeWindowStart1"] + pd.to_timedelta(45, unit="m")
                orders.loc[idx[1:], "TimeWindowEnd1"] = orders.loc[idx[1:], "TimeWindowEnd1"] + pd.to_timedelta(45, unit="m")

        elif scenario_key == "RT_1":
            anchor = orders.sample(1, random_state=9).iloc[0]
            new_order = anchor.copy()
            new_order["OrderId"] = f"{anchor['OrderId']}_URGENT"
            new_order["Name"] = f"{anchor['Name']}_URGENT"
            new_order["Priority"] = 5
            new_order["ServiceTime"] = max(20.0, float(anchor["ServiceTime"]) * 0.8)
            new_order["TimeWindowStart1"] = pd.to_datetime(anchor["TimeWindowStart1"])
            new_order["TimeWindowEnd1"] = pd.to_datetime(anchor["TimeWindowStart1"]) + pd.to_timedelta(120, unit="m")
            new_order["SLAFlag"] = 1
            orders = pd.concat([orders, pd.DataFrame([new_order])], ignore_index=True)

        elif scenario_key == "RT_2":
            idx = pick_order(1)
            if idx:
                orders = orders.drop(index=idx).reset_index(drop=True)

        elif scenario_key == "RT_3":
            idx = pick_order(3)
            if idx:
                orders.loc[idx, "ServiceTime"] = orders.loc[idx, "ServiceTime"] * 1.35

        elif scenario_key == "RT_4":
            idx = pick_route(1)
            if idx:
                routes.loc[idx, "EarliestStartTime"] = routes.loc[idx, "EarliestStartTime"] + pd.to_timedelta(90, unit="m")
                routes.loc[idx, "MaxOrderCount"] = np.maximum(routes.loc[idx, "MaxOrderCount"] - 6, 1)

        elif scenario_key == "RT_5":
            idx = pick_order(3)
            if idx:
                orders.loc[idx, "TimeWindowEnd1"] = orders.loc[idx, "TimeWindowStart1"] + pd.to_timedelta(90, unit="m")

        elif scenario_key == "RT_6":
            idx = pick_route(1)
            if idx:
                routes = routes.drop(index=idx).reset_index(drop=True)

        elif scenario_key == "RT_7":
            idx = pick_order(2)
            if idx:
                orders.loc[idx[:1], "ServiceTime"] = orders.loc[idx[:1], "ServiceTime"] + 35
                orders.loc[idx[1:], "TimeWindowStart1"] = orders.loc[idx[1:], "TimeWindowStart1"] + pd.to_timedelta(30, unit="m")
                orders.loc[idx[1:], "TimeWindowEnd1"] = orders.loc[idx[1:], "TimeWindowEnd1"] + pd.to_timedelta(30, unit="m")
            ridx = pick_route(1)
            if ridx:
                routes.loc[ridx, "EarliestStartTime"] = routes.loc[ridx, "EarliestStartTime"] + pd.to_timedelta(45, unit="m")

        elif scenario_key == "GB_1":
            override_weights = {"travel": 2.2, "rating": 0.1, "sla": 0.35, "overtime": 0.25, "skill": 0.6, "priority": 0.15}

        elif scenario_key == "GB_2":
            override_weights = {"travel": 0.45, "rating": 2.2, "sla": 0.5, "overtime": 0.35, "skill": 0.9, "priority": 0.25}

        elif scenario_key == "GB_3":
            override_weights = {"travel": 0.5, "rating": 0.6, "sla": 0.7, "overtime": 0.45, "skill": 2.6, "priority": 0.35}

        elif scenario_key == "GB_4":
            override_weights = {"travel": 0.35, "rating": 0.35, "sla": 2.8, "overtime": 0.9, "skill": 1.0, "priority": 0.75}

        elif scenario_key == "GB_5":
            override_weights = {"travel": 0.45, "rating": 0.35, "sla": 1.1, "overtime": 3.0, "skill": 0.9, "priority": 0.4}

        elif scenario_key == "GB_6":
            override_weights = None

        return orders.reset_index(drop=True), routes.reset_index(drop=True), override_weights


def map_catalog_to_requirements() -> pd.DataFrame:
    rows = []
    requirement_text = {
        "EAS_1": "Jobs starting earlier or later than planned",
        "EAS_2": "Jobs finishing earlier or later",
        "EAS_3": "Technicians skipping scheduled job order",
        "EAS_4": "Automatically re-optimize downstream activities",
        "RT_1": "New or urgent activities",
        "RT_2": "Cancellations",
        "RT_3": "Duration changes / activity shortened or lengthened",
        "RT_4": "Technician availability or shift changes",
        "RT_5": "Location hours changes",
        "RT_6": "Technician calls in sick",
        "RT_7": "All EAS scenarios included in real-time flow",
        "GB_1": "Minimize travel time",
        "GB_2": "Prefer higher-rated technicians",
        "GB_3": "Prefer specialized technicians",
        "GB_4": "Maximize on-time completion / SLA adherence",
        "GB_5": "Reduce overtime",
        "GB_6": "Customer-defined soft-constraint weights",
        "BASE": "Baseline impacted subset solve",
    }
    for key, label, area in DEFAULT_SCENARIOS:
        rows.append(
            {
                "Scenario": key,
                "ScenarioLabel": label,
                "Area": area,
                "Requirement": requirement_text.get(key, label),
                "Status": "Implemented in interactive scenario engine",
            }
        )
    return pd.DataFrame(rows)


def run_all_scenarios(
    inputs: Dict[str, pd.DataFrame],
    max_orders: int,
    max_routes: int,
    depot_focus: Optional[str],
    random_seed: int,
    speed_kmph: float,
    fast_mode: bool,
    sample_by_priority: bool,
    use_agents: bool,
    use_two_opt: bool,
    strict_skill_bias: bool,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    intake = IntakeAgent(inputs)
    subset_orders, subset_routes = build_impacted_subset(
        orders=inputs["orders"],
        routes=inputs["routes"],
        depots=inputs["depots"],
        max_orders=max_orders,
        max_routes=max_routes,
        depot_focus=depot_focus,
        random_seed=random_seed,
        sample_by_priority=sample_by_priority,
    )

    policy = PolicyAgent(inputs["goal_profiles"], strict_skill_bias=strict_skill_bias)
    optimizer = AssignmentAgent(inputs["depots"], speed_kmph=speed_kmph, use_two_opt=use_two_opt)
    scenario_agent = ScenarioAgent(random_seed=random_seed)
    explainer = ExplainAgent()

    summaries = []
    all_orders = []
    all_routes = []
    all_stops = []
    all_actions = []
    all_timings = []

    catalog = DEFAULT_SCENARIOS
    total = len(catalog)
    for idx, (scenario_key, scenario_label, area) in enumerate(catalog, start=1):
        if progress_cb:
            progress_cb(idx, total, f"{scenario_key} — {scenario_label}")
        start = time.perf_counter()
        scenario_orders, scenario_routes, override_weights = scenario_agent.apply(scenario_key, subset_orders, subset_routes)
        result = optimizer.assign_and_sequence(
            orders_df=scenario_orders,
            routes_df=scenario_routes,
            policy_agent=policy,
            scenario_name=scenario_key,
            area=area,
            override_weights=override_weights,
        )
        runtime_s = time.perf_counter() - start

        summaries.append(result["scenario_summary"])
        all_orders.append(result["orders"])
        all_routes.append(result["routes"])
        all_stops.append(result["stops"])
        all_timings.append(
            pd.DataFrame(
                [
                    {
                        "Scenario": scenario_key,
                        "ScenarioLabel": scenario_label,
                        "Area": area,
                        "RuntimeSeconds": runtime_s,
                    }
                ]
            )
        )

        for action in explainer.actions_for(scenario_key):
            all_actions.append(
                {
                    "Scenario": scenario_key,
                    "ScenarioLabel": scenario_label,
                    "Area": area,
                    "Action": action,
                }
            )

    scenario_summary = pd.concat(summaries, ignore_index=True)
    route_output_all = pd.concat(all_routes, ignore_index=True) if all_routes else pd.DataFrame()
    stop_output_all = pd.concat(all_stops, ignore_index=True) if all_stops else pd.DataFrame()
    order_output_all = pd.concat(all_orders, ignore_index=True) if all_orders else pd.DataFrame()
    scenario_actions = pd.DataFrame(all_actions)
    scenario_timings = pd.concat(all_timings, ignore_index=True)

    requirement_checklist = map_catalog_to_requirements()
    run_meta = {
        "subset_orders": int(len(subset_orders)),
        "subset_routes": int(len(subset_routes)),
        "full_orders": int(len(inputs["orders"])),
        "full_routes": int(len(inputs["routes"])),
        "depot_focus": depot_focus or (subset_routes["StartDepotName"].astype(str).value_counts().index[0] if not subset_routes.empty else None),
        "fast_mode": fast_mode,
        "random_seed": int(random_seed),
        "speed_kmph": float(speed_kmph),
        "use_agents": bool(use_agents),
        "use_two_opt": bool(use_two_opt),
        "strict_skill_bias": bool(strict_skill_bias),
        "intake_summary": intake.summarize(),
    }

    return {
        "scenario_summary": scenario_summary,
        "route_output_all": route_output_all,
        "stop_output_all": stop_output_all,
        "order_output_all": order_output_all,
        "scenario_actions": scenario_actions,
        "scenario_timings": scenario_timings,
        "requirement_checklist": requirement_checklist,
        "run_meta": run_meta,
    }


def build_download_bundle(bundle: Dict[str, Any]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        bundle["scenario_summary"].to_excel(writer, sheet_name="Scenario_Summary", index=False)
        bundle["scenario_actions"].to_excel(writer, sheet_name="Scenario_Actions", index=False)
        bundle["requirement_checklist"].to_excel(writer, sheet_name="Requirement_Checklist", index=False)
        bundle["scenario_timings"].to_excel(writer, sheet_name="Scenario_Timings", index=False)
        bundle["route_output_all"].to_excel(writer, sheet_name="Route_Output_All", index=False)
        bundle["stop_output_all"].to_excel(writer, sheet_name="Stop_Output_All", index=False)
        bundle["order_output_all"].to_excel(writer, sheet_name="Order_Output_All", index=False)
        pd.DataFrame([bundle["run_meta"]]).to_excel(writer, sheet_name="Run_Meta", index=False)
    return output.getvalue()


def render_route_map(
    stops_df: pd.DataFrame,
    depots_df: pd.DataFrame,
    scenario_name: str,
    route_name: Optional[str] = None,
) -> folium.Map:
    def _empty_map(message: str) -> folium.Map:
        depots_ok = pd.DataFrame()
        if depots_df is not None and not depots_df.empty and {"Latitude", "Longitude"}.issubset(depots_df.columns):
            depots_ok = depots_df.loc[depots_df["Latitude"].notna() & depots_df["Longitude"].notna()].copy()

        if not depots_ok.empty:
            center_lat = float(depots_ok["Latitude"].mean())
            center_lon = float(depots_ok["Longitude"].mean())
        else:
            center_lat, center_lon = 32.85, -96.8

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
        folium.Marker([center_lat, center_lon], tooltip=message).add_to(fmap)
        for _, depot in depots_ok.iterrows():
            folium.Marker(
                [float(depot["Latitude"]), float(depot["Longitude"])],
                tooltip=str(depot.get("Name", "Depot")),
                icon=folium.Icon(color="black", icon="home"),
            ).add_to(fmap)
        return fmap

    if stops_df is None or not isinstance(stops_df, pd.DataFrame) or stops_df.empty:
        return _empty_map("No stops for selection")

    sdf = stops_df.copy()
    if "Scenario" in sdf.columns:
        sdf = sdf.loc[sdf["Scenario"].astype(str) == str(scenario_name)].copy()
    if route_name and route_name != "All routes" and "RouteName" in sdf.columns:
        sdf = sdf.loc[sdf["RouteName"].astype(str) == str(route_name)].copy()

    if sdf.empty or "Latitude" not in sdf.columns or "Longitude" not in sdf.columns:
        return _empty_map("No stops for selection")

    sdf = sdf.loc[sdf["Latitude"].notna() & sdf["Longitude"].notna()].copy()
    if sdf.empty:
        return _empty_map("No stops for selection")

    if "RouteName" not in sdf.columns:
        sdf["RouteName"] = "Route"
    if "Sequence" not in sdf.columns:
        sdf["Sequence"] = range(1, len(sdf) + 1)

    center_lat = float(sdf["Latitude"].mean())
    center_lon = float(sdf["Longitude"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    for i, (rname, group) in enumerate(sdf.groupby("RouteName", sort=True)):
        color = palette[i % len(palette)]
        ordered = group.sort_values("Sequence")
        coords = ordered[["Latitude", "Longitude"]].astype(float).values.tolist()
        if len(coords) >= 2:
            folium.PolyLine(coords, color=color, weight=4, opacity=0.9, tooltip=str(rname)).add_to(fmap)
        for _, row in ordered.iterrows():
            label = row.get("OrderName", row.get("OrderId", "Stop"))
            late_minutes = row.get("LateMinutes", 0)
            try:
                late_minutes = float(late_minutes)
            except Exception:
                late_minutes = 0.0
            popup = folium.Popup(
                f"<b>{label}</b><br>"
                f"Route: {row.get('RouteName', '')}<br>"
                f"Seq: {row.get('Sequence', '')}<br>"
                f"Late min: {late_minutes:.1f}",
                max_width=300,
            )
            folium.CircleMarker(
                [float(row["Latitude"]), float(row["Longitude"])],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.95,
                popup=popup,
                tooltip=str(label),
            ).add_to(fmap)

    if depots_df is not None and not depots_df.empty and {"Latitude", "Longitude"}.issubset(depots_df.columns):
        depots_ok = depots_df.loc[depots_df["Latitude"].notna() & depots_df["Longitude"].notna()].copy()
        for _, depot in depots_ok.iterrows():
            folium.Marker(
                [float(depot["Latitude"]), float(depot["Longitude"])] ,
                tooltip=str(depot.get("Name", "Depot")),
                icon=folium.Icon(color="black", icon="home"),
            ).add_to(fmap)

    return fmap
