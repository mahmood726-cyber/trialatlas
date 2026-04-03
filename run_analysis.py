#!/usr/bin/env python
"""TrialAtlas pipeline runner: fetch -> resolve -> build -> detect -> export.

Checks for cached raw data before hitting the CT.gov API.
Writes dashboard_data.json to data/processed/.
"""

import json
import os
import sys
import io
from datetime import datetime, timezone

# Windows UTF-8 safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to sys.path so src package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.harvester import parse_api_trial, fetch_trials_with_locations
from src.entity_resolver import resolve_sponsor, resolve_site
from src.network_builder import build_sponsor_site_network
from src.community_detector import detect_communities
from src.geographic_analyzer import analyze_geography
from src.factory_detector import detect_factories

# --- Configuration ---
CONDITION = "cardiovascular"
PHASE = "PHASE3"
MAX_TRIALS = 200

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CACHE_PATH = os.path.join(RAW_DIR, "trials_raw.json")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "dashboard_data.json")


def ensure_dirs():
    """Create data directories if they don't exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_or_fetch_trials():
    """Load trials from cache or fetch from CT.gov API.

    Returns:
        list of trial dicts
    """
    if os.path.exists(CACHE_PATH):
        print(f"[cache] Loading cached trials from {CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            trials = json.load(f)
        print(f"[cache] Loaded {len(trials)} cached trials")
        return trials

    print(f"[fetch] Fetching trials from CT.gov API: condition={CONDITION}, "
          f"phase={PHASE}, max={MAX_TRIALS}")
    trials = fetch_trials_with_locations(CONDITION, phase=PHASE, max_trials=MAX_TRIALS)
    print(f"[fetch] Retrieved {len(trials)} trials with locations")

    # Cache raw data
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)
    print(f"[cache] Saved raw trials to {CACHE_PATH}")

    return trials


def run_pipeline(trials):
    """Run the full analysis pipeline.

    Args:
        trials: list of trial dicts

    Returns:
        dict, the complete dashboard data payload
    """
    print(f"\n=== TrialAtlas Pipeline ===")
    print(f"Input: {len(trials)} trials")

    # Step 1: Entity resolution (already happens inside network_builder)
    print("[1/5] Building sponsor-site network...")
    graph = build_sponsor_site_network(trials)
    node_count = len(graph["nodes"])
    edge_count = len(graph["edges"])
    print(f"       {node_count} nodes, {edge_count} edges")

    # Step 2: Community detection
    print("[2/5] Detecting communities...")
    communities, modularity = detect_communities(graph, seed=42, return_modularity=True)
    num_communities = len(set(communities.values()))
    print(f"       {num_communities} communities (modularity={modularity:.4f})")

    # Step 3: Geographic analysis
    print("[3/5] Analyzing geography...")
    geography = analyze_geography(trials)
    print(f"       {geography['uniqueCountries']} countries, "
          f"Gini={geography['gini']:.3f}, "
          f"{len(geography['deserts'])} trial deserts")

    # Step 4: Factory detection
    print("[4/5] Detecting trial factories...")
    factories = detect_factories(trials, min_concurrent=5, min_sponsors=2)
    print(f"       {len(factories)} potential factory sites")

    # Step 5: Assemble dashboard data
    print("[5/5] Assembling dashboard data...")

    # Enrich network nodes with community IDs
    enriched_nodes = {}
    for name, meta in graph["nodes"].items():
        enriched = dict(meta)
        enriched["community"] = communities.get(name, -1)
        enriched_nodes[name] = enriched

    # Build sponsor profiles
    sponsors = {}
    for name, meta in enriched_nodes.items():
        if meta.get("type") == "sponsor":
            # Find sites connected to this sponsor
            sponsor_sites = []
            for edge in graph["edges"]:
                if edge["source"] == name:
                    site_name = edge["target"]
                    site_meta = enriched_nodes.get(site_name, {})
                    sponsor_sites.append({
                        "site": site_name,
                        "country": site_meta.get("country", "Unknown"),
                        "sharedTrials": edge["weight"],
                    })
            sponsors[name] = {
                "trialCount": meta.get("trialCount", 0),
                "conditions": meta.get("conditions", []),
                "sponsorClass": meta.get("sponsorClass", "OTHER"),
                "community": meta.get("community", -1),
                "sites": sorted(sponsor_sites, key=lambda s: -s["sharedTrials"]),
            }

    dashboard_data = {
        "meta": {
            "fetchDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "condition": CONDITION,
            "phase": PHASE,
            "trialCount": len(trials),
            "nodeCount": node_count,
            "edgeCount": edge_count,
            "communityCount": num_communities,
            "modularity": round(modularity, 4),
        },
        "network": {
            "nodes": enriched_nodes,
            "edges": graph["edges"],
        },
        "communities": communities,
        "geography": geography,
        "factories": factories,
        "sponsors": sponsors,
    }

    return dashboard_data


def main():
    """Entry point: fetch/cache -> pipeline -> write output."""
    ensure_dirs()
    trials = load_or_fetch_trials()

    if not trials:
        print("[error] No trials found. Cannot run pipeline.")
        sys.exit(1)

    dashboard_data = run_pipeline(trials)

    # Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    print(f"\n[done] Dashboard data written to {OUTPUT_PATH}")
    print(f"       {os.path.getsize(OUTPUT_PATH):,} bytes")


if __name__ == "__main__":
    main()
