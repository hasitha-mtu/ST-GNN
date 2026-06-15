"""
download_sentinel1_validation.py
─────────────────────────────────────────────────────────────────────
Download Sentinel-1 IW GRD products for flood map validation over the
River Lee catchment from the Copernicus Data Space Ecosystem (CDSE).

Two download modes
───────────────────
--mode flood      Acquisitions within ±FLOOD_WINDOW_DAYS of each flood
                  event peak identified in the inundation area timeseries.
                  These are used as the 'sar-flood' input to validate_flood_maps.py.

--mode reference  Dry-period acquisitions from summer (June–August) of a
                  specified year.  These form the change-detection reference
                  baseline.  Already have: 2025-07-29 (relative orbit 23).

Key facts decoded from already-downloaded files
────────────────────────────────────────────────
  Relative orbit   : 23  (ASCENDING, 06:47:51 UTC overpass)
  Absolute orbits  : 060295 (2025-07-29) and 061870 (2025-11-14)
  Both images verified consistent orbit geometry — use orbit 23.

  The November 14 image is in the RECESSION phase (0.20 km² inundation).
  The flood PEAK (0.73 km²) occurred on 2025-11-11 evening.
  The previous orbit-23 pass before November 14 was approximately
  November 11 (61870 - 175×5 = 61870 - 875 = 60995 → ~ Nov 1).
  Check: orbit 23 revisit is 12 days for S1A alone — the nearest
  acquisition before Nov 14 on this orbit would be ~Nov 02 or Nov 14.
  If no Nov 11–12 orbit-23 pass exists, download orbit 96 (also Cork
  coverage, descending) which may have passed on Nov 11–12.

CDSE credentials
─────────────────
  $env:CDSE_EMAIL    = "your@email.com"
  $env:CDSE_PASSWORD = "yourpassword"
  Or pass via --email / --password arguments.

Usage
──────
  # Check what flood-period products exist (no download)
  python src/download_sentinel1_validation.py --mode flood --search-only

  # Download flood-period products for orbit 23 (same as existing images)
  python src/download_sentinel1_validation.py --mode flood --orbit 23

  # Download reference images for summer 2025
  python src/download_sentinel1_validation.py --mode reference --ref-year 2025

  # Search all orbits to find what was available on Nov 11–12
  python src/download_sentinel1_validation.py --mode flood --all-orbits --search-only

  # Download both in one run
  python src/download_sentinel1_validation.py --mode both --orbit 23
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────
# Paths and constants
# ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent.parent.parent.parent
VAL_DIR       = BASE_DIR / "dataset/validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV  = VAL_DIR / "s1_validation_manifest.csv"

# CDSE API endpoints
CDSE_TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
                  "/protocol/openid-connect/token")
CDSE_ODATA     = "https://catalogue.dataspace.copernicus.eu/odata/v1"

# Lee catchment bounding box (WGS84) — 5 km buffer beyond gauge network
LEE_BBOX = dict(W=-9.10, S=51.70, E=-8.00, N=52.10)

# Relative orbit 23 (ascending, 06:47 UTC) — verified from existing files:
#   s1a-iw-grd-vv-20251114... abs=061870 → rel=23
#   s1a-iw-grd-vv-20250729... abs=060295 → rel=23
# Use this orbit for all validation downloads to ensure consistent
# viewing geometry between flood and reference images.
DEFAULT_ORBIT = 74  # orbit 74 ascending — Nov 2025 peak flood

# ── Validation events (Nov 2025 Lee flood) ───────────────────────────
FLOOD_EVENTS_VALIDATION = [
    {"name": "lee_flood_nov2025_peak",      "peak_date": "2025-11-11",
     "peak_stage_m": 0.73, "orbit_hint": 74},
    {"name": "lee_flood_nov2025_secondary", "peak_date": "2025-11-13",
     "peak_stage_m": 0.44, "orbit_hint": 96},
]

# ── DFC-GNN training events ───────────────────────────────────────────
# Needed for: SAR wetness similarity features, pixel flood-probability
# labels, and calibration of the learned depth-scale τ_k per node.
# Peak stages from OPW annual maxima at 19114 Carrigrohane Bridge.
FLOOD_EVENTS_DFC = [
    # Oct 2023 — exceptional (stage 3.055 m), Cork city flooded
    {"name": "lee_flood_oct2023",  "peak_date": "2023-10-20",
     "peak_stage_m": 3.055, "orbit_hint": None},
    # Nov 2024 — significant (stage ~2.26 m visible in gaugings)
    {"name": "lee_flood_nov2024",  "peak_date": "2024-11-28",
     "peak_stage_m": 2.260, "orbit_hint": None},
    # Dec 2023 — post-exceptional recovery (stage ~2.1 m estimated)
    {"name": "lee_flood_dec2023",  "peak_date": "2023-12-27",
     "peak_stage_m": 2.100, "orbit_hint": None},
    # Mar 2022 — annual max 2022 (stage 2.018 m at Carrigrohane)
    {"name": "lee_flood_mar2022",  "peak_date": "2022-02-18",
     "peak_stage_m": 2.018, "orbit_hint": None},
    # Jan 2024 — Storm Henk / Atlantic low (stage ~1.9 m estimated)
    {"name": "lee_flood_jan2024",  "peak_date": "2024-01-08",
     "peak_stage_m": 1.900, "orbit_hint": None},
]

# Orbit hints: populated from --mode dfc --search-only output.
# Each value is the relative orbit of the best Sentinel-1 pass for that event.
# Locking the orbit here ensures reference images share the same viewing
# geometry as the flood image — mandatory for change-detection accuracy.
DFC_ORBIT_HINTS: dict = {
    "lee_flood_oct2023":  23,    # S1 orbit 23, 2023-10-20, 0d from peak
    "lee_flood_nov2024":  147,   # S1 orbit 147, 2024-11-27, 1d from peak
    "lee_flood_dec2023":  147,   # S1 orbit 147, 2023-12-27, 0d from peak
    "lee_flood_mar2022":  74,    # S1 orbit 74,  2022-02-18, 0d from peak
    "lee_flood_jan2024":  147,   # S1 orbit 147, 2024-01-08, 0d from peak
}

# Default flood list (used by --mode flood/both)
FLOOD_EVENTS      = FLOOD_EVENTS_VALIDATION
FLOOD_WINDOW_DAYS = 2   # ±2 days around each peak

# Reference: summer dry period (already have 2025-07-29, more is better)
REF_YEAR         = 2025
REF_START_MONTH  = 6    # June
REF_END_MONTH    = 8    # August

# Download settings
CHUNK_SIZE      = 8192
RETRY_N         = 3
RETRY_DELAY_S   = 15
TOKEN_REFRESH_S = 540   # 9 min — token expires at 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Authentication
# ─────────────────────────────────────────────────────────────────────

def get_token(email: str, password: str) -> str:
    data = {
        "client_id":  "cdse-public",
        "username":   email,
        "password":   password,
        "grant_type": "password",
    }
    r = requests.post(CDSE_TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    log.info("CDSE authentication OK")
    return r.json()["access_token"]


def resolve_credentials(args) -> tuple[str, str]:
    email    = args.email    or os.environ.get("CDSE_EMAIL",    "")
    password = args.password or os.environ.get("CDSE_PASSWORD", "")
    if not email:
        email    = input("CDSE email: ").strip()
    if not password:
        password = getpass.getpass("CDSE password: ")
    return email, password


# ─────────────────────────────────────────────────────────────────────
# Footprint
# ─────────────────────────────────────────────────────────────────────

def lee_footprint_wkt(buffer_deg: float = 0.0) -> str:
    """WKT polygon for the Lee catchment bounding box."""
    W = LEE_BBOX["W"] - buffer_deg
    S = LEE_BBOX["S"] - buffer_deg
    E = LEE_BBOX["E"] + buffer_deg
    N = LEE_BBOX["N"] + buffer_deg
    wkt = (f"POLYGON(({W:.4f} {S:.4f},{E:.4f} {S:.4f},"
           f"{E:.4f} {N:.4f},{W:.4f} {N:.4f},{W:.4f} {S:.4f}))")
    log.info("Lee catchment footprint: W=%.2f S=%.2f E=%.2f N=%.2f",
             W, S, E, N)
    return wkt


# ─────────────────────────────────────────────────────────────────────
# Product search
# ─────────────────────────────────────────────────────────────────────

def search_products(footprint: str, start: str, end: str,
                    orbit: int | None = None) -> list[dict]:
    """
    Search CDSE OData for S1 IW GRDH products intersecting the
    Lee catchment between start and end dates.
    """
    filter_parts = [
        "Collection/Name eq 'SENTINEL-1'",
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S')",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')",
        f"ContentDate/Start gt {start}T00:00:00.000Z",
        f"ContentDate/Start lt {end}T23:59:59.999Z",
    ]
    if orbit is not None:
        filter_parts.append(
            "Attributes/OData.CSC.IntegerAttribute/any("
            "att:att/Name eq 'relativeOrbitNumber' and "
            f"att/OData.CSC.IntegerAttribute/Value eq {orbit})"
        )

    odata_filter = " and ".join(filter_parts)
    products     = []
    skip         = 0

    while True:
        url = (f"{CDSE_ODATA}/Products"
               f"?$filter={odata_filter}"
               f"&$orderby=ContentDate/Start asc"
               f"&$top=100&$skip={skip}"
               f"&$expand=Attributes")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        batch = resp.json().get("value", [])
        if not batch:
            break

        for item in batch:
            attrs = {a["Name"]: a.get("Value")
                     for a in item.get("Attributes", [])}
            products.append({
                "product_id":    item["Id"],
                "name":          item["Name"],
                "date":          item["ContentDate"]["Start"][:10],
                "time_utc":      item["ContentDate"]["Start"][11:19],
                "size_mb":       round(item.get("ContentLength", 0) / 1e6, 1),
                "orbit_rel":     attrs.get("relativeOrbitNumber", ""),
                "orbit_abs":     attrs.get("absoluteOrbitNumber", ""),
                "pass":          attrs.get("orbitDirection", ""),
                "polarisation":  attrs.get("polarisationChannels", ""),
            })
        skip += 100
        if len(batch) < 100:
            break

    return products


def search_flood_products(footprint: str,
                           orbit: int | None) -> pd.DataFrame:
    """
    Search for acquisitions within ±FLOOD_WINDOW_DAYS of each flood peak.
    Also checks ±1 day (tighter window) specifically for the peak date
    to maximise chances of finding an image closest to peak inundation.
    """
    all_products = []

    for event in FLOOD_EVENTS:
        peak   = pd.Timestamp(event["peak_date"])
        start  = (peak - pd.Timedelta(days=FLOOD_WINDOW_DAYS)).strftime("%Y-%m-%d")
        end    = (peak + pd.Timedelta(days=FLOOD_WINDOW_DAYS)).strftime("%Y-%m-%d")

        log.info("Searching flood window: %s → %s  (event: %s)",
                 start, end, event["name"])
        batch = search_products(footprint, start, end, orbit)
        for p in batch:
            p["event_name"] = event["name"]
            p["peak_date"]  = event["peak_date"]
            dist_days = abs((pd.Timestamp(p["date"]) - peak).days)
            p["days_from_peak"] = dist_days
        all_products.extend(batch)

    if not all_products:
        return pd.DataFrame()

    df = (pd.DataFrame(all_products)
          .drop_duplicates("product_id")
          .sort_values(["peak_date", "days_from_peak"])
          .reset_index(drop=True))
    return df


def search_reference_products(footprint: str,
                               orbit: int | None,
                               year: int) -> pd.DataFrame:
    """
    Search for summer dry-period reference images.
    Summer (Jun–Aug) of the specified year.
    Excludes any dates where a flood event occurred.
    """
    start = f"{year}-0{REF_START_MONTH}-01"
    end   = f"{year}-0{REF_END_MONTH}-31"
    log.info("Searching reference images: %s → %s", start, end)

    products = search_products(footprint, start, end, orbit)
    if not products:
        return pd.DataFrame()

    df = pd.DataFrame(products).drop_duplicates("product_id")

    # Exclude dates within 5 days of any known flood event
    flood_dates = [pd.Timestamp(e["peak_date"]) for e in FLOOD_EVENTS]
    def is_flood_period(date_str):
        d = pd.Timestamp(date_str)
        return any(abs((d - fd).days) <= 5 for fd in flood_dates)

    df = df[~df["date"].apply(is_flood_period)].reset_index(drop=True)
    df["role"] = "reference"
    return df


# ─────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────

def download_product(product_id: str, name: str, size_mb: float,
                     token: str, out_dir: Path) -> Path | None:
    """
    Download a single SAFE product as a zip using the CDSE redirect pattern.
    Skips files that already exist (checks by name and size).
    """
    out_path = out_dir / f"{name}.zip"

    # Check if already exists with correct size
    if out_path.exists():
        existing_mb = out_path.stat().st_size / 1e6
        if abs(existing_mb - size_mb) < size_mb * 0.05:   # within 5%
            log.info("  Already exists (%.0f MB): %s", existing_mb, out_path.name)
            return out_path
        else:
            log.warning("  Partial download found (%.0f/%.0f MB) — re-downloading",
                        existing_mb, size_mb)
            out_path.unlink()

    valid_redirect_codes = (301, 302, 303, 307)

    for attempt in range(1, RETRY_N + 1):
        try:
            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {token}"})

            url = (f"https://catalogue.dataspace.copernicus.eu"
                   f"/odata/v1/Products({product_id})/$value")
            resp = session.get(url, allow_redirects=False, timeout=60)

            while resp.status_code in valid_redirect_codes:
                url  = resp.headers["Location"]
                resp = session.get(url, allow_redirects=False, timeout=60)

            resp = session.get(url, verify=False,
                               allow_redirects=True, stream=True, timeout=300)
            resp.raise_for_status()

            log.info("  Downloading %s  (%.0f MB) ...", name[:60], size_mb)
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)

            actual_mb = out_path.stat().st_size / 1e6
            log.info("  Saved → %s  (%.0f MB)", out_path.name, actual_mb)
            return out_path

        except Exception as exc:
            log.warning("  Attempt %d/%d failed: %s", attempt, RETRY_N, exc)
            if out_path.exists():
                out_path.unlink()
            if attempt < RETRY_N:
                time.sleep(RETRY_DELAY_S)

    log.error("  FAILED after %d attempts: %s", RETRY_N, name)
    return None


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────

def print_flood_summary(df: pd.DataFrame):
    """Print a concise table of flood-period search results."""
    log.info("")
    log.info("=== Flood-period products ===")
    log.info("  Total found    : %d", len(df))
    if df.empty:
        log.warning("  No products found — try --all-orbits to check other orbits")
        return

    log.info("  Date range     : %s → %s", df["date"].min(), df["date"].max())
    log.info("  Total size     : %.1f GB", df["size_mb"].sum() / 1000)
    log.info("  Orbits         : %s", sorted(df["orbit_rel"].unique().tolist()))
    log.info("")
    log.info("  Products by event and proximity to flood peak:")
    log.info("  %-12s  %-10s  %-6s  %-4s  %-12s  %s",
             "peak_date", "date", "days", "orb", "pass", "name")
    log.info("  " + "─"*72)
    for _, r in df.iterrows():
        flag = " ← PEAK" if r["days_from_peak"] == 0 else \
               " ← nearest" if r["days_from_peak"] == df[
                   df["peak_date"]==r["peak_date"]]["days_from_peak"].min() \
               else ""
        log.info("  %-12s  %-10s  %-6d  %-4s  %-12s  %s%s",
                 r["peak_date"], r["date"], r["days_from_peak"],
                 r["orbit_rel"], r.get("pass",""), r["name"][:35], flag)
    log.info("")

    # Key recommendation
    for event in FLOOD_EVENTS:
        sub = df[df["peak_date"] == event["peak_date"]]
        if not sub.empty:
            best = sub.nsmallest(1, "days_from_peak").iloc[0]
            log.info("  Best image for %s: %s  (%dd from peak)",
                     event["peak_date"], best["date"], best["days_from_peak"])
    log.info("")


def print_reference_summary(df: pd.DataFrame):
    """Print reference product search results."""
    log.info("")
    log.info("=== Reference (dry-period) products ===")
    log.info("  Total found  : %d", len(df))
    if df.empty:
        log.warning("  No reference products found")
        return
    log.info("  Date range   : %s → %s", df["date"].min(), df["date"].max())
    log.info("  Total size   : %.1f GB", df["size_mb"].sum() / 1000)
    already = VAL_DIR / "s1a-iw-grd-vv-20250729t064751-20250729t064816-060295-077e4c-001.tiff"
    if already.exists():
        log.info("  Already have : 2025-07-29 (no need to re-download)")
    log.info("")
    log.info("  Available reference dates:")
    for _, r in df.iterrows():
        flag = " (already downloaded)" if "20250729" in r["name"] else ""
        log.info("    %s  orbit=%s  %.0f MB%s",
                 r["date"], r["orbit_rel"], r["size_mb"], flag)
    log.info("")
    log.info("  Recommendation: use the multi-image median composite")
    log.info("  of 3–5 acquisitions for a more robust reference baseline.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def search_dfc_products(footprint: str) -> pd.DataFrame:
    """
    Search for DFC-GNN training flood events across ALL orbits.
    Returns one best product per event (closest to peak date).
    """
    all_products = []
    log.info("")
    log.info("=== DFC-GNN: searching %d training events (all orbits) ===",
             len(FLOOD_EVENTS_DFC))
    for event in FLOOD_EVENTS_DFC:
        peak  = pd.Timestamp(event["peak_date"])
        start = (peak - pd.Timedelta(days=FLOOD_WINDOW_DAYS)).strftime("%Y-%m-%d")
        end   = (peak + pd.Timedelta(days=FLOOD_WINDOW_DAYS)).strftime("%Y-%m-%d")
        hint  = DFC_ORBIT_HINTS.get(event["name"])
        log.info("  %s  %s ± %dd  orbit=%s  (peak stage %.2f m)",
                 event["name"], event["peak_date"],
                 FLOOD_WINDOW_DAYS, hint or "all", event["peak_stage_m"])
        batch = search_products(footprint, start, end, orbit=hint)
        for p in batch:
            p["event_name"]     = event["name"]
            p["peak_date"]      = event["peak_date"]
            p["peak_stage_m"]   = event["peak_stage_m"]
            p["days_from_peak"] = abs((pd.Timestamp(p["date"]) - peak).days)
            p["role"]           = "dfc_flood"
        all_products.extend(batch)
    if not all_products:
        return pd.DataFrame()
    return (pd.DataFrame(all_products)
              .drop_duplicates("product_id")
              .sort_values(["event_name", "days_from_peak"])
              .reset_index(drop=True))


def search_dfc_references(
    footprint: str,
    flood_df:  pd.DataFrame,
    ref_year:  int | None = None,
) -> pd.DataFrame:
    """
    For each DFC-GNN training event, search for summer reference images
    on the SAME orbit as the winning flood image.

    Orbit consistency is required so the viewing geometry and local
    incidence angle are identical between each flood/reference pair.
    Inconsistent orbits introduce systematic backscatter offsets on
    slopes that contaminate the change-detection flood signal.

    ref_year (optional)
    ────────────────────
    When None (default): the reference year is auto-detected per event.
      Rule: use the summer of the same calendar year as the flood if the
      flood occurs in September or later (autumn/winter events), otherwise
      use the summer of the prior calendar year.

      Examples with auto-detection:
        Oct 2023 flood  → summer 2023 reference  (Sep+ → same year)
        Nov 2024 flood  → summer 2024 reference  (Sep+ → same year)
        Mar 2022 flood  → summer 2021 reference  (pre-Sep → prior year)
        Jan 2024 flood  → summer 2023 reference  (pre-Sep → prior year)

    When set (e.g. ref_year=2024): ALL events use that year's summer as
    their reference, overriding the auto-detection. Useful when you want a
    consistent reference baseline across all training events, or when the
    auto-detected year has insufficient Sentinel-1 coverage.
    Pass via CLI: --dfc-ref-year 2024
    """
    if flood_df.empty:
        return pd.DataFrame()

    best_per_event = (flood_df.sort_values("days_from_peak")
                               .groupby("event_name").first().reset_index())
    all_ref    = []
    flood_dates = [pd.Timestamp(d) for d in flood_df["peak_date"].unique()]

    log.info("")
    log.info("=== DFC-GNN: searching orbit-matched reference images ===")
    if ref_year:
        log.info("  ref_year override: %d (all events use summer %d)",
                 ref_year, ref_year)
    else:
        log.info("  ref_year: auto-detected per event "
                 "(Sep+ floods → same year, pre-Sep floods → prior year)")

    for _, row in best_per_event.iterrows():
        event  = row["event_name"]
        orbit  = int(row["orbit_rel"])
        f_year = int(row["date"][:4])
        flood_month = int(row["date"][5:7])

        if ref_year:
            # Explicit override: always use the user-supplied year
            r_year = ref_year
        else:
            # Auto-detect: autumn/winter floods reference the same year's
            # summer; spring floods reference the prior year's summer
            r_year = f_year if flood_month >= 9 else f_year - 1

        r_start = f"{r_year}-06-01"
        r_end   = f"{r_year}-08-31"

        log.info("  %-30s  orbit=%d  ref: %s → %s",
                 event[:30], orbit, r_start, r_end)

        batch = search_products(footprint, r_start, r_end, orbit=orbit)
        for p in batch:
            d = pd.Timestamp(p["date"])
            # Exclude any date within 10 days of a known flood peak
            if all(abs((d - fd).days) > 10 for fd in flood_dates):
                p["event_name"] = event
                p["ref_year"]   = r_year
                p["role"]       = "dfc_reference"
                all_ref.append(p)

    if not all_ref:
        log.warning("No reference images found. Try --dfc-ref-year YYYY "
                    "to specify a year with known SAR coverage.")
        return pd.DataFrame()

    return (pd.DataFrame(all_ref)
              .drop_duplicates("product_id")
              .sort_values(["event_name", "date"])
              .reset_index(drop=True))


def print_dfc_summary(flood_df: pd.DataFrame, ref_df: pd.DataFrame):
    """Print a per-event summary of what will be downloaded for DFC-GNN."""
    log.info("")
    log.info("=== DFC-GNN download plan ===")
    log.info("  %-28s  %-12s  %-5s  %s", "Event", "Best date", "Orb", "Stage(m)")
    log.info("  " + "─" * 60)
    total_gb = 0.0
    for event in FLOOD_EVENTS_DFC:
        name  = event["name"]
        sub_f = flood_df[flood_df["event_name"] == name] if not flood_df.empty                 else pd.DataFrame()
        sub_r = ref_df[ref_df["event_name"]   == name] if not ref_df.empty                 else pd.DataFrame()
        if sub_f.empty:
            log.info("  %-28s  NOT FOUND", name[:28]); continue
        best = sub_f.nsmallest(1, "days_from_peak").iloc[0]
        gb   = (best["size_mb"] + sub_r["size_mb"].sum()) / 1000
        total_gb += gb
        log.info("  %-28s  %-12s  %-5s  %.3f m  (%dd from peak  %d refs  %.1f GB)",
                 name[:28], best["date"], best["orbit_rel"],
                 event["peak_stage_m"], best["days_from_peak"],
                 len(sub_r), gb)
    log.info("  " + "─" * 60)
    log.info("  Total download: %.1f GB", total_gb)
    log.info("")
    log.info("  After downloading, run build_sar_reference.py for each event")
    log.info("  to produce calibrated sigma0 GeoTIFFs. Extract node-level")
    log.info("  backscatter values at the 27 gauge pixels → SAR wetness")
    log.info("  feature matrix W[events × nodes] for DFC-GNN training.")
    log.info("")
    # Show how much will actually be downloaded with the ref cap
    if not flood_df.empty:
        max_r = 3  # default --max-dfc-ref
        flood_gb = (flood_df.sort_values("days_from_peak")
                                .groupby("event_name").head(1)["size_mb"]
                                .sum() / 1000)
        ref_gb   = (ref_df["size_mb"].sum() / 1000 if not ref_df.empty
                    else ref_df.groupby("event_name").head(max_r)["size_mb"]
                       .sum() / 1000 if not ref_df.empty else 0)
        log.info("  Estimated download (1 flood + %d refs per event): %.1f GB",
                 max_r, flood_gb + ref_gb)
        log.info("  Adjust with --max-dfc-ref N (default 3)")


def main():
    p = argparse.ArgumentParser(
        description="Download Sentinel-1 IW GRD for Lee catchment flood validation"
    )
    p.add_argument("--mode",        choices=["flood", "reference", "both", "dfc"],
                   default="flood",
                   help="flood=Nov2025 validation, reference=dry period, "
                        "both=validation+reference, dfc=DFC-GNN training events")
    p.add_argument("--orbit",       type=int, default=DEFAULT_ORBIT,
                   help=f"Relative orbit number (default: {DEFAULT_ORBIT} — "
                        f"same as existing downloaded files)")
    p.add_argument("--all-orbits",  action="store_true",
                   help="Search all orbits (useful for finding nearest-to-peak images)")
    p.add_argument("--ref-year",    type=int, default=REF_YEAR,
                   help=f"Year for reference images (default: {REF_YEAR})")
    p.add_argument("--search-only", action="store_true",
                   help="Search and report without downloading")
    p.add_argument("--email",       default="",
                   help="CDSE email (or set $env:CDSE_EMAIL)")
    p.add_argument("--password",    default="",
                   help="CDSE password (or set $env:CDSE_PASSWORD)")
    p.add_argument("--max-flood",   type=int, default=4,
                   help="Max flood products to download (default 4 — one per event ±1d)")
    p.add_argument("--max-ref",     type=int, default=5,
                   help="Max reference products to download (default 5)")
    p.add_argument("--max-dfc-ref", type=int, default=3,
                   help="Max reference images per DFC event (default 3)")
    p.add_argument("--dfc-search-only", action="store_true",
                   help="Search DFC events and show orbit table without downloading")
    args = p.parse_args()

    orbit     = None if args.all_orbits else args.orbit
    footprint = lee_footprint_wkt()

    # ── 1. Print what we already have ────────────────────────────────
    existing = list(VAL_DIR.glob("*.tiff")) + list(VAL_DIR.glob("*.zip"))
    if existing:
        log.info("")
        log.info("=== Already in dataset/validation/ ===")
        for f in sorted(existing):
            mb = f.stat().st_size / 1e6
            log.info("  %-75s  %.0f MB", f.name[:75], mb)
        log.info("")

    flood_df = ref_df = pd.DataFrame()

    # ── 2. Search flood products ──────────────────────────────────────
    if args.mode in ("flood", "both"):
        flood_df = search_flood_products(footprint, orbit)
        print_flood_summary(flood_df)

        if flood_df.empty and not args.all_orbits:
            log.warning("No flood products found on orbit %d.", args.orbit)
            log.warning("Try: --all-orbits to check if another orbit passed on Nov 11–12")

    # ── 3. Search reference products ─────────────────────────────────
    if args.mode in ("reference", "both"):
        ref_df = search_reference_products(footprint, orbit, args.ref_year)
        print_reference_summary(ref_df)

    # ── 3b. DFC-GNN mode ─────────────────────────────────────────────
    dfc_flood_df = dfc_ref_df = pd.DataFrame()
    if args.mode == "dfc":
        dfc_flood_df = search_dfc_products(footprint)
        if not dfc_flood_df.empty:
            ref_yr = getattr(args, "dfc_ref_year", None)  # None = auto-detect
            dfc_ref_df = search_dfc_references(
                footprint, dfc_flood_df, ref_year=ref_yr)
        print_dfc_summary(dfc_flood_df, dfc_ref_df)

    # ── 4. Combine and save manifest ──────────────────────────────────
    frames = [df for df in [flood_df, ref_df,
                             dfc_flood_df, dfc_ref_df] if not df.empty]
    if frames:
        manifest = pd.concat(frames, ignore_index=True)
        manifest.to_csv(MANIFEST_CSV, index=False)
        log.info("Manifest saved → %s  (%d products total)",
                 MANIFEST_CSV.name, len(manifest))

    if args.search_only:
        log.info("--search-only: no downloads.")
        if args.mode == "dfc":
            log.info("Update DFC_ORBIT_HINTS in the script with the orbits shown above,")
            log.info("then re-run without --search-only to begin downloading.")
            # Show per-event orbit recommendations
            log.info("")
            log.info("Suggested DFC_ORBIT_HINTS update:")
            if not dfc_flood_df.empty:
                best = (dfc_flood_df.sort_values("days_from_peak")
                                    .groupby("event_name").first().reset_index())
                for _, r in best.iterrows():
                    log.info("  \"%s\": %s,  # %s  %dd from peak",
                             r["event_name"], r["orbit_rel"],
                             r["date"], r["days_from_peak"])
        else:
            log.info("Inspect the results above, then re-run without --search-only.")
        return

    nothing = (flood_df.empty and ref_df.empty
               and dfc_flood_df.empty and dfc_ref_df.empty)
    if nothing:
        log.error("Nothing to download.")
        return

    # ── 5. Authenticate ───────────────────────────────────────────────
    email, password = resolve_credentials(args)
    token      = get_token(email, password)
    token_time = time.time()

    downloaded, failed = 0, 0

    def maybe_refresh():
        nonlocal token, token_time
        if time.time() - token_time > TOKEN_REFRESH_S:
            log.info("Refreshing token ...")
            token      = get_token(email, password)
            token_time = time.time()

    # ── 6. Download flood products ────────────────────────────────────
    if args.mode in ("flood", "both") and not flood_df.empty:
        # Prioritise: one image per event, closest to peak
        to_dl = (flood_df
                 .sort_values("days_from_peak")
                 .groupby("event_name").head(1)
                 .head(args.max_flood))

        log.info("")
        log.info("=== Downloading %d flood product(s) ===", len(to_dl))
        for _, row in to_dl.iterrows():
            maybe_refresh()
            log.info("  Event: %s  |  date: %s  |  days from peak: %d",
                     row["event_name"], row["date"], row["days_from_peak"])
            r = download_product(row["product_id"], row["name"],
                                 row["size_mb"], token, VAL_DIR)
            downloaded += r is not None
            failed     += r is None

    # ── 7. Download reference products ───────────────────────────────
    if args.mode in ("reference", "both") and not ref_df.empty:
        # Skip dates we already have
        already_dates = {f.name[20:28] for f in VAL_DIR.glob("*.tiff")}
        new_ref = ref_df[~ref_df["date"].str.replace("-","").isin(already_dates)]
        to_dl   = new_ref.head(args.max_ref)

        if to_dl.empty:
            log.info("All reference images already downloaded.")
        else:
            log.info("")
            log.info("=== Downloading %d reference product(s) ===", len(to_dl))
            for _, row in to_dl.iterrows():
                maybe_refresh()
                r = download_product(row["product_id"], row["name"],
                                     row["size_mb"], token, VAL_DIR)
                downloaded += r is not None
                failed     += r is None

    # ── 8. Download DFC-GNN flood products ───────────────────────────
    if args.mode == "dfc" and not dfc_flood_df.empty:
        dfc_to_dl = (dfc_flood_df
                     .sort_values("days_from_peak")
                     .groupby("event_name").head(1))
        log.info("")
        log.info("=== Downloading %d DFC flood image(s) ===", len(dfc_to_dl))
        for _, row in dfc_to_dl.iterrows():
            maybe_refresh()
            log.info("  %s  %s  orbit=%s  %.0f MB",
                     row["event_name"][:30], row["date"],
                     row["orbit_rel"], row["size_mb"])
            r = download_product(row["product_id"], row["name"],
                                 row["size_mb"], token, VAL_DIR)
            downloaded += r is not None
            failed     += r is None

    # ── 9. Download DFC-GNN reference images ─────────────────────────
    if args.mode == "dfc" and not dfc_ref_df.empty:
        max_ref = getattr(args, "max_dfc_ref", 3)
        already_dates = {f.stem[:25] for f in VAL_DIR.glob("*.SAFE.zip")}
        dfc_ref_to_dl = (dfc_ref_df
                         .groupby("event_name").head(max_ref))
        log.info("")
        log.info("=== Downloading %d DFC reference image(s) ===",
                 len(dfc_ref_to_dl))
        for _, row in dfc_ref_to_dl.iterrows():
            maybe_refresh()
            log.info("  %s ref  %s  orbit=%s  %.0f MB",
                     row["event_name"][:20], row["date"],
                     row.get("orbit_rel", "?"), row["size_mb"])
            r = download_product(row["product_id"], row["name"],
                                 row["size_mb"], token, VAL_DIR)
            downloaded += r is not None
            failed     += r is None

    # ── 10. Final summary ─────────────────────────────────────────────
    log.info("")
    log.info("=== Download complete ===")
    log.info("  Downloaded : %d", downloaded)
    log.info("  Failed     : %d", failed)
    log.info("  Location   : %s", VAL_DIR)
    log.info("")
    log.info("Next steps:")
    if args.mode == "dfc":
        log.info("  1. python src/build_sar_reference.py  (calibrate all events)")
        log.info("  2. Extract node-level sigma0 from each processed GeoTIFF")
        log.info("     → SAR wetness feature matrix for DFC-GNN training")
        log.info("  3. python src/train_model.py --model dfc_gnn")
    else:
        log.info("  1. python src/preprocess_sentinel1.py  (calibrate + geocode)")
        log.info("  2. python src/validate_flood_maps.py --mode sar --event-date YYYY-MM-DD")


if __name__ == "__main__":
    main()
