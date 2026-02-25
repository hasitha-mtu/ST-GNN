"""
download_data.py
----------------
Main entry point for the OPW bulk ZIP downloader.

Each station downloads as a single ZIP -> tsvalues.csv (full history).

Usage
-----
  python download_data.py                          # active_subset from config
  python download_data.py --subset lee_full
  python download_data.py --start 2021-01-01 --end 2024-12-31
  python download_data.py --skip-existing          # skip already-downloaded CSVs
  python download_data.py --dry-run                # print plan, no HTTP calls
  python download_data.py --no-process             # download only, skip processing
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloaders.rainfall_downloader   import RainfallDownloader
from src.data.downloaders.waterlevel_downloader import WaterLevelDownloader
from src.data.downloaders.discharge_downloader import DischargeDownloader
from src.data.processors.timeseries_processor  import TimeSeriesProcessor


def setup_logging(log_cfg: dict):
    log_dir = Path(log_cfg["log_file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    fmt   = "%(asctime)s  %(levelname)-8s  %(message)s"

    # Force UTF-8 on both handlers so Windows cp1252 terminals don't break
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.stream.reconfigure(encoding="utf-8", errors="replace") if hasattr(stream_handler.stream, "reconfigure") else None

    file_handler = logging.FileHandler(log_cfg["log_file"], encoding="utf-8")

    logging.basicConfig(level=level, format=fmt,
                        handlers=[stream_handler, file_handler])


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_subset(config: dict, name: str = None) -> tuple[str, dict]:
    name = name or config["active_subset"]
    subset = config["subsets"].get(name)
    if subset is None:
        raise ValueError(f"Unknown subset '{name}'. Available: {list(config['subsets'])}")
    return name, subset


def skip_existing(stations: list[dict], raw_dir: Path, prefix: str) -> list[dict]:
    keep = []
    for s in stations:
        p = raw_dir / f"{prefix}{s['ref']}.csv"
        if p.exists():
            logging.getLogger(__name__).info(
                "   Skip (exists): %s (%s)", s.get("name"), s["ref"]
            )
        else:
            keep.append(s)
    return keep


def main():
    ap = argparse.ArgumentParser(description="OPW hydrological bulk downloader")
    ap.add_argument("--config",        default="C:\\Users\AdikariAdikari\PycharmProjects\ST-GNN\config\config.yaml")
    ap.add_argument("--subset",        default=None)
    ap.add_argument("--start",         default=None, help="Analysis start YYYY-MM-DD")
    ap.add_argument("--end",           default=None, help="Analysis end YYYY-MM-DD")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--no-process",    action="store_true")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    setup_logging(config["logging"])
    logger = logging.getLogger(__name__)

    subset_name, subset = get_subset(config, args.subset)
    start_date = args.start or "2020-01-01"  # ZIPs contain full history; this window is for processing only
    end_date   = args.end   or "2025-12-31"

    wl_stations   = subset.get("water_level_stations", [])
    discharge_stations = subset.get("discharge_stations", [])
    rain_stations = subset.get("rainfall_stations",    [])

    logger.info("+==========================================================+")
    logger.info("|         OPW Bulk ZIP Downloader                         |")
    logger.info("+==========================================================+")
    logger.info("Subset       : %s -- %s", subset_name, subset.get("description", ""))
    logger.info("Analysis     : %s -> %s", start_date, end_date)
    logger.info("Water level  : %d station(s)", len(wl_stations))
    logger.info("Discharge  : %d station(s)", len(discharge_stations))
    logger.info("Rainfall     : %d station(s)", len(rain_stations))
    logger.info("Water level URL pattern  : %s", config["api"]["waterlevel_zip_url"].format(station_no="XXXXX"))
    logger.info("Discharge URL pattern  : %s", config["api"]["discharge_zip_url"].format(station_no="XXXXX"))

    if args.dry_run:
        logger.info("\nDRY RUN -- no HTTP calls will be made")

        logger.info("\nWater level stations:")
        for s in wl_stations:
            url = config["api"]["waterlevel_zip_url"].format(station_no=s["ref"])
            logger.info("  * %-28s  %s  ->  %s", s.get("name"), s["ref"], url)

        logger.info("\nDischarge stations:")
        for s in wl_stations:
            url = config["api"]["discharge_zip_url"].format(station_no=s["ref"])
            logger.info("  * %-28s  %s  ->  %s", s.get("name"), s["ref"], url)

        logger.info("\nRainfall stations:")
        for s in rain_stations:
            url = config["api"]["rainfall_zip_url"].format(station_no=s["ref"])
            logger.info("  * %-28s  %s  ->  %s", s.get("name"), s["ref"], url)
        return

    # -- Optionally skip already-downloaded stations ------------------------
    wl_dl   = wl_stations
    discharge_dl = discharge_stations
    rain_dl = rain_stations
    if args.skip_existing:
        wl_dl   = skip_existing(wl_stations,   Path(config["output"]["raw_water_level_dir"]), "wl_")
        discharge_dl = skip_existing(discharge_stations, Path(config["output"]["raw_discharge_dir"]), "discharge_")
        rain_dl = skip_existing(rain_stations, Path(config["output"]["raw_rainfall_dir"]),    "rain_")

    # -- Save station metadata ---------------------------------------------
    import pandas as pd
    meta_dir = Path(config["output"]["metadata_dir"])
    meta_dir.mkdir(parents=True, exist_ok=True)
    if wl_stations:
        pd.DataFrame(wl_stations).to_csv(meta_dir / "waterlevel_stations.csv", index=False)
    if discharge_stations:
        pd.DataFrame(discharge_stations).to_csv(meta_dir / "discharge_stations.csv", index=False)
    if rain_stations:
        pd.DataFrame(rain_stations).to_csv(meta_dir / "rainfall_stations.csv", index=False)

    summaries = {}
    t0 = datetime.now()

    # -- Download water level ----------------------------------------------
    if wl_dl:
        logger.info("\n---  Downloading WATER LEVEL  (%d stations)  -------------", len(wl_dl))
        with WaterLevelDownloader(config) as dl:
            summaries["water_level"] = dl.download(wl_dl)

    # -- Download water level ----------------------------------------------
    if discharge_dl:
        logger.info("\n---  Downloading DISCHARGE  (%d stations)  -------------", len(discharge_dl))
        with DischargeDownloader(config) as dl:
            summaries["discharge"] = dl.download(discharge_dl)

    # -- Download rainfall -------------------------------------------------
    if rain_dl:
        logger.info("\n---  Downloading RAINFALL  (%d stations)  ----------------", len(rain_dl))
        with RainfallDownloader(config) as dl:
            summaries["rainfall"] = dl.download(rain_dl)

    logger.info("\nDownload complete in %.1f s", (datetime.now() - t0).total_seconds())

    # -- Process ------------------------------------------------------------
    if not args.no_process:
        logger.info("\n---  Processing & Quality Report  -------------------------")
        processor = TimeSeriesProcessor(config)
        summaries["quality"] = processor.process(
            rainfall_stations  = rain_stations,
            waterlevel_stations= wl_stations,
            discharge_stations= discharge_stations,
            start_date=start_date,
            end_date=end_date,
        )

    # -- Save summary JSON -------------------------------------------------
    summary_path = meta_dir / "download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    logger.info("\nSummary -> %s", summary_path)

    # Exit 1 if any station failed
    failed = [
        ref for d in summaries.values() if isinstance(d, dict)
        for ref, info in d.items()
        if isinstance(info, dict) and info.get("status") == "failed"
    ]
    if failed:
        logger.warning("Failed stations: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
