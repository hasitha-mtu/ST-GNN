"""
rainfall_downloader.py
----------------------
Downloads OPW rainfall dataset via bulk ZIP download.

URL pattern:
  https://waterlevel.ie/hydro-data/data/internet/stations/0/{station_no}/Precip/Rainfall_complete.zip

tsvalues.csv format (semicolon-delimited, 4 columns for rainfall):
  #Timestamp;Value;Quality Code;Aggregation Accuracy %
  2022-03-27T00:00:00.000Z;0.00;254;

Key notes:
  - 15-minute resolution natively
  - Values are cumulative mm per 15-min interval
  - Quality code 254 = good
  - 4th column (Aggregation Accuracy %) may be empty -- handled gracefully
"""

import io
import logging
from pathlib import Path

import pandas as pd

from .base_downloader import BaseDownloader, DownloadError

logger = logging.getLogger(__name__)


class RainfallDownloader(BaseDownloader):

    def __init__(self, config: dict):
        super().__init__(config)
        self.url_template = self.api_cfg["rainfall_zip_url"]
        self.zip_entry    = self.api_cfg["zip_entry_name"]
        self.out_dir      = Path(self.out_cfg["raw_rainfall_dir"])
        self.ensure_dir(self.out_dir)

    # -- Public interface ------------------------------------------------------

    def download(self, stations: list[dict]) -> dict:
        summary = {}
        for station in stations:
            ref  = station["ref"]
            name = station.get("name", ref)
            logger.info("-- Rainfall: %s (%s)", name, ref)

            url = self.url_template.format(station_no=ref)
            try:
                zip_bytes = self.download_zip(url)
                csv_text  = self.extract_tsvalues(zip_bytes, self.zip_entry)
                meta, df  = self._parse(csv_text, ref)
                df        = self._quality_filter(df)
                out_path  = self._save(df, ref)

                missing_pct = round(df["value"].isna().mean() * 100, 2)
                summary[ref] = {
                    "name":        name,
                    "status":      "ok",
                    "records":     len(df),
                    "start":       str(df.index.min()),
                    "end":         str(df.index.max()),
                    "missing_pct": missing_pct,
                    "total_mm":    round(df["value"].sum(skipna=True), 1),
                    "path":        str(out_path),
                    "meta":        meta,
                }
                logger.info(
                    "   [OK] %d records @ 15-min  |  missing %.1f%%  "
                    "|  total %.0f mm  |  %s -> %s",
                    len(df), missing_pct,
                    summary[ref]["total_mm"],
                    summary[ref]["start"][:10],
                    summary[ref]["end"][:10],
                )

            except DownloadError as exc:
                logger.error("   [!!] %s: %s", ref, exc)
                summary[ref] = {"name": name, "status": "failed", "error": str(exc)}
            except Exception as exc:
                logger.error("   [!!] Unexpected error for %s: %s", ref, exc, exc_info=True)
                summary[ref] = {"name": name, "status": "error", "error": str(exc)}

        return summary

    # -- Parsing ---------------------------------------------------------------

    def _parse(self, text: str, station_ref: str) -> tuple[dict, pd.DataFrame]:
        """
        Parse OPW tsvalues.csv for rainfall.

        Rainfall CSV has 4 columns (last may be empty):
          Timestamp ; Value ; Quality Code ; Aggregation Accuracy %
        """
        meta: dict = {}
        data_lines: list[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                content = line.lstrip("#")
                if ";" in content:
                    k, v = content.split(";", 1)
                    meta[k.strip()] = v.strip()
            else:
                data_lines.append(line)

        if not data_lines:
            raise DownloadError(f"No dataset rows found for rainfall station {station_ref}")

        raw = "\n".join(data_lines)

        # Use read_csv with flexible column count (4th col often missing/empty)
        df = pd.read_csv(
            io.StringIO(raw),
            sep=";",
            header=None,
            names=["timestamp", "value", "quality_code", "agg_accuracy"],
            usecols=[0, 1, 2],      # drop agg_accuracy column
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        df.index.name = "datetime"

        df["value"]        = pd.to_numeric(df["value"],        errors="coerce")
        df["quality_code"] = pd.to_numeric(df["quality_code"], errors="coerce").fillna(0).astype(int)
        df["quality_ok"]   = df["quality_code"] == self.quality_cfg["good_quality_code"]
        df["station_ref"]  = station_ref

        return meta, df

    # -- Quality filter --------------------------------------------------------

    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        lo, hi = self.quality_cfg["rainfall_range"]
        neg = df["value"] < lo
        if neg.any():
            logger.debug("   %d negative rainfall values -> NaN", int(neg.sum()))
            df.loc[neg, "value"] = float("nan")
        over = df["value"] > hi
        if over.any():
            logger.debug("   %d values > %.0f mm/15-min -> NaN", int(over.sum()), hi)
            df.loc[over, "value"] = float("nan")
        return df

    # -- Save ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame, station_ref: str) -> Path:
        out_path = self.out_dir / f"rain_{station_ref}.csv"
        df.to_csv(out_path)
        return out_path
