#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET


BASE_URL = "https://reports-public.ieso.ca/public/VGForecastSummary/"
INDEX_URL = BASE_URL
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = OUTPUT_DIR / "VGForecastSummary_latest_hourly_30d.csv"
OUTPUT_METADATA = OUTPUT_DIR / "VGForecastSummary_latest_hourly_30d_metadata.json"
WINDOW_DAYS = 30
NS = {"ns": "http://www.ieso.ca/schema"}

FILENAME_RE = re.compile(
    r"^PUB_VGForecastSummary(?:_(?P<day>\d{8})(?:_v(?P<version>\d+))?)?\.xml$"
)


@dataclass(frozen=True)
class Snapshot:
    source_file: str
    created_at: datetime
    forecast_timestamp: datetime
    organization_rows: list[dict[str, str | int | float]]
    version_rank: int


def fetch_text(url: str) -> str:
    try:
        with urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8", "ignore")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def list_candidate_files(window_start: date, window_end: date) -> list[str]:
    html = fetch_text(INDEX_URL)
    names = re.findall(r'href="(PUB_VGForecastSummary[^"/]*\.xml)"', html)
    selected: set[str] = set()

    for name in names:
        match = FILENAME_RE.match(name)
        if not match:
            continue

        day_token = match.group("day")
        if day_token is None:
            selected.add(name)
            continue

        publication_day = datetime.strptime(day_token, "%Y%m%d").date()
        if window_start <= publication_day <= window_end:
            selected.add(name)

    return sorted(selected)


def parse_version_rank(source_file: str) -> int:
    if source_file == "PUB_VGForecastSummary.xml":
        return 1_000_000

    match = FILENAME_RE.match(source_file)
    if not match:
        return -1

    version = match.group("version")
    if version is not None:
        return int(version)

    if match.group("day") is not None:
        return 10_000

    return -1


def parse_snapshot(source_file: str) -> Snapshot:
    xml_text = fetch_text(f"{BASE_URL}{source_file}")
    root = ET.fromstring(xml_text)

    created_at_text = root.findtext("./ns:DocHeader/ns:CreatedAt", namespaces=NS)
    forecast_ts_text = root.findtext(
        "./ns:DocBody/ns:ForecastTimeStamp", namespaces=NS
    )
    if not created_at_text or not forecast_ts_text:
        raise RuntimeError(f"missing timestamps in {source_file}")

    created_at = datetime.fromisoformat(created_at_text)
    forecast_timestamp = datetime.fromisoformat(forecast_ts_text)
    rows: list[dict[str, str | int | float]] = []

    for organization_data in root.findall("./ns:DocBody/ns:OrganizationData", NS):
        organization_type = organization_data.findtext(
            "./ns:OrganizationType", default="", namespaces=NS
        )
        for fuel_data in organization_data.findall("./ns:FuelData", NS):
            fuel_type = fuel_data.findtext("./ns:FuelType", default="", namespaces=NS)
            for resource_data in fuel_data.findall("./ns:ResourceData", NS):
                zone_name = resource_data.findtext(
                    "./ns:ZoneName", default="", namespaces=NS
                )
                for energy_forecast in resource_data.findall("./ns:EnergyForecast", NS):
                    forecast_date = energy_forecast.findtext(
                        "./ns:ForecastDate", default="", namespaces=NS
                    )
                    for interval in energy_forecast.findall("./ns:ForecastInterval", NS):
                        forecast_hour = interval.findtext(
                            "./ns:ForecastHour", default="", namespaces=NS
                        )
                        mw_output = interval.findtext(
                            "./ns:MWOutput", default="", namespaces=NS
                        )
                        if not forecast_date or not forecast_hour or mw_output == "":
                            continue

                        rows.append(
                            {
                                "organization_type": organization_type,
                                "fuel_type": fuel_type,
                                "zone_name": zone_name,
                                "forecast_date": forecast_date,
                                "forecast_hour": int(forecast_hour),
                                "mw_output": float(mw_output),
                            }
                        )

    return Snapshot(
        source_file=source_file,
        created_at=created_at,
        forecast_timestamp=forecast_timestamp,
        organization_rows=rows,
        version_rank=parse_version_rank(source_file),
    )


def load_snapshots(source_files: Iterable[str]) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {
            executor.submit(parse_snapshot, source_file): source_file
            for source_file in source_files
        }
        for future in as_completed(future_map):
            snapshots.append(future.result())
    return snapshots


def keep_latest_per_publication_hour(
    snapshots: Iterable[Snapshot], window_start: date, window_end: date
) -> list[Snapshot]:
    latest: dict[tuple[date, int], Snapshot] = {}

    for snapshot in snapshots:
        publication_day = snapshot.created_at.date()
        if not (window_start <= publication_day <= window_end):
            continue

        key = (publication_day, snapshot.created_at.hour)
        current = latest.get(key)
        if current is None:
            latest[key] = snapshot
            continue

        snapshot_rank = (
            snapshot.created_at,
            snapshot.forecast_timestamp,
            snapshot.version_rank,
            snapshot.source_file,
        )
        current_rank = (
            current.created_at,
            current.forecast_timestamp,
            current.version_rank,
            current.source_file,
        )
        if snapshot_rank > current_rank:
            latest[key] = snapshot

    return sorted(latest.values(), key=lambda s: (s.created_at, s.source_file))


def write_outputs(snapshots: list[Snapshot], window_start: date, window_end: date) -> None:
    fieldnames = [
        "source_file",
        "created_at",
        "forecast_timestamp",
        "publication_date",
        "publication_hour",
        "organization_type",
        "fuel_type",
        "zone_name",
        "forecast_date",
        "forecast_hour",
        "mw_output",
    ]

    row_count = 0
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for snapshot in snapshots:
            publication_date = snapshot.created_at.date().isoformat()
            publication_hour = snapshot.created_at.hour
            for row in sorted(
                snapshot.organization_rows,
                key=lambda item: (
                    str(item["organization_type"]),
                    str(item["fuel_type"]),
                    str(item["zone_name"]),
                    str(item["forecast_date"]),
                    int(item["forecast_hour"]),
                ),
            ):
                writer.writerow(
                    {
                        "source_file": snapshot.source_file,
                        "created_at": snapshot.created_at.isoformat(timespec="seconds"),
                        "forecast_timestamp": snapshot.forecast_timestamp.isoformat(
                            timespec="seconds"
                        ),
                        "publication_date": publication_date,
                        "publication_hour": publication_hour,
                        **row,
                    }
                )
                row_count += 1

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_index": INDEX_URL,
        "window_days": WINDOW_DAYS,
        "publication_window_start": window_start.isoformat(),
        "publication_window_end": window_end.isoformat(),
        "snapshots_kept": len(snapshots),
        "rows_written": row_count,
        "output_csv": OUTPUT_CSV.name,
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    window_end = date.today()
    window_start = window_end - timedelta(days=WINDOW_DAYS - 1)

    candidate_files = list_candidate_files(window_start, window_end)
    snapshots = load_snapshots(candidate_files)
    latest_snapshots = keep_latest_per_publication_hour(
        snapshots, window_start, window_end
    )
    write_outputs(latest_snapshots, window_start, window_end)

    print(
        json.dumps(
            {
                "candidate_files": len(candidate_files),
                "snapshots_kept": len(latest_snapshots),
                "output_csv": str(OUTPUT_CSV),
                "output_metadata": str(OUTPUT_METADATA),
            }
        )
    )


if __name__ == "__main__":
    main()
