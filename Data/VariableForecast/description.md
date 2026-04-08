## Variable Generation Forecast Summary

Source: `https://reports-public.ieso.ca/public/VGForecastSummary/`

Artifacts in this folder:

- `scrape_vgforecastsummary.py`: downloads the public XML reports from IESO.
- `VGForecastSummary_latest_hourly_30d.csv`: flattened data for the most recent 30 publication days.
- `VGForecastSummary_latest_hourly_30d_metadata.json`: scrape metadata and row counts.

Selection rule:

- Keep one snapshot per publication day and publication hour, based on the XML `CreatedAt` timestamp.
- If multiple files land in the same day-hour bucket, keep the latest available version.

CSV columns:

- `source_file`: original XML filename from the index.
- `created_at`: report creation timestamp from `DocHeader/CreatedAt`.
- `forecast_timestamp`: forecast timestamp from `DocBody/ForecastTimeStamp`.
- `publication_date`: date portion of `created_at`.
- `publication_hour`: hour portion of `created_at`.
- `organization_type`
- `fuel_type`
- `zone_name`
- `forecast_date`
- `forecast_hour`
- `mw_output`
