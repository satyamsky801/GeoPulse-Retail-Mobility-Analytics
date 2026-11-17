# GeoPulse: Hyper-Local Retail Mobility Analytics

Data analytics project (Infotact Solutions internship). Analyses footfall, visitor movement and
catchment areas for 30 retail stores across 10 Bengaluru zones.

> **Note:** The dataset is **synthetic** (1 Aug - 31 Oct 2026), generated because no dataset was provided.
> Findings describe patterns in simulated data, not real-world facts about Bengaluru.

## Project status
| Week | Focus | Status |
|---|---|---|
| 1 | Requirements, data sourcing, cleaning | Done |
| 2 | EDA, feature engineering, spatial analysis | Done |
| 3 | Forecasting model and dashboard | In progress |
| 4 | Report, PPT, demo | Planned |

## Data files
| File | Description |
|---|---|
| `zones.csv` | 10 localities: centroid, population, income level |
| `stores.csv` | 30 stores: category, zone, lat/long, area |
| `footfall_hourly.csv` | Store x date x hour footfall, dwell time, repeat visitors |
| `sales_daily.csv` | Daily transactions and revenue (INR) |
| `calendar_weather.csv` | Weekend, holiday, weather, temperature |
| `device_visits_sample.csv` | ~40k raw visits (contains deliberate data-quality issues) |

## How to run (in this order)
```bash
pip install -r requirements.txt
python clean_data.py            # cleans visits -> device_visits_clean.csv, cleaning_log.csv
python eda.py                   # 7 charts in charts/, kpi_summary.csv
python spatial.py               # store_catchment.csv, zone_opportunity.csv, store_map.html
python feature_engineering.py   # footfall_features.csv, store_features.csv, visits_features.csv
```

## Week 2 work
1. **Cleaning:** removed 400 duplicates and 200 invalid coordinates, fixed negative dwell times, imputed ~900 missing values with the store-category median (see `cleaning_log.csv`).
2. **EDA:** peak hour 6pm, weekends ~23% busier, rain ~14% lower footfall, each category has its own peak hours.
3. **Spatial analysis:** median visitor lives ~1.7 km away; DBSCAN found 5 customer hotspots; zone opportunity score ranks Yelahanka and Whitefield highest. Open `store_map.html` in a browser (needs internet for map tiles).
4. **Feature engineering:** peak-hour flag, time band, rain flag, distance features, lag features. See `feature_dictionary.csv`.

## Tech stack
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Leaflet.

## Author
Satyam Yadav - Data Associate L1 Intern, Infotact Solutions
