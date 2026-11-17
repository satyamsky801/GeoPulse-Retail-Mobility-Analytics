"""GeoPulse - Feature engineering (Week 2)
Run after clean_data.py and spatial.py:  python feature_engineering.py
Outputs: footfall_features.csv, visits_features.csv, store_features.csv, feature_dictionary.csv
"""
import numpy as np, pandas as pd

stores = pd.read_csv("stores.csv")
zones = pd.read_csv("zones.csv")
ff = pd.read_csv("footfall_hourly.csv", parse_dates=["date"])
cal = pd.read_csv("calendar_weather.csv", parse_dates=["date"])
cal["holiday_name"] = cal["holiday_name"].fillna("None")
visits = pd.read_csv("device_visits_clean.csv", parse_dates=["visit_ts"])
catch = pd.read_csv("store_catchment.csv")[["store_id","median_km","p80_km","share_within_2km","share_from_own_zone"]]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0; p = np.pi/180
    a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

CITY_LAT, CITY_LON = 12.9758, 77.6068   # MG Road, used as city-centre reference
GRID = 0.01                              # ~1.1 km grid cells (H3 alternative: pip install h3)
def grid_id(lat, lon):
    return (np.floor(lat/GRID).astype(int).astype(str) + "_" + np.floor(lon/GRID).astype(int).astype(str))

# ---------- 1. Store-level spatial features ----------
st = stores.copy()
zc = zones.set_index("zone_id")
st["grid_id"] = grid_id(st.lat, st.lon)
st["dist_city_centre_km"] = haversine(CITY_LAT, CITY_LON, st.lat, st.lon).round(2)
st["dist_zone_centre_km"] = [round(haversine(r.lat, r.lon, zc.loc[r.zone_id,"lat"], zc.loc[r.zone_id,"lon"]), 2) for r in st.itertuples()]
st["zone_population"] = st.zone_id.map(zc.population)
st["zone_income_level"] = st.zone_id.map(zc.income_level)
st["store_age_years"] = (2026 - pd.to_datetime(st.open_date).dt.year)
nc, cnt_same, cnt_all = [], [], []
for r in st.itertuples():
    d = haversine(r.lat, r.lon, st.lat.values, st.lon.values)
    same = (st.category.values == r.category) & (st.store_id.values != r.store_id)
    nc.append(d[same].min()); cnt_same.append(int((d[same] <= 2).sum()))
    cnt_all.append(int(((d <= 2) & (st.store_id.values != r.store_id)).sum()))
st["nearest_competitor_km"] = np.round(nc, 2)
st["competitors_within_2km"] = cnt_same
st["all_stores_within_2km"] = cnt_all          # proxy for local retail density / clustering
st = st.merge(catch, on="store_id")
st.to_csv("store_features.csv", index=False)

# ---------- 2. Hourly footfall features ----------
f = ff.merge(cal[["date","is_weekend","is_holiday","holiday_name","weather","temp_c"]], on="date")
f = f.merge(st.drop(columns=["store_name","lat","lon","open_date"]), on="store_id")
f["day_of_week"] = f.date.dt.day_name()
f["dow_num"] = f.date.dt.dayofweek
f["week_of_year"] = f.date.dt.isocalendar().week.astype(int)
f["month"] = f.date.dt.month
f["day_type"] = np.where(f.is_holiday == 1, "Holiday", np.where(f.is_weekend == 1, "Weekend", "Weekday"))
f["time_band"] = pd.cut(f.hour, [7,11,15,19,22], labels=["Morning","Afternoon","Evening","Night"]).astype(str)
# peak flag: the 3 busiest hours of each category (computed on weekdays+weekends overall)
top3 = f.groupby(["category","hour"]).footfall.mean().reset_index()
top3 = top3.sort_values(["category","footfall"], ascending=[True,False]).groupby("category").head(3)
f = f.merge(top3[["category","hour"]].assign(is_peak_hour=1), on=["category","hour"], how="left")
f["is_peak_hour"] = f.is_peak_hour.fillna(0).astype(int)
f["is_rain"] = (f.weather == "Rain").astype(int)
f["footfall_per_1000sqft"] = (f.footfall / f.area_sqft * 1000).round(3)
f["repeat_share"] = (f.repeat_visitors / f.footfall.replace(0, np.nan)).round(3)
# lag / rolling features (useful for the forecasting model)
f = f.sort_values(["store_id","date","hour"]).reset_index(drop=True)
g = f.groupby(["store_id","hour"]).footfall
f["lag_1d"] = g.shift(1)                       # same hour, previous day
f["lag_7d"] = g.shift(7)                       # same hour, same weekday last week
f["roll7d_mean"] = g.transform(lambda s: s.shift(1).rolling(7).mean()).round(2)
f.to_csv("footfall_features.csv", index=False)

# ---------- 3. Visit-level features ----------
v = visits.merge(st[["store_id","zone_id","grid_id","category"]].rename(columns={"category":"store_category","grid_id":"store_grid_id","zone_id":"store_zone"}), on="store_id", how="left", suffixes=("","_dup"))
v["home_grid_id"] = grid_id(v.home_lat, v.home_lon)
v["same_zone_visit"] = (v.home_zone == v.store_zone).astype(int)
v["distance_band"] = pd.cut(v.distance_km, [-0.01,1,3,5,10,100], labels=["0-1 km","1-3 km","3-5 km","5-10 km","10+ km"]).astype(str)
v["dwell_band"] = pd.cut(v.dwell_min, [0,10,30,60,1000], labels=["Quick (<10)","Short (10-30)","Medium (30-60)","Long (60+)"]).astype(str)
v["time_band"] = pd.cut(v.hour, [7,11,15,19,22], labels=["Morning","Afternoon","Evening","Night"]).astype(str)
v.drop(columns=[c for c in v.columns if c.endswith("_dup")]).to_csv("visits_features.csv", index=False)

# ---------- 4. Feature dictionary (for documentation) ----------
dic = [
("grid_id / home_grid_id / store_grid_id","Spatial","0.01 deg lat/long cell (~1.1 km) used as a simple grid; H3 can replace it"),
("dist_city_centre_km","Spatial","Distance from the store to MG Road (city-centre reference)"),
("dist_zone_centre_km","Spatial","Distance from the store to its zone centroid"),
("nearest_competitor_km","Spatial","Distance to the nearest store of the same category"),
("competitors_within_2km","Spatial","Same-category stores within 2 km"),
("all_stores_within_2km","Spatial","All stores within 2 km (local retail density)"),
("median_km / p80_km","Catchment","Distance covering 50% / 80% of a store's visitors"),
("share_within_2km / share_from_own_zone","Catchment","Share of visitors living within 2 km / in the store's own zone"),
("day_of_week / dow_num / week_of_year / month","Temporal","Calendar fields from the date"),
("day_type","Temporal","Weekday / Weekend / Holiday"),
("time_band","Temporal","Morning (8-11), Afternoon (12-15), Evening (16-19), Night (20-21)"),
("is_peak_hour","Temporal","1 if the hour is among the 3 busiest hours for that store category"),
("is_rain / temp_c","Weather","Rain flag and temperature"),
("footfall_per_1000sqft","Store efficiency","Footfall normalised by store area"),
("repeat_share","Behaviour","Repeat visitors / footfall"),
("lag_1d / lag_7d / roll7d_mean","Forecast","Same-hour footfall 1 day ago, 7 days ago, and 7-day rolling mean"),
("same_zone_visit","Visit","1 if the visitor lives in the store's zone"),
("distance_band / dwell_band","Visit","Binned distance travelled and dwell time"),
("store_age_years / zone_population / zone_income_level","Context","Store and zone attributes"),
]
pd.DataFrame(dic, columns=["feature","group","description"]).to_csv("feature_dictionary.csv", index=False)

print("store_features.csv   ", st.shape)
print("footfall_features.csv", f.shape, "| nulls from lags:", int(f.lag_7d.isna().sum()))
print("visits_features.csv  ", v.shape)
print("\nPeak hours per category:\n", top3.groupby("category").hour.apply(lambda s: sorted(s.tolist())).to_string())
print("\nFootfall by day_type (mean):\n", f.groupby("day_type").footfall.mean().round(1).to_string())
print("\nCorrelation of footfall with numeric features:")
print(f[["footfall","is_peak_hour","is_weekend","is_holiday","is_rain","temp_c","competitors_within_2km","area_sqft","dist_city_centre_km","lag_7d","roll7d_mean"]].corr()["footfall"].round(2).drop("footfall").to_string())
