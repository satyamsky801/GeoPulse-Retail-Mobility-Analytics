"""GeoPulse - Exploratory Data Analysis
Run after clean_data.py:  python eda.py
Saves charts to ./charts and KPI tables to ./kpi_summary.csv
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

stores = pd.read_csv("stores.csv")
ff = pd.read_csv("footfall_hourly.csv", parse_dates=["date"])
sales = pd.read_csv("sales_daily.csv", parse_dates=["date"])
cal = pd.read_csv("calendar_weather.csv", parse_dates=["date"])
cal["holiday_name"] = cal["holiday_name"].fillna("None")
visits = pd.read_csv("device_visits_clean.csv", parse_dates=["visit_ts"])

ff = ff.merge(stores[["store_id","category","zone_name","area_sqft"]], on="store_id") \
       .merge(cal[["date","is_weekend","is_holiday","weather"]], on="date")
ff["day_type"] = np.where(ff.is_weekend==1, "Weekend", "Weekday")

# 1. Footfall by hour and category
plt.figure(figsize=(10,5))
sns.lineplot(data=ff.groupby(["hour","category"],as_index=False).footfall.mean(),
             x="hour", y="footfall", hue="category", marker="o")
plt.title("Average hourly footfall by store category"); plt.ylabel("Avg footfall per store-hour")
plt.tight_layout(); plt.savefig("charts/01_hourly_by_category.png", dpi=150); plt.close()

# 2. Heatmap: weekday x hour
ff["dow"] = ff.date.dt.day_name()
order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
hm = ff.pivot_table(index="dow", columns="hour", values="footfall", aggfunc="mean").reindex(order)
plt.figure(figsize=(11,4.5)); sns.heatmap(hm, cmap="YlOrRd", annot=False)
plt.title("Average footfall: day of week x hour"); plt.tight_layout()
plt.savefig("charts/02_heatmap_day_hour.png", dpi=150); plt.close()

# 3. Weekday vs weekend, rain vs not
fig, ax = plt.subplots(1,2, figsize=(11,4))
sns.barplot(data=ff.groupby(["category","day_type"],as_index=False).footfall.mean(),
            x="category", y="footfall", hue="day_type", ax=ax[0]); ax[0].set_title("Weekday vs weekend")
sns.barplot(data=ff.groupby(["category","weather"],as_index=False).footfall.mean(),
            x="category", y="footfall", hue="weather", ax=ax[1]); ax[1].set_title("Effect of weather")
plt.tight_layout(); plt.savefig("charts/03_weekend_weather.png", dpi=150); plt.close()

# 4. Daily trend
daily = ff.groupby("date", as_index=False).footfall.sum()
daily["ma7"] = daily.footfall.rolling(7).mean()
plt.figure(figsize=(11,4)); plt.plot(daily.date, daily.footfall, alpha=.4, label="Daily")
plt.plot(daily.date, daily.ma7, lw=2, label="7-day avg"); plt.legend()
plt.title("Total daily footfall (all stores)"); plt.tight_layout()
plt.savefig("charts/04_daily_trend.png", dpi=150); plt.close()

# 5. Store KPIs: footfall, revenue, conversion, footfall per 1000 sq ft
store_ff = ff.groupby("store_id").agg(total_footfall=("footfall","sum"),
                                       avg_dwell_min=("avg_dwell_min","mean"),
                                       repeat=("repeat_visitors","sum")).reset_index()
store_sales = sales.groupby("store_id").agg(transactions=("transactions","sum"),
                                             revenue_inr=("revenue_inr","sum")).reset_index()
kpi = stores.merge(store_ff,on="store_id").merge(store_sales,on="store_id")
kpi["conversion_pct"] = (kpi.transactions/kpi.total_footfall*100).round(1)
kpi["repeat_rate_pct"] = (kpi.repeat/kpi.total_footfall*100).round(1)
kpi["footfall_per_1000sqft"] = (kpi.total_footfall/kpi.area_sqft*1000).round(0)
kpi["revenue_per_visitor"] = (kpi.revenue_inr/kpi.total_footfall).round(0)
kpi = kpi.drop(columns=["repeat"]).sort_values("total_footfall", ascending=False)
kpi.to_csv("kpi_summary.csv", index=False)

plt.figure(figsize=(8,5))
sns.scatterplot(data=kpi, x="total_footfall", y="revenue_inr", hue="category", s=90)
plt.title("Footfall vs revenue by store"); plt.tight_layout()
plt.savefig("charts/05_footfall_vs_revenue.png", dpi=150); plt.close()

# 6. Zone footfall
zone = ff.groupby("zone_name").footfall.sum().sort_values()
plt.figure(figsize=(8,5)); zone.plot.barh(color="#3b82f6"); plt.title("Total footfall by zone")
plt.tight_layout(); plt.savefig("charts/06_zone_footfall.png", dpi=150); plt.close()

# 7. Visits: dwell distribution, distance, transport
fig, ax = plt.subplots(1,3, figsize=(15,4))
sns.histplot(visits.dwell_min, bins=40, ax=ax[0]); ax[0].set_title("Dwell time (min)")
sns.histplot(visits.distance_km, bins=40, ax=ax[1]); ax[1].set_title("Home-to-store distance (km)")
visits.transport_mode.value_counts().plot.bar(ax=ax[2]); ax[2].set_title("Transport mode")
plt.tight_layout(); plt.savefig("charts/07_visit_profile.png", dpi=150); plt.close()

# Quick insights printed for your report
peak = ff.groupby("hour").footfall.mean().idxmax()
wk = ff.groupby("day_type").footfall.mean()
rain = ff.groupby("weather").footfall.mean()
print("Peak hour overall:", peak)
print("Weekend uplift: %.1f%%" % ((wk["Weekend"]/wk["Weekday"]-1)*100))
print("Rain vs Clear: %.1f%%" % ((rain["Rain"]/rain["Clear"]-1)*100))
print("Share of visits within 3 km: %.1f%%" % ((visits.distance_km<=3).mean()*100))
print("Median distance (km): %.2f" % visits.distance_km.median())
print("\nTop 5 stores by footfall:\n", kpi[["store_name","total_footfall","conversion_pct"]].head().to_string(index=False))
print("\nLowest revenue per visitor:\n", kpi.sort_values("revenue_per_visitor")[["store_name","revenue_per_visitor"]].head(3).to_string(index=False))