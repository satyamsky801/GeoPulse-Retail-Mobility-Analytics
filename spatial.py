"""GeoPulse - Spatial analysis
Run after clean_data.py and eda.py:  python spatial.py
Needs: pandas, numpy, scikit-learn.  The map is plain HTML (Leaflet loaded from a CDN), so it needs internet when you open it.
Outputs: store_catchment.csv, zone_opportunity.csv, home_hotspots.csv, store_map.html
"""
import json
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN

stores = pd.read_csv("stores.csv")
zones = pd.read_csv("zones.csv")
kpi = pd.read_csv("kpi_summary.csv")[["store_id","total_footfall","revenue_inr","conversion_pct"]]
visits = pd.read_csv("device_visits_clean.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0; p = np.pi/180
    a = np.sin((lat2-lat1)*p/2)**2 + np.cos(lat1*p)*np.cos(lat2*p)*np.sin((lon2-lon1)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# 1. Catchment area per store (from real visitor home -> store distances)
g = visits.groupby("store_id").distance_km
catch = pd.DataFrame({
    "median_km": g.median(),
    "p80_km": g.quantile(0.80),          # radius that covers 80% of visitors
    "share_within_2km": visits.assign(w=visits.distance_km <= 2).groupby("store_id").w.mean() * 100,
    "share_from_own_zone": visits.merge(stores[["store_id","zone_id"]], on="store_id")
                                 .assign(o=lambda d: d.home_zone == d.zone_id).groupby("store_id").o.mean() * 100,
}).round(2).reset_index()

# 2. Nearest same-category competitor (feature for the dashboard / model)
nearest = []
for _, s in stores.iterrows():
    rivals = stores[(stores.category == s.category) & (stores.store_id != s.store_id)]
    nearest.append(haversine(s.lat, s.lon, rivals.lat.values, rivals.lon.values).min())
stores["nearest_competitor_km"] = np.round(nearest, 2)

store_out = stores.merge(catch, on="store_id").merge(kpi, on="store_id")
store_out.to_csv("store_catchment.csv", index=False)

# 3. Customer hotspots: DBSCAN on visitor home locations (haversine metric)
coords = np.radians(visits[["home_lat","home_lon"]].values)
eps_km = 1.2
db = DBSCAN(eps=eps_km/6371.0, min_samples=300, metric="haversine", algorithm="ball_tree").fit(coords)
visits["hotspot"] = db.labels_
hs = (visits[visits.hotspot >= 0].groupby("hotspot")
      .agg(visits=("device_id","size"), center_lat=("home_lat","mean"), center_lon=("home_lon","mean"))
      .round(5).reset_index())
hs.to_csv("home_hotspots.csv", index=False)
print(f"DBSCAN (eps={eps_km} km): {len(hs)} hotspot(s), noise points: {(db.labels_==-1).sum()}")

# 4. Zone opportunity: where is demand high but supply low?
zs = stores.groupby("zone_id").agg(n_stores=("store_id","count"), area_sqft=("area_sqft","sum")).reset_index()
foot = store_out.groupby("zone_id").total_footfall.sum().rename("zone_footfall").reset_index()
res_visits = visits.groupby("home_zone").size().rename("resident_visits").reset_index().rename(columns={"home_zone":"zone_id"})
# residents who shop outside their own zone ("leakage")
vz = visits.merge(stores[["store_id","zone_id"]].rename(columns={"zone_id":"store_zone"}), on="store_id")
leak = (vz.assign(out=vz.home_zone != vz.store_zone).groupby("home_zone").out.mean() * 100).rename("pct_shop_elsewhere").reset_index().rename(columns={"home_zone":"zone_id"})
z = zones.merge(zs, on="zone_id").merge(foot, on="zone_id").merge(res_visits, on="zone_id").merge(leak, on="zone_id")
z["residents_per_store"] = (z.population / z.n_stores).round(0)
z["sqft_per_1000_residents"] = (z.area_sqft / z.population * 1000).round(1)
inc = {"High":1.2,"Medium":1.0,"Low":0.8}
z["demand_index"] = z.population * z.income_level.map(inc)
def mm(s): return (s - s.min()) / (s.max() - s.min())
# higher score = more attractive for a new store
z["opportunity_score"] = (100 * (0.4*mm(z.demand_index/z.area_sqft) + 0.3*mm(z.pct_shop_elsewhere) + 0.3*mm(z.residents_per_store))).round(1)
z = z.sort_values("opportunity_score", ascending=False)
z.drop(columns=["demand_index"]).to_csv("zone_opportunity.csv", index=False)
print("\nTop zones for a new store:\n", z[["zone_name","population","n_stores","residents_per_store","pct_shop_elsewhere","opportunity_score"]].head(5).round(1).to_string(index=False))

# Category gaps: zone x category with no store within 3 km
gaps = []
for _, zr in zones.iterrows():
    for cat in stores.category.unique():
        s = stores[stores.category == cat]
        d = haversine(zr.lat, zr.lon, s.lat.values, s.lon.values).min()
        if d > 3: gaps.append((zr.zone_name, cat, round(d, 1)))
print("\nCategory gaps (no store of that category within 3 km of zone centre):")
print(pd.DataFrame(gaps, columns=["zone","category","nearest_km"]).to_string(index=False) if gaps else "none")

# 5. Interactive map (Leaflet, standalone HTML)
colors = {"Grocery":"#16a34a","Fashion":"#9333ea","Electronics":"#2563eb","Cafe":"#d97706","Pharmacy":"#dc2626"}
fmax = store_out.total_footfall.max()
st_js = [dict(name=r.store_name, cat=r.category, lat=r.lat, lon=r.lon, color=colors[r.category],
              rad=int(6 + 14*r.total_footfall/fmax), p80=float(r.p80_km), p50=float(r.median_km), foot=int(r.total_footfall),
              rev=int(r.revenue_inr), conv=float(r.conversion_pct), own=float(r.share_from_own_zone),
              comp=float(r.nearest_competitor_km)) for r in store_out.itertuples()]
zn_js = [dict(name=r.zone_name, lat=r.lat, lon=r.lon, score=float(r.opportunity_score), pop=int(r.population),
              n=int(r.n_stores), leak=float(r.pct_shop_elsewhere)) for r in z.itertuples()]
hs_js = [dict(lat=r.center_lat, lon=r.center_lon, v=int(r.visits)) for r in hs.itertuples()]

html = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>GeoPulse Store Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{height:100%%;margin:0;font-family:Arial,sans-serif}
.legend{background:#fff;padding:8px 10px;border-radius:6px;line-height:1.5;font-size:12px}</style></head>
<body><div id="map"></div><script>
const stores=%s, zones=%s, hot=%s;
const map=L.map('map').setView([12.97,77.62],11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap contributors'}).addTo(map);
const sL=L.layerGroup().addTo(map), cL=L.layerGroup(), zL=L.layerGroup().addTo(map), hL=L.layerGroup();
stores.forEach(s=>{
  L.circleMarker([s.lat,s.lon],{radius:s.rad,color:s.color,fillColor:s.color,fillOpacity:.7,weight:1})
   .bindPopup(`<b>${s.name}</b><br>Category: ${s.cat}<br>Footfall: ${s.foot.toLocaleString()}<br>Revenue: Rs ${s.rev.toLocaleString()}<br>Conversion: ${s.conv}%%<br>50%% of visitors within: ${s.p50} km, 80%% within: ${s.p80} km<br>Visitors from own zone: ${s.own}%%<br>Nearest competitor: ${s.comp} km`).addTo(sL);
  L.circle([s.lat,s.lon],{radius:s.p50*1000,color:s.color,weight:1,fillOpacity:.05}).addTo(cL);
});
zones.forEach(z=>{
  const c=z.score>=66?'#b91c1c':z.score>=33?'#f59e0b':'#6b7280';
  L.marker([z.lat,z.lon],{icon:L.divIcon({className:'',html:`<div style="background:${c};color:#fff;border-radius:4px;padding:2px 5px;font-size:11px;white-space:nowrap">${z.name}: ${z.score}</div>`})})
   .bindPopup(`<b>${z.name}</b><br>Opportunity score: ${z.score}<br>Population: ${z.pop.toLocaleString()}<br>Stores: ${z.n}<br>Residents shopping elsewhere: ${z.leak}%%`).addTo(zL);
});
hot.forEach(h=>L.circle([h.lat,h.lon],{radius:1200,color:'#000',dashArray:'6',fillOpacity:.08}).bindPopup(`Visitor hotspot: ${h.v.toLocaleString()} visits`).addTo(hL));
L.control.layers(null,{'Stores (size = footfall)':sL,'Core catchment (50%% of visitors)':cL,'Zone opportunity score':zL,'Visitor hotspots':hL}).addTo(map);
const lg=L.control({position:'bottomleft'});lg.onAdd=()=>{const d=L.DomUtil.create('div','legend');
d.innerHTML=Object.entries(%s).map(([k,v])=>`<span style="color:${v}">&#9679;</span> ${k}`).join('<br>');return d};lg.addTo(map);
</script></body></html>""" % (json.dumps(st_js), json.dumps(zn_js), json.dumps(hs_js), json.dumps(colors))
open("store_map.html", "w", encoding="utf-8").write(html)
print("\nSaved: store_catchment.csv, zone_opportunity.csv, home_hotspots.csv, store_map.html")
print("\nCatchment summary:\n", store_out.groupby("category")[["median_km","p80_km","share_within_2km"]].mean().round(2))