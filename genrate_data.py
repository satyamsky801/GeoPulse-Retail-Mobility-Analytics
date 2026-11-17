"""GeoPulse - synthetic data generator. Run first: python generate_data.py
Creates zones.csv, stores.csv, calendar_weather.csv, footfall_hourly.csv, sales_daily.csv, device_visits_sample.csv (seeded, so output is identical every run)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

# ---- Zones (Bengaluru localities, approx centroids) ----
zones = pd.DataFrame([
 ("Z01","Koramangala",12.9352,77.6245,95000,"High"),
 ("Z02","Indiranagar",12.9784,77.6408,80000,"High"),
 ("Z03","Whitefield",12.9698,77.7500,150000,"High"),
 ("Z04","Electronic City",12.8456,77.6603,120000,"Medium"),
 ("Z05","Jayanagar",12.9250,77.5938,110000,"Medium"),
 ("Z06","Malleshwaram",13.0035,77.5710,90000,"Medium"),
 ("Z07","HSR Layout",12.9116,77.6389,100000,"High"),
 ("Z08","Yelahanka",13.1007,77.5963,130000,"Medium"),
 ("Z09","Marathahalli",12.9569,77.7011,140000,"Medium"),
 ("Z10","Banashankari",12.9255,77.5468,125000,"Low"),
], columns=["zone_id","zone_name","lat","lon","population","income_level"])
zones.to_csv("zones.csv", index=False)

# ---- Stores ----
cats = ["Grocery","Fashion","Electronics","Cafe","Pharmacy"]
cat_base = {"Grocery":90,"Fashion":55,"Electronics":30,"Cafe":70,"Pharmacy":45}
cat_ticket = {"Grocery":650,"Fashion":1800,"Electronics":9500,"Cafe":380,"Pharmacy":420}
rows=[]; sid=1
for _,z in zones.iterrows():
    for _ in range(3):
        c = cats[(sid-1)%5] if sid<=30 else rng.choice(cats)
        rows.append((f"S{sid:02d}", f"{z.zone_name} {c} #{sid}", c, z.zone_id, z.zone_name,
            round(z.lat+rng.normal(0,0.006),6), round(z.lon+rng.normal(0,0.006),6),
            int(rng.integers(600,4500)), f"20{rng.integers(15,25)}-{rng.integers(1,13):02d}-01"))
        sid+=1
stores = pd.DataFrame(rows, columns=["store_id","store_name","category","zone_id","zone_name","lat","lon","area_sqft","open_date"])
stores.to_csv("stores.csv", index=False)
store_factor = dict(zip(stores.store_id, rng.uniform(0.6,1.5,len(stores))))
zone_pop = dict(zip(zones.zone_id, zones.population))
inc_mult = {"High":1.2,"Medium":1.0,"Low":0.8}
zone_inc = dict(zip(zones.zone_id, zones.income_level.map(inc_mult)))

# ---- Calendar ----
dates = pd.date_range("2026-08-01","2026-10-31")
holidays = {"2026-08-15":"Independence Day","2026-09-14":"Ganesh Chaturthi","2026-10-02":"Gandhi Jayanti","2026-10-20":"Dussehra"}
cal = pd.DataFrame({"date":dates})
cal["is_weekend"]=(cal.date.dt.dayofweek>=5).astype(int)
cal["holiday_name"]=cal.date.dt.strftime("%Y-%m-%d").map(holidays).fillna("")
cal["is_holiday"]=(cal.holiday_name!="").astype(int)
# monsoon-ish weather
p_rain = np.where(cal.date.dt.month==10,0.25,0.45)
cal["weather"]=np.where(rng.random(len(cal))<p_rain,"Rain",np.where(rng.random(len(cal))<0.3,"Cloudy","Clear"))
cal["temp_c"]=np.round(rng.normal(26,2.2,len(cal)),1)
cal.to_csv("calendar_weather.csv", index=False)

# ---- Hourly footfall ----
hours = np.arange(8,22)
profiles = {
 "Grocery":   [.3,.5,.6,.5,.5,.6,.6,.7,.9,1.2,1.4,1.3,.9,.5],
 "Fashion":   [.1,.2,.4,.6,.8,1.0,1.1,1.2,1.3,1.5,1.4,1.2,.8,.4],
 "Electronics":[.1,.2,.4,.6,.8,1.0,1.0,1.0,1.2,1.4,1.3,1.1,.7,.3],
 "Cafe":      [1.2,1.4,1.0,.7,.9,1.3,1.0,.7,.8,1.0,1.2,1.1,.8,.5],
 "Pharmacy":  [.8,1.0,.9,.7,.6,.6,.6,.7,.9,1.2,1.2,1.0,.7,.4],
}
recs=[]
cal_i = cal.set_index("date")
for _,s in stores.iterrows():
    prof=np.array(profiles[s.category]); prof=prof/prof.mean()
    base = cat_base[s.category]*store_factor[s.store_id]*zone_inc[s.zone_id]*(s.area_sqft/2000)**0.4
    for d in dates:
        c=cal_i.loc[d]
        m=1.0
        if c.is_weekend: m*=1.35 if s.category in("Fashion","Cafe","Electronics") else 1.15
        if c.is_holiday: m*=1.5 if s.category in("Fashion","Electronics") else 1.2
        if c.weather=="Rain": m*=0.82 if s.category!="Cafe" else 0.95
        m*= 1+0.0012*(d-dates[0]).days  # slight growth
        lam=base*prof*m
        ff=rng.poisson(np.clip(lam,0.5,None))
        dwell=np.clip(rng.normal({"Grocery":18,"Fashion":42,"Electronics":35,"Cafe":28,"Pharmacy":9}[s.category],4,size=len(hours)),3,None)
        rep=rng.binomial(ff,np.clip(rng.normal(0.28,0.04),0.1,0.5))
        for h,f,dw,r in zip(hours,ff,dwell,rep):
            recs.append((s.store_id,d.date(),int(h),int(f),round(float(dw),1),int(r)))
ff_df=pd.DataFrame(recs,columns=["store_id","date","hour","footfall","avg_dwell_min","repeat_visitors"])
ff_df.to_csv("footfall_hourly.csv",index=False)

# ---- Daily sales ----
daily=ff_df.groupby(["store_id","date"],as_index=False).footfall.sum()
daily=daily.merge(stores[["store_id","category"]],on="store_id")
conv={"Grocery":.55,"Fashion":.22,"Electronics":.12,"Cafe":.7,"Pharmacy":.65}
daily["transactions"]=[rng.binomial(f,np.clip(conv[c]*rng.normal(1,.08),.02,.95)) for f,c in zip(daily.footfall,daily.category)]
daily["revenue_inr"]=[int(t*cat_ticket[c]*rng.lognormal(0,.15)) for t,c in zip(daily.transactions,daily.category)]
daily[["store_id","date","transactions","revenue_inr"]].to_csv("sales_daily.csv",index=False)

# ---- Device-level visit sample (for catchment/spatial) ----
n=40000
sample=stores.sample(n,replace=True,random_state=1,weights=[store_factor[s]*zone_pop[z]/1e5 for s,z in zip(stores.store_id,stores.zone_id)]).reset_index(drop=True)
zr=zones.set_index("zone_id")
zprob=zones.population/zones.population.sum()
home_z=[]
for _,r in sample.iterrows():
    if rng.random()<0.6: home_z.append(r.zone_id)
    else: home_z.append(rng.choice(zones.zone_id,p=zprob))
sample["home_zone"]=home_z
sample["home_lat"]=[round(zr.loc[z,"lat"]+rng.normal(0,0.008),6) for z in home_z]
sample["home_lon"]=[round(zr.loc[z,"lon"]+rng.normal(0,0.008),6) for z in home_z]
def hav(a,b,c,d):
    R=6371;p=np.pi/180
    x=np.sin((c-a)*p/2)**2+np.cos(a*p)*np.cos(c*p)*np.sin((d-b)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(x))
sample["distance_km"]=np.round(hav(sample.lat,sample.lon,sample.home_lat,sample.home_lon),2)
day=rng.choice(dates,n); hr=rng.choice(hours,n)
sample["visit_ts"]=pd.to_datetime(day)+pd.to_timedelta(hr,unit="h")+pd.to_timedelta(rng.integers(0,60,n),unit="m")
sample["dwell_min"]=np.round(np.clip(rng.gamma(2.2,10,n),2,180),1)
sample["device_id"]=["D"+str(x) for x in rng.integers(10000,22000,n)]
sample["transport_mode"]=rng.choice(["Walk","Two-wheeler","Car","Bus/Metro","Auto/Cab"],n,p=[.25,.35,.15,.15,.10])
vis=sample[["device_id","store_id","visit_ts","dwell_min","home_zone","home_lat","home_lon","distance_km","transport_mode"]].copy()
# inject dirtiness for cleaning practice
idx=rng.choice(n,int(n*.02),replace=False); vis.loc[idx,"dwell_min"]=np.nan
idx=rng.choice(n,int(n*.005),replace=False); vis.loc[idx,"home_lat"]=0.0
idx=rng.choice(n,int(n*.003),replace=False); vis.loc[idx,"dwell_min"]=-5
vis=pd.concat([vis,vis.sample(int(n*.01),random_state=3)]).sample(frac=1,random_state=5)
vis.to_csv("device_visits_sample.csv",index=False)
for f in ["zones","stores","calendar_weather","footfall_hourly","sales_daily","device_visits_sample"]:
    print(f, len(pd.read_csv(f+".csv")))