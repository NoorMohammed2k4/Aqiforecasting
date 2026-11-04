import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
CSV_PATH      = 'openaq_location_3409478_measurments.csv'
HORIZON       = 1      # hours ahead to forecast (try 1, 3, 6, 12, 24)
TEST_FRAC     = 0.20
MAX_GAP_HOURS = 3      # fill gaps up to this many hours by interpolation
POLLUTANTS    = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co']

# Molecular weights for ppb/ppm -> ug/m3 conversion (25 C, 1 atm)
GAS_MW    = {'no2': 46.01, 'o3': 48.00, 'so2': 64.07, 'co': 28.01}
MOLAR_VOL = 24.45

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Parse timestamps; drop timezone but keep local wall-clock time
try:
    ts = pd.to_datetime(df['datetimeLocal'], errors='coerce')
    if getattr(ts.dt, 'tz', None) is not None:
        ts = ts.dt.tz_localize(None)
except Exception:
    ts = (pd.to_datetime(df['datetimeLocal'], errors='coerce', utc=True)
            .dt.tz_convert('Asia/Kolkata').dt.tz_localize(None))
df['datetimeLocal'] = ts
df = df.dropna(subset=['datetimeLocal']).copy()

# Normalise parameter names ("pm2.5" -> "pm25") and values
df['parameter'] = df['parameter'].astype(str).str.lower().str.replace('.', '', regex=False)
df['value'] = pd.to_numeric(df['value'], errors='coerce')

print("Parameters in file:", sorted(df['parameter'].unique()))
if 'unit' in df.columns:
    print("\nUnits per parameter:")
    print(df.groupby('parameter')['unit'].unique().to_string())
else:
    print("\nWARNING: no 'unit' column; assuming ug/m3 (CO in mg/m3).")

# ------------------------------------------------------------
# 2. UNIT CONVERSION (to CPCB units: ug/m3, CO in mg/m3)
# ------------------------------------------------------------
unit = df['unit'].astype(str).str.lower() if 'unit' in df.columns else pd.Series('', index=df.index)
is_ppb = unit.str.contains('ppb', na=False)
is_ppm = unit.str.contains('ppm', na=False)
is_ug  = unit.str.contains('µg|ug', na=False, regex=True)

for p, mw in GAS_MW.items():
    m = df['parameter'] == p
    f = mw / MOLAR_VOL                      # ug/m3 per ppb
    if p == 'co':                           # target unit: mg/m3
        df.loc[m & is_ppb, 'value'] *= f / 1000
        df.loc[m & is_ppm, 'value'] *= f
        df.loc[m & is_ug,  'value'] /= 1000
    else:                                   # target unit: ug/m3
        df.loc[m & is_ppb, 'value'] *= f
        df.loc[m & is_ppm, 'value'] *= f * 1000

# Negative readings are sensor noise
df.loc[df['value'] < 0, 'value'] = np.nan

# ------------------------------------------------------------
# 3. PIVOT LONG -> WIDE, THEN REGULAR HOURLY GRID
# ------------------------------------------------------------
wide = df.pivot_table(index='datetimeLocal', columns='parameter',
                      values='value', aggfunc='mean')
wide.columns.name = None

# Resample so that 1 row == exactly 1 hour (fixes shift/lag/rolling with gaps)
wide = wide.resample('1h').mean()
wide = wide.interpolate(limit=MAX_GAP_HOURS, limit_area='inside')
wide = wide.reset_index()

# Time features (built AFTER resampling so they exist for every hour)
wide['hour']        = wide['datetimeLocal'].dt.hour
wide['day_of_week'] = wide['datetimeLocal'].dt.dayofweek
wide['is_weekend']  = (wide['day_of_week'] >= 5).astype(int)
wide['hour_sin']    = np.sin(2 * np.pi * wide['hour'] / 24)
wide['hour_cos']    = np.cos(2 * np.pi * wide['hour'] / 24)
wide['dow_sin']     = np.sin(2 * np.pi * wide['day_of_week'] / 7)
wide['dow_cos']     = np.cos(2 * np.pi * wide['day_of_week'] / 7)

present = [p for p in POLLUTANTS if p in wide.columns]
print("\nWide shape:", wide.shape)
print("Columns:", wide.columns.tolist())
print("Pollutants available for AQI:", present)
if not present:
    raise SystemExit("No supported pollutant columns found; cannot compute AQI.")

# ------------------------------------------------------------
# 4. COMPUTE AQI (CPCB India breakpoints)
# ------------------------------------------------------------
def sub_index(c, bp):
    if pd.isna(c):
        return np.nan
    for cl, ch, il, ih in bp:
        if cl <= c <= ch:
            return ((ih - il) / (ch - cl)) * (c - cl) + il
    return bp[-1][3]

BP = {
    'pm25': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300),
             (120, 250, 301, 400), (250, 380, 401, 500)],
    'pm10': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300),
             (350, 430, 301, 400), (430, 600, 401, 500)],
    'no2':  [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200), (180, 280, 201, 300),
             (280, 400, 301, 400), (400, 600, 401, 500)],
    'o3':   [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200), (168, 208, 201, 300),
             (208, 748, 301, 400), (748, 1000, 401, 500)],
    'so2':  [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200), (380, 800, 201, 300),
             (800, 1600, 301, 400), (1600, 2000, 401, 500)],
    'co':   [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200), (10, 17, 201, 300),
             (17, 34, 301, 400), (34, 50, 401, 500)],
}

subindex_cols = []
for p in present:
    col = f'AQI_{p}'
    wide[col] = wide[p].apply(lambda x, p=p: sub_index(x, BP[p]))
    subindex_cols.append(col)

wide['AQI'] = wide[subindex_cols].max(axis=1)   # NaN only if ALL pollutants are NaN
print("\nAQI stats:\n", wide['AQI'].describe().round(2))

# ------------------------------------------------------------
# 5. FEATURE ENGINEERING (lags, rolling, differences)
# ------------------------------------------------------------
for col in ['AQI'] + present:
    for lag in [1, 2, 3, 6, 12, 24]:
        wide[f'{col}_lag{lag}'] = wide[col].shift(lag)

for w in [3, 6, 12, 24]:
    roll = wide['AQI'].rolling(w)
    wide[f'AQI_roll_mean_{w}'] = roll.mean()
    wide[f'AQI_roll_std_{w}']  = roll.std()
    wide[f'AQI_roll_max_{w}']  = roll.max()
    wide[f'AQI_roll_min_{w}']  = roll.min()

wide['AQI_diff_1']  = wide['AQI'] - wide['AQI_lag1']
wide['AQI_diff_24'] = wide['AQI'] - wide['AQI_lag24']

# Features = everything except timestamp and per-pollutant sub-indices
exclude = ['datetimeLocal'] + subindex_cols
feature_cols = [c for c in wide.columns if c not in exclude]
aqi_feats = [c for c in feature_cols if c.startswith('AQI')]
feature_cols = [c for c in feature_cols if not wide[c].isna().all()]

# ------------------------------------------------------------
# 6. DATASET BUILDER (chronological split, NO shuffle)
# ------------------------------------------------------------
def prepare(h):
    d = wide.copy()
    d['target'] = d['AQI'].shift(-h)
    d = d.dropna(subset=['target'] + [c for c in aqi_feats if c in feature_cols]).reset_index(drop=True)

    split = int(len(d) * (1 - TEST_FRAC))
    medians = d.loc[:split - 1, feature_cols].median().fillna(0)   # train-only medians

    X = d[feature_cols].fillna(medians)
    y = d['target']
    return {
        'X': X, 'y': y, 'split': split, 'medians': medians,
        'X_train': X.iloc[:split], 'X_test': X.iloc[split:],
        'y_train': y.iloc[:split], 'y_test': y.iloc[split:],
        'test_times': d['datetimeLocal'].iloc[split:],
    }

data = prepare(HORIZON)
X, y = data['X'], data['y']
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
test_times = data['test_times']

print(f"\nModel-ready rows: {len(X)}")
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}  |  Features: {len(feature_cols)}")
if len(X) < 200:
    print("WARNING: very little data; results will be unreliable.")

# ------------------------------------------------------------
# 7. TRAIN RANDOM FOREST
# ------------------------------------------------------------
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------
# 8. EVALUATE (with persistence baseline)
# ------------------------------------------------------------
def evaluate(y_true, y_pred, name):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    print(f"{name:>12} | MAE={mae:6.2f} | RMSE={rmse:6.2f} | R2={r2:.4f} | MAPE={mape:5.2f}%")

print(f"\n--- Performance (Horizon = {HORIZON}h) ---")
evaluate(y_train, rf.predict(X_train), "RF Train")
evaluate(y_test,  rf.predict(X_test),  "RF Test")
evaluate(y_test,  X_test['AQI'].values, "Persistence")   # "future = now" baseline

# Time-series CV (robustness check)
tscv = TimeSeriesSplit(n_splits=5)
cv_maes = []
for tr, va in tscv.split(X):
    m = RandomForestRegressor(n_estimators=200, max_depth=20,
                              random_state=42, n_jobs=-1)
    m.fit(X.iloc[tr], y.iloc[tr])
    cv_maes.append(mean_absolute_error(y.iloc[va], m.predict(X.iloc[va])))
print(f"\nCV MAE: {np.mean(cv_maes):.2f} +/- {np.std(cv_maes):.2f}")

# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------
imp = pd.DataFrame({'feature': feature_cols,
                    'importance': rf.feature_importances_}
                   ).sort_values('importance', ascending=False)

print("\nTop 15 features:\n", imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))
top = imp.head(20)
plt.barh(top['feature'][::-1], top['importance'][::-1], color='seagreen')
plt.xlabel('Importance')
plt.title(f'Feature Importance - RF predicting AQI (t+{HORIZON}h)')
plt.tight_layout()
plt.savefig('aqi_feature_importance.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 10. FORECAST PLOTS
# ------------------------------------------------------------
y_pred = rf.predict(X_test)

plt.figure(figsize=(15, 5))
plt.plot(test_times, y_test.values, label='Actual AQI', color='black', lw=1.2)
plt.plot(test_times, y_pred, label=f'Predicted AQI (t+{HORIZON}h)',
         color='crimson', lw=1.2, alpha=0.85)
plt.xlabel('Time'); plt.ylabel('AQI')
plt.title(f'Random Forest AQI Forecast (Horizon = {HORIZON}h)')
plt.legend(); plt.tight_layout()
plt.savefig('aqi_forecast.png', dpi=150)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.35, s=12, color='teal')
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', lw=2)
plt.xlabel('Actual AQI'); plt.ylabel('Predicted AQI')
plt.title(f'Actual vs Predicted AQI (Horizon = {HORIZON}h)')
plt.tight_layout()
plt.savefig('aqi_scatter.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# 11. AQI CATEGORY ACCURACY (operational metric)
# ------------------------------------------------------------
def aqi_category(a):
    if pd.isna(a): return 'Unknown'
    if a <= 50:    return 'Good'
    if a <= 100:   return 'Satisfactory'
    if a <= 200:   return 'Moderate'
    if a <= 300:   return 'Poor'
    if a <= 400:   return 'Very Poor'
    return 'Severe'

y_cat  = pd.Series(y_test.values).apply(aqi_category)
y_catp = pd.Series(y_pred).apply(aqi_category)
print(f"\nAQI Category Accuracy: {(y_cat == y_catp).mean() * 100:.2f}%")
print("\nCrosstab (actual rows x predicted cols):")
print(pd.crosstab(y_cat, y_catp))

# ------------------------------------------------------------
# 12. MULTI-HORIZON COMPARISON
# ------------------------------------------------------------
def run_horizon(h):
    d = prepare(h)
    m = RandomForestRegressor(n_estimators=200, max_depth=20,
                              random_state=42, n_jobs=-1)
    m.fit(d['X_train'], d['y_train'])
    p = m.predict(d['X_test'])
    base = d['X_test']['AQI'].values
    return (mean_absolute_error(d['y_test'], p),
            np.sqrt(mean_squared_error(d['y_test'], p)),
            r2_score(d['y_test'], p),
            mean_absolute_error(d['y_test'], base))

print("\n--- Multi-Horizon Performance ---")
print(f"{'Horizon':>8} | {'MAE':>8} | {'RMSE':>8} | {'R2':>8} | {'Persist MAE':>11}")
print("-" * 56)
for h in [1, 3, 6, 12, 24]:
    mae, rmse, r2, pmae = run_horizon(h)
    print(f"{h:>6}h | {mae:>8.2f} | {rmse:>8.2f} | {r2:>8.4f} | {pmae:>11.2f}")

# ------------------------------------------------------------
# 13. SAVE MODEL + PREDICTION FROM THE LATEST READING
# ------------------------------------------------------------
joblib.dump(rf, f'rf_aqi_forecast_h{HORIZON}.pkl')
joblib.dump(feature_cols, 'rf_aqi_features.pkl')
joblib.dump(data['medians'], 'rf_aqi_medians.pkl')
print(f"\nSaved model -> rf_aqi_forecast_h{HORIZON}.pkl")

# Use the most recent row of `wide` (the training table dropped these rows
# because they have no known future value yet).
latest = wide[feature_cols].iloc[[-1]]
if latest[[c for c in aqi_feats if c in feature_cols]].isna().any().any():
    print("WARNING: the latest row has missing AQI inputs (data gap); "
          "the prediction below may be unreliable.")
latest = latest.fillna(data['medians'])
next_aqi = rf.predict(latest)[0]
last_time = wide['datetimeLocal'].iloc[-1]
print(f"Latest data point: {last_time}")
print(f"Predicted AQI {HORIZON}h ahead: {next_aqi:.1f} ({aqi_category(next_aqi)})") 