"""
AWY-1: Machine Learning-Based Detection of Objects Moving Away
from Horizontally Oriented Ultrasonic Sensors with Workspace Clearance Prediction

Group    : AWY-1
Students : Parmar, Ayush (ayush.parmar@stud.fra-uas.de)
           Venkatesh, Baitipuli Akanksha (akanksha.venkatesh-baitipuli@stud.fra-uas.de)
Duration : 02-Dec-2025 – 20-Mar-2026
Sensor   : FIUS (Frankfurt Intelligent Ultrasonic Sensor)

Dataset  : Day 1 (single object, single condition)
           + Day 2 (Human / Chair / Metal Plate, Daylight + Night, Slow / Fast / Medium)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    classification_report
)
import warnings
import os
import sys
import joblib

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────────────────────
# 0. OUTPUT DIRECTORIES
# ─────────────────────────────────────────────────────────────
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables',  exist_ok=True)
os.makedirs('results/models',  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────
ZONE_THRESHOLD   = 0.90   # metres
WINDOW_SIZE      = 300    # readings per feature window (~7.9 s)
STRIDE_TRAIN     = 50
STRIDE_TEST      = 100
STRIDE_REG       = 50
TRAIN_FRAC       = 0.75

MEASUREMENT_RATE  = 38    # Hz
TIME_PER_READING  = 1.0 / MEASUREMENT_RATE

DISTANCE_COL = 10

LABEL_APPROACHING = 0
LABEL_AWAY        = 1
LABEL_STATIONARY  = 2
LABEL_NAMES  = {0: 'Approaching', 1: 'Away', 2: 'Stationary'}
CLASS_COLORS = {0: '#e74c3c', 1: '#27ae60', 2: '#2980b9'}

print("=" * 65)
print("AWY-1  Machine Learning Project — Initialising")
print("=" * 65)
print(f"  Zone threshold     : {ZONE_THRESHOLD} m  "
      f"({ZONE_THRESHOLD * 39.37:.1f} inches)")
print(f"  Window size        : {WINDOW_SIZE} readings "
      f"({WINDOW_SIZE * TIME_PER_READING * 1000:.0f} ms)")
print(f"  Measurement rate   : {MEASUREMENT_RATE} Hz  "
      f"({TIME_PER_READING * 1000:.1f} ms/reading)")

# ─────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────────────────────────
BASE    = r'D:/3rd sem/New folder/Machine Learning'
BASE_D2 = r'D:/3rd sem/New folder/Machine Learning Day 2'

DATA_FILES = {
    LABEL_APPROACHING: [
        f'{BASE}/Approaching/50-7_TWY_10000/signal_10000.csv',
        f'{BASE_D2}/Daylight/Approaching/Fast/signal_1000_120inch_1inch_Chair.csv',
        f'{BASE_D2}/Daylight/Approaching/Fast/signal_1000_120inch_1inch_Human.csv',
        f'{BASE_D2}/Daylight/Approaching/Slow/signal_3000_FROM_120inch_TO_1inch_HUMAN.csv',
        f'{BASE_D2}/Daylight/Approaching/Slow/signal_3000_120inch_10inch_CHAIR.csv',
    ],
    LABEL_AWAY: [
        f'{BASE}/Away/10_50cm_AWY_10000/signal_10000.csv',
        f'{BASE_D2}/Daylight/Away/Fast/signal_1000_inch_to_80inch_human.csv',
        f'{BASE_D2}/Daylight/Away/Fast/signal_1147_1inch_to_120inch_chair.csv',
        f'{BASE_D2}/Daylight/Away/Fast/signal_1198_1inch_to_120inch_metalplate.csv',
        f'{BASE_D2}/Daylight/Away/Fast/signal_1200_1inch_to_120inch_human.csv',
        f'{BASE_D2}/Daylight/Away/Slow/signal_4190_1inch_to_120inch_chair.csv',
        f'{BASE_D2}/Daylight/Away/Slow/signal_4820_1inch_to120inch_metalplate.csv',
        f'{BASE_D2}/Daylight/Away/Slow/signal_5000_1inch_to_120inch_human.csv',
        f'{BASE_D2}/at night/Away/Fast/signal_578_from1_to_92_Metalplate.csv',
        f'{BASE_D2}/at night/Away/medium/signal_2763_From_1_to_92_inch.csv',
        f'{BASE_D2}/at night/Away/Slow/signal_2632_1inch_to_100inch_Chair.csv',
        f'{BASE_D2}/at night/Away/Slow/signal_2693_1inch_to_120inch_human.csv',
        f'{BASE_D2}/at night/Away/Slow/signal_4027_1inch_to 120inch_metalplate.csv',
    ],
    LABEL_STATIONARY: [
        f'{BASE}/Stationary/signal_5000_50cm/signal_5000.csv',
        f'{BASE}/Stationary/signal_10000_30cm/signal_10000.csv',
        f'{BASE_D2}/Daylight/Stationary/signal_5000_50_CHAIR.csv',
        f'{BASE_D2}/Daylight/Stationary/signal_5000_100_CHAIR.csv',
    ],
}

CLIP_RANGES = {
    LABEL_APPROACHING: (0.05, 5.0),
    LABEL_AWAY:        (0.05, 5.0),
    LABEL_STATIONARY:  (0.05, 5.0),
}


def load_and_clean(filepath, clip_lo, clip_hi):
    d = pd.read_csv(filepath, header=None, usecols=[DISTANCE_COL])[DISTANCE_COL].values
    d = np.clip(d, clip_lo, clip_hi)
    s = pd.Series(d)
    roll_med = s.rolling(21, center=True, min_periods=1).median()
    roll_mad = (s - roll_med).abs().rolling(21, center=True, min_periods=1).median()
    roll_mad = roll_mad.replace(0, roll_mad[roll_mad > 0].min())
    mask = ((s - roll_med).abs() / roll_mad) < 6
    return d[mask.values]


raw_data_files = {}
raw_data       = {}

print("\nLoading sensor data …")
for label, files in DATA_FILES.items():
    lo, hi = CLIP_RANGES[label]
    arrays = []
    for f in files:
        arr = load_and_clean(f, lo, hi)
        arrays.append(arr)
        print(f"  [{LABEL_NAMES[label]:12s}] {os.path.basename(f):55s} "
              f"n={len(arr):5d}  d=[{arr.min():.2f},{arr.max():.2f}]m")
    raw_data_files[label] = arrays
    raw_data[label]       = np.concatenate(arrays)
    d_all = raw_data[label]
    print(f"  → {LABEL_NAMES[label]} total : {len(d_all):6d} readings | "
          f"d = [{d_all.min():.3f}, {d_all.max():.3f}] m | "
          f"mean = {d_all.mean():.3f} m\n")

Y_MAX = max(d.max() for d in raw_data.values()) * 1.05

# ─────────────────────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────────────────────
print("\nGenerating EDA plots …")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('AWY-1: FIUS Sensor Data — Exploratory Data Analysis (Day 1 + Day 2)',
             fontsize=14, fontweight='bold')

for col, (label, distances) in enumerate(raw_data.items()):
    name  = LABEL_NAMES[label]
    color = CLASS_COLORS[label]
    show  = distances[:min(len(distances), 6000)]
    t     = np.arange(len(show)) * TIME_PER_READING

    ax = axes[0, col]
    ax.plot(t, show, color=color, linewidth=0.9, alpha=0.8)
    ax.axhline(ZONE_THRESHOLD, color='#7f8c8d', linestyle='--',
               linewidth=1.8, label=f'Zone threshold ({ZONE_THRESHOLD} m)')
    ax.fill_between(t, 0, ZONE_THRESHOLD, alpha=0.07, color='red')
    ax.fill_between(t, ZONE_THRESHOLD, Y_MAX, alpha=0.04, color='green')
    ax.set_title(f'{name} — Distance vs Time', fontsize=12)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (m)')
    ax.set_ylim(0.0, Y_MAX)
    ax.legend(fontsize=8)

    ax = axes[1, col]
    ax.hist(distances, bins=60, color=color, alpha=0.75,
            edgecolor='white', linewidth=0.4, density=True)
    ax.axvline(ZONE_THRESHOLD, color='#7f8c8d', linestyle='--',
               linewidth=1.8, label='Zone threshold')
    ax.set_title(f'{name} — Distance Distribution', fontsize=12)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/figures/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/01_eda_overview.png")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'mean_dist', 'std_dist', 'min_dist', 'max_dist', 'range_dist',
    'velocity_ms', 'acceleration_ms2', 'trend_noise',
    'start_dist', 'end_dist', 'center_dist',
    'net_displacement', 'norm_velocity',
    'frac_increasing', 'frac_decreasing',
    'in_zone',
]


def extract_features(distances, start_idx, win=WINDOW_SIZE):
    w = distances[start_idx: start_idx + win]
    t = np.arange(win, dtype=float)

    mean_d   = float(np.mean(w))
    std_d    = float(np.std(w))
    min_d    = float(np.min(w))
    max_d    = float(np.max(w))
    range_d  = max_d - min_d

    slope, intercept = np.polyfit(t, w, 1)
    velocity_ms      = slope * MEASUREMENT_RATE

    h  = win // 2
    v1 = float(np.polyfit(t[:h], w[:h], 1)[0])
    v2 = float(np.polyfit(t[h:], w[h:], 1)[0])
    accel = (v2 - v1) / (h / MEASUREMENT_RATE)

    residuals   = w - (slope * t + intercept)
    trend_noise = float(np.std(residuals))
    net_disp    = float(w[-1] - w[0])
    norm_vel    = velocity_ms / max(mean_d, 0.1)

    diffs    = np.diff(w)
    frac_inc = float((diffs > 0).mean())
    frac_dec = float((diffs < 0).mean())

    return {
        'mean_dist':        mean_d,
        'std_dist':         std_d,
        'min_dist':         min_d,
        'max_dist':         max_d,
        'range_dist':       range_d,
        'velocity_ms':      velocity_ms,
        'acceleration_ms2': accel,
        'trend_noise':      trend_noise,
        'start_dist':       float(w[0]),
        'end_dist':         float(w[-1]),
        'center_dist':      float(w[win // 2]),
        'net_displacement': net_disp,
        'norm_velocity':    norm_vel,
        'frac_increasing':  frac_inc,
        'frac_decreasing':  frac_dec,
        'in_zone':          1 if mean_d < ZONE_THRESHOLD else 0,
    }


# ─────────────────────────────────────────────────────────────
# 5. TtC GROUND TRUTH
# ─────────────────────────────────────────────────────────────
ttc_lookup = {}
offset_away = 0

for arr in raw_data_files[LABEL_AWAY]:
    n = len(arr)
    for i in range(n):
        if arr[i] >= ZONE_THRESHOLD:
            continue
        for j in range(i + 1, n):
            if arr[j] >= ZONE_THRESHOLD:
                ttc_lookup[offset_away + i] = (j - i) * TIME_PER_READING
                break
    offset_away += n

away_dist = raw_data[LABEL_AWAY]
ttc_idx   = np.array(sorted(ttc_lookup.keys()))
ttc_val   = np.array([ttc_lookup[i] for i in ttc_idx])

print(f"\nTtC ground truth computed:")
print(f"  Readings with TtC : {len(ttc_idx)} / {len(away_dist)}")
if len(ttc_val) > 0:
    print(f"  TtC range         : [{ttc_val.min():.2f}, {ttc_val.max():.2f}] s")
    print(f"  TtC mean ± std    : {ttc_val.mean():.2f} ± {ttc_val.std():.2f} s")

# ─────────────────────────────────────────────────────────────
# 6. WINDOWED DATASET CONSTRUCTION
# ─────────────────────────────────────────────────────────────
def build_windows(distances, label, stride, i_start, i_end):
    rows = []
    for i in range(i_start, min(i_end, len(distances) - WINDOW_SIZE + 1), stride):
        feats = extract_features(distances, i)
        feats['label']        = label
        feats['window_start'] = i
        rows.append(feats)
    return pd.DataFrame(rows)


train_dfs, test_dfs = [], []
for label, arrays in raw_data_files.items():
    for arr in arrays:
        n     = len(arr)
        split = int(n * TRAIN_FRAC)
        train_dfs.append(build_windows(arr, label, STRIDE_TRAIN, 0,     split))
        test_dfs .append(build_windows(arr, label, STRIDE_TEST,  split, n    ))

clf_train = pd.concat(train_dfs, ignore_index=True)
clf_test  = pd.concat(test_dfs,  ignore_index=True)

print(f"\nClassification windows:")
print(f"  Train : {len(clf_train):5d}  {clf_train['label'].value_counts().sort_index().to_dict()}")
print(f"  Test  : {len(clf_test):5d}  {clf_test['label'].value_counts().sort_index().to_dict()}")

reg_all_rows = []
offset_reg   = 0
for arr in raw_data_files[LABEL_AWAY]:
    n = len(arr)
    for i in range(0, n - WINDOW_SIZE + 1, STRIDE_REG):
        end_idx = offset_reg + i + WINDOW_SIZE - 1
        if end_idx in ttc_lookup:
            feats = extract_features(arr, i)
            feats['ttc']          = ttc_lookup[end_idx]
            feats['window_start'] = offset_reg + i
            reg_all_rows.append(feats)
    offset_reg += n

reg_all      = pd.DataFrame(reg_all_rows).reset_index(drop=True)
reg_all_shuf = reg_all.sample(frac=1, random_state=42).reset_index(drop=True)
n_reg        = len(reg_all_shuf)
split_reg    = int(n_reg * TRAIN_FRAC)
reg_train    = reg_all_shuf.iloc[:split_reg].copy()
reg_test     = reg_all_shuf.iloc[split_reg:].copy()
split_away   = reg_all.iloc[split_reg]['window_start'] if split_reg < n_reg else len(away_dist)

print(f"\nRegression windows (TtC, Away in zone):")
print(f"  Total : {n_reg:4d}  TtC=[{reg_all['ttc'].min():.2f}, {reg_all['ttc'].max():.2f}] s")
print(f"  Train : {len(reg_train):4d}")
print(f"  Test  : {len(reg_test):4d}")

# ─────────────────────────────────────────────────────────────
# 7. FEATURE DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────
print("\nGenerating feature distribution plots …")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('AWY-1: Feature Distributions by Motion Class (Training Set)',
             fontsize=14, fontweight='bold')

key_feats   = ['velocity_ms', 'net_displacement', 'frac_increasing',
               'mean_dist', 'std_dist', 'range_dist']
feat_labels = ['Velocity (m/s)', 'Net Displacement (m)', 'Frac. Increasing',
               'Mean Distance (m)', 'Std Distance (m)', 'Range Distance (m)']

for ax, feat, label in zip(axes.flat, key_feats, feat_labels):
    for lbl in [0, 1, 2]:
        data = clf_train.loc[clf_train['label'] == lbl, feat]
        ax.hist(data, bins=30, alpha=0.6, density=True,
                label=LABEL_NAMES[lbl], color=CLASS_COLORS[lbl],
                edgecolor='white', linewidth=0.4)
    ax.set_title(label, fontsize=11)
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/figures/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/02_feature_distributions.png")

# ─────────────────────────────────────────────────────────────
# 8. PREPARE MATRICES & SCALE
# ─────────────────────────────────────────────────────────────
X_tr_clf = clf_train[FEATURE_COLS].values
y_tr_clf = clf_train['label'].values
X_te_clf = clf_test[FEATURE_COLS].values
y_te_clf = clf_test['label'].values

X_tr_reg = reg_train[FEATURE_COLS].values
y_tr_reg = reg_train['ttc'].values
X_te_reg = reg_test[FEATURE_COLS].values
y_te_reg = reg_test['ttc'].values

sc_clf = StandardScaler().fit(X_tr_clf)
Xs_tr_clf = sc_clf.transform(X_tr_clf)
Xs_te_clf = sc_clf.transform(X_te_clf)

sc_reg = StandardScaler().fit(X_tr_reg)
Xs_tr_reg = sc_reg.transform(X_tr_reg)
Xs_te_reg = sc_reg.transform(X_te_reg)

print(f"\nFeature matrices ready:")
print(f"  Classification — train: {X_tr_clf.shape}, test: {X_te_clf.shape}")
print(f"  Regression     — train: {X_tr_reg.shape}, test: {X_te_reg.shape}")

# ─────────────────────────────────────────────────────────────
# 9. CLASSIFICATION — LOGISTIC REGRESSION (BASELINE)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("9. CLASSIFICATION — Logistic Regression (baseline)")
print("─" * 55)

lr_clf = LogisticRegression(max_iter=2000, random_state=42,
                             solver='lbfgs', C=1.0)
lr_clf.fit(Xs_tr_clf, y_tr_clf)
lr_clf_pred = lr_clf.predict(Xs_te_clf)
lr_clf_acc  = accuracy_score(y_te_clf, lr_clf_pred)
print(f"Accuracy : {lr_clf_acc*100:.2f}%")
print(classification_report(y_te_clf, lr_clf_pred,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))

# ─────────────────────────────────────────────────────────────
# 10. CLASSIFICATION — DECISION TREE
# ─────────────────────────────────────────────────────────────
print("─" * 55)
print("10. CLASSIFICATION — Decision Tree")
print("─" * 55)

dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_tr_clf, y_tr_clf)
dt_clf_pred = dt_clf.predict(X_te_clf)
dt_clf_acc  = accuracy_score(y_te_clf, dt_clf_pred)
print(f"Accuracy : {dt_clf_acc*100:.2f}%")
print(classification_report(y_te_clf, dt_clf_pred,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))

# ─────────────────────────────────────────────────────────────
# 11. CLASSIFICATION — KNN
# ─────────────────────────────────────────────────────────────
print("─" * 55)
print("11. CLASSIFICATION — K-Nearest Neighbors")
print("─" * 55)

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(Xs_tr_clf, y_tr_clf)
knn_clf_pred = knn_clf.predict(Xs_te_clf)
knn_clf_acc  = accuracy_score(y_te_clf, knn_clf_pred)
print(f"Accuracy : {knn_clf_acc*100:.2f}%")
print(classification_report(y_te_clf, knn_clf_pred,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))

# ─────────────────────────────────────────────────────────────
# 12. CLASSIFICATION — SVM
# ─────────────────────────────────────────────────────────────
print("─" * 55)
print("12. CLASSIFICATION — Support Vector Machine")
print("─" * 55)

svm_clf = SVC(kernel='rbf', C=1.0, random_state=42)
svm_clf.fit(Xs_tr_clf, y_tr_clf)
svm_clf_pred = svm_clf.predict(Xs_te_clf)
svm_clf_acc  = accuracy_score(y_te_clf, svm_clf_pred)
print(f"Accuracy : {svm_clf_acc*100:.2f}%")
print(classification_report(y_te_clf, svm_clf_pred,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))

# ─────────────────────────────────────────────────────────────
# 13. CLASSIFICATION — RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("─" * 55)
print("13. CLASSIFICATION — Random Forest")
print("─" * 55)

rf_clf = RandomForestClassifier(n_estimators=500, max_depth=None,
                                 min_samples_split=3, random_state=42, n_jobs=-1)
rf_clf.fit(X_tr_clf, y_tr_clf)
joblib.dump(rf_clf, 'results/models/rf_classifier.joblib')
print("  Model saved → results/models/rf_classifier.joblib")
rf_clf_pred = rf_clf.predict(X_te_clf)
rf_clf_acc  = accuracy_score(y_te_clf, rf_clf_pred)
print(f"Accuracy : {rf_clf_acc*100:.2f}%")
print(classification_report(y_te_clf, rf_clf_pred,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))

feat_imp_clf = pd.Series(rf_clf.feature_importances_,
                          index=FEATURE_COLS).sort_values(ascending=False)
print("Feature importances (top 6):")
print(feat_imp_clf.head(6).round(4))

# ─────────────────────────────────────────────────────────────
# 14. CONFUSION MATRICES (all 5 classifiers + 2 baselines)
# ─────────────────────────────────────────────────────────────
print("\nGenerating confusion matrix plots …")
class_names = [LABEL_NAMES[i] for i in range(3)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('AWY-1: Classification Confusion Matrices', fontsize=14, fontweight='bold')

for ax, pred, title in zip(axes,
                            [lr_clf_pred, rf_clf_pred],
                            ['Logistic Regression (Baseline)', 'Random Forest']):
    cm      = confusion_matrix(y_te_clf, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor='#ecf0f1',
                annot_kws={'size': 12, 'weight': 'bold'})
    acc = accuracy_score(y_te_clf, pred)
    ax.set_title(f'{title}\nAccuracy = {acc*100:.1f}%', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('results/figures/03_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/03_confusion_matrices.png")

# Feature importance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
colors_fi = ['#27ae60' if v >= feat_imp_clf.mean() else '#bdc3c7'
             for v in feat_imp_clf.sort_values()]
feat_imp_clf.sort_values().plot(kind='barh', ax=ax,
                                 color=colors_fi, edgecolor='white')
ax.axvline(feat_imp_clf.mean(), color='#e74c3c', linestyle='--',
           alpha=0.8, label='Mean importance')
ax.set_title('Random Forest — Feature Importance (Classification)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/04_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 15. TtC REGRESSION — LINEAR REGRESSION (BASELINE)
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("15. TtC REGRESSION — Linear Regression (baseline)")
print("─" * 55)

lr_reg = LinearRegression()
lr_reg.fit(Xs_tr_reg, y_tr_reg)
lr_reg_pred = np.clip(lr_reg.predict(Xs_te_reg), 0, None)
lr_reg_mae  = mean_absolute_error(y_te_reg, lr_reg_pred)
lr_reg_rmse = np.sqrt(mean_squared_error(y_te_reg, lr_reg_pred))
g_print(f"MAE  : {lr_remae:.4f} s   target <0.6 s → {'PASS ✓' if lr_reg_mae < 0.6 else 'FAIL ✗'}")
print(f"RMSE : {lr_reg_rmse:.4f} s")

# ─────────────────────────────────────────────────────────────
# 16. TtC REGRESSION — RANDOM FOREST
# ─────────────────────────────────────────────────────────────
print("─" * 55)
print("16. TtC REGRESSION — Random Forest")
print("─" * 55)

rf_reg = RandomForestRegressor(n_estimators=500, max_depth=None,
                                min_samples_split=3, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr_reg, y_tr_reg)
joblib.dump(rf_reg, 'results/models/rf_regressor.joblib')
print("  Model saved → results/models/rf_regressor.joblib")
rf_reg_pred = np.clip(rf_reg.predict(X_te_reg), 0, None)
rf_reg_mae  = mean_absolute_error(y_te_reg, rf_reg_pred)
rf_reg_rmse = np.sqrt(mean_squared_error(y_te_reg, rf_reg_pred))
print(f"MAE  : {rf_reg_mae:.4f} s   target <0.6 s → {'PASS ✓' if rf_reg_mae < 0.6 else 'FAIL ✗'}")
print(f"RMSE : {rf_reg_rmse:.4f} s")

feat_imp_reg = pd.Series(rf_reg.feature_importances_,
                          index=FEATURE_COLS).sort_values(ascending=False)
print("Feature importances (top 6):")
print(feat_imp_reg.head(6).round(4))

# ─────────────────────────────────────────────────────────────
# 17. TtC SCATTER PLOT
# ─────────────────────────────────────────────────────────────
print("\nGenerating TtC prediction plots …")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('AWY-1: Time-to-Clear Prediction — Actual vs Predicted',
             fontsize=14, fontweight='bold')

for ax, pred, title, mae_v, rmse_v in zip(
        axes,
        [lr_reg_pred, rf_reg_pred],
        ['Linear Regression (Baseline)', 'Random Forest'],
        [lr_reg_mae, rf_reg_mae],
        [lr_reg_rmse, rf_reg_rmse]):

    lo = min(y_te_reg.min(), pred.min()) - 1
    hi = max(y_te_reg.max(), pred.max()) + 1
    ax.scatter(y_te_reg, pred, alpha=0.7, color='#2980b9',
               edgecolor='white', s=50)
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.8, label='Ideal (y=x)')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'{title}\nMAE={mae_v:.3f}s | RMSE={rmse_v:.3f}s',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Actual TtC (s)')
    ax.set_ylabel('Predicted TtC (s)')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/05_ttc_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/05_ttc_scatter.png")

# ─────────────────────────────────────────────────────────────
# 18. RULE-BASED BASELINES
# ─────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("18. RULE-BASED BASELINES")
print("─" * 55)

VELOCITY_EPSILON = 0.02


def baseline_threshold_only(row):
    return LABEL_STATIONARY


def baseline_trend(row):
    v = row['velocity_ms']
    if v > VELOCITY_EPSILON:
        return LABEL_AWAY
    elif v < -VELOCITY_EPSILON:
        return LABEL_APPROACHING
    else:
        return LABEL_STATIONARY


threshold_preds = np.array([baseline_threshold_only(r)
                             for _, r in clf_test.iterrows()])
trend_preds     = np.array([baseline_trend(r)
                             for _, r in clf_test.iterrows()])

threshold_acc = accuracy_score(y_te_clf, threshold_preds)
trend_acc     = accuracy_score(y_te_clf, trend_preds)

print(f"Threshold-only accuracy : {threshold_acc*100:.2f}%")
print(f"Trend-based    accuracy : {trend_acc*100:.2f}%")
print(classification_report(y_te_clf, trend_preds,
                             target_names=[LABEL_NAMES[i] for i in range(3)]))


def trend_ttc(row):
    v = row['velocity_ms']
    d = row['mean_dist']
    if v > 0 and d < ZONE_THRESHOLD:
        return max((ZONE_THRESHOLD - d) / v, 0.0)
    return 0.0


trend_ttc_preds = np.array([trend_ttc(r) for _, r in reg_test.iterrows()])
trend_ttc_mae   = mean_absolute_error(y_te_reg, trend_ttc_preds)
trend_ttc_rmse  = np.sqrt(mean_squared_error(y_te_reg, trend_ttc_preds))
print(f"Trend-based TtC  MAE  : {trend_ttc_mae:.4f} s")
print(f"Trend-based TtC  RMSE : {trend_ttc_rmse:.4f} s")

# ─────────────────────────────────────────────────────────────
# 19. RESULTS COMPARISON VISUALISATION
# ─────────────────────────────────────────────────────────────
print("\nGenerating results comparison plots …")
fig = plt.figure(figsize=(18, 12))
fig.suptitle('AWY-1: Complete Results — Classification & TtC Prediction (Day 1 + Day 2)',
             fontsize=15, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# (A) Classification accuracy bar chart — all 5 classifiers + 2 baselines
ax_acc = fig.add_subplot(gs[0, :2])
models_clf   = ['Threshold\nOnly\n(B1)', 'Trend\nBased\n(B2)',
                'Logistic\nRegression', 'Decision\nTree',
                'KNN', 'SVM', 'Random\nForest']
accs_clf     = [threshold_acc*100, trend_acc*100,
                lr_clf_acc*100, dt_clf_acc*100,
                knn_clf_acc*100, svm_clf_acc*100,
                rf_clf_acc*100]
bar_cols_clf = ['#95a5a6', '#bdc3c7', '#3498db', '#9b59b6',
                '#e67e22', '#1abc9c', '#27ae60']
bars_clf     = ax_acc.bar(models_clf, accs_clf, color=bar_cols_clf,
                           edgecolor='white', linewidth=0.5, width=0.55)
ax_acc.axhline(85, color='#e74c3c', linestyle='--',
               linewidth=2, label='Target ≥85%')
ax_acc.set_title('Classification Accuracy — All Models', fontsize=12, fontweight='bold')
ax_acc.set_ylabel('Accuracy (%)')
ax_acc.set_ylim(0, 108)
ax_acc.legend(fontsize=10)
for bar, acc in zip(bars_clf, accs_clf):
    ax_acc.text(bar.get_x() + bar.get_width()/2., acc + 1,
                f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

# (B) TtC MAE bar chart
ax_mae = fig.add_subplot(gs[0, 2])
models_reg   = ['Trend\nBased\n(B2)', 'Linear\nReg.', 'Random\nForest']
maes_reg     = [trend_ttc_mae, lr_reg_mae, rf_reg_mae]
bar_cols_reg = ['#bdc3c7', '#3498db', '#27ae60']
bars_mae     = ax_mae.bar(models_reg, maes_reg, color=bar_cols_reg,
                           edgecolor='white', linewidth=0.5, width=0.5)
ax_mae.axhline(0.6, color='#e74c3c', linestyle='--',
               linewidth=2, label='Target <0.6 s')
ax_mae.set_title('TtC Prediction MAE', fontsize=12, fontweight='bold')
ax_mae.set_ylabel('MAE (seconds)')
ax_mae.legend(fontsize=9)
for bar, m in zip(bars_mae, maes_reg):
    ax_mae.text(bar.get_x() + bar.get_width()/2., m + 0.02,
                f'{m:.3f}s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

# (C) Away trajectory
ax_traj = fig.add_subplot(gs[1, :2])
away_d1 = raw_data_files[LABEL_AWAY][0]
t_a     = np.arange(len(away_d1)) * TIME_PER_READING
y_traj  = min(away_d1.max() * 1.1, Y_MAX)
ax_traj.fill_between(t_a, 0, ZONE_THRESHOLD, alpha=0.12,
                     color='#e74c3c', label='Zone occupied')
ax_traj.fill_between(t_a, ZONE_THRESHOLD, y_traj, alpha=0.08,
                     color='#27ae60', label='Zone clear')
ax_traj.plot(t_a, away_d1, color='#27ae60', linewidth=1.5,
             alpha=0.85, label='Measured distance')
ax_traj.axhline(ZONE_THRESHOLD, color='#7f8c8d', linestyle='--',
               linewidth=2, label=f'Zone threshold ({ZONE_THRESHOLD} m)')
ax_traj.set_title('Away Trajectory (Day 1) with Zone Definition',
                  fontsize=12, fontweight='bold')
ax_traj.set_xlabel('Time (s)')
ax_traj.set_ylabel('Distance (m)')
ax_traj.set_ylim(0.0, y_traj)
ax_traj.legend(fontsize=9, loc='lower right')

# (D) TtC actual vs predicted
ax_ttc = fig.add_subplot(gs[1, 2])
ax_ttc.plot(ttc_idx * TIME_PER_READING, ttc_val,
            color='#2980b9', linewidth=2.5, label='Actual TtC', alpha=0.9)
all_reg_sorted = reg_all.sort_values('window_start')
if len(all_reg_sorted) > 0:
    all_rf_ttc = np.clip(rf_reg.predict(
        all_reg_sorted[FEATURE_COLS].values), 0, None)
    ax_ttc.scatter(all_reg_sorted['window_start'] * TIME_PER_READING,
                   all_rf_ttc, color='#e74c3c', s=18, alpha=0.7,
                   zorder=5, label='RF predicted TtC')
ax_ttc.axvline(int(split_away) * TIME_PER_READING, color='black',
               linestyle=':', linewidth=2, label='Train|Test split')
ax_ttc.set_title('TtC: Actual vs RF Predicted', fontsize=12, fontweight='bold')
ax_ttc.set_xlabel('Time (s)')
ax_ttc.set_ylabel('TtC (s)')
ax_ttc.legend(fontsize=8)

plt.savefig('results/figures/06_results_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/06_results_summary.png")
# ─────────────────────────────────────────────────────────────
# 19B. DEDICATED MODEL COMPARISON FIGURE
# ─────────────────────────────────────────────────────────────
print("\nGenerating dedicated model comparison figure …")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('AWY-1: Model Performance Comparison',
             fontsize=15, fontweight='bold')

# --- Left: Classification accuracy all models ---
ax1 = axes[0]
models_all   = ['Threshold\nOnly\n(B1)', 'Trend\nBased\n(B2)',
                'Logistic\nRegression', 'Decision\nTree',
                'KNN', 'SVM', 'Random\nForest']
accs_all     = [threshold_acc*100, trend_acc*100,
                lr_clf_acc*100, dt_clf_acc*100,
                knn_clf_acc*100, svm_clf_acc*100,
                rf_clf_acc*100]
colors_all   = ['#95a5a6', '#bdc3c7', '#3498db', '#9b59b6',
                '#e67e22', '#1abc9c', '#27ae60']

bars = ax1.bar(models_all, accs_all, color=colors_all,
               edgecolor='white', linewidth=0.8, width=0.6)
ax1.axhline(85, color='#e74c3c', linestyle='--',
            linewidth=2, label='Target ≥ 85%')
ax1.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(0, 112)
ax1.legend(fontsize=11)
ax1.tick_params(axis='x', labelsize=10)
for bar, acc in zip(bars, accs_all):
    ax1.text(bar.get_x() + bar.get_width()/2., acc + 1.2,
             f'{acc:.1f}%', ha='center', va='bottom',
             fontsize=10, fontweight='bold')

# --- Right: TtC MAE all regression models ---
ax2 = axes[1]
reg_models = ['Trend\nBased\n(B2)', 'Linear\nRegression', 'Random\nForest']
reg_maes   = [trend_ttc_mae, lr_reg_mae, rf_reg_mae]
reg_colors = ['#bdc3c7', '#3498db', '#27ae60']

bars2 = ax2.bar(reg_models, reg_maes, color=reg_colors,
                edgecolor='white', linewidth=0.8, width=0.5)
ax2.axhline(0.6, color='#e74c3c', linestyle='--',
            linewidth=2, label='Target < 0.6 s')
ax2.set_title('TtC Regression — MAE (seconds)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error (s)', fontsize=12)
ax2.legend(fontsize=11)
ax2.tick_params(axis='x', labelsize=11)
for bar, m in zip(bars2, reg_maes):
    ax2.text(bar.get_x() + bar.get_width()/2., m + 1.5,
             f'{m:.2f} s', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/08_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/08_model_comparison.png")

# ─────────────────────────────────────────────────────────────
# 20. PER-CLASS PERFORMANCE (RF only)
# ─────────────────────────────────────────────────────────────
print("\nGenerating per-class performance breakdown …")
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle('AWY-1: Random Forest — Per-Class Performance (Test Set)',
             fontsize=14, fontweight='bold')

metrics_per_class = {}
for i in range(3):
    prec = precision_score(y_te_clf == i, rf_clf_pred == i)
    rec  = recall_score(y_te_clf == i,    rf_clf_pred == i)
    f1   = f1_score(y_te_clf == i,        rf_clf_pred == i)
    metrics_per_class[LABEL_NAMES[i]] = dict(Precision=prec, Recall=rec, F1=f1)

names_m    = list(metrics_per_class.keys())
bar_w      = 0.22
x          = np.arange(len(names_m))
met_names  = ['Precision', 'Recall', 'F1']
met_colors = ['#3498db', '#27ae60', '#e74c3c']

ax = axes[0]
for k, (mn, mc) in enumerate(zip(met_names, met_colors)):
    vals = [metrics_per_class[n][mn] for n in names_m]
    ax.bar(x + k * bar_w, vals, width=bar_w, label=mn,
           color=mc, edgecolor='white')
ax.set_xticks(x + bar_w)
ax.set_xticklabels(names_m)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 per Class')
ax.axhline(0.85, color='gray', linestyle=':', linewidth=1.5)
ax.legend(fontsize=9)

cm   = confusion_matrix(y_te_clf, rf_clf_pred)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt='.1%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1], linewidths=0.5,
            annot_kws={'size': 13, 'weight': 'bold'})
axes[1].set_title('Confusion Matrix (RF)')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

residuals = y_te_reg - rf_reg_pred
axes[2].hist(residuals, bins=20, color='#2980b9', edgecolor='white', alpha=0.8)
axes[2].axvline(0, color='red', linestyle='--', linewidth=1.5)
axes[2].set_title(f'TtC Residuals (RF)\nMAE={rf_reg_mae:.3f}s')
axes[2].set_xlabel('Actual − Predicted (s)')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('results/figures/07_detailed_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/figures/07_detailed_performance.png")

# ─────────────────────────────────────────────────────────────
# 21. RESULTS TABLES (CSV)
# ─────────────────────────────────────────────────────────────
print("\nSaving results tables …")

clf_results = pd.DataFrame([
    {
        'Model': 'Threshold-only (Baseline 1)',
        'Accuracy': f"{threshold_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, threshold_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    threshold_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        threshold_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Notes': 'Cannot classify motion direction; predicts all Stationary',
    },
    {
        'Model': 'Trend-based (Baseline 2)',
        'Accuracy': f"{trend_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, trend_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    trend_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        trend_preds, average='weighted', zero_division=0)*100:.2f}%",
        'Notes': 'Velocity sign with fixed epsilon threshold',
    },
    {
        'Model': 'Logistic Regression',
        'Accuracy': f"{lr_clf_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, lr_clf_pred, average='weighted')*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    lr_clf_pred, average='weighted')*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        lr_clf_pred, average='weighted')*100:.2f}%",
        'Notes': 'Linear baseline ML; StandardScaler applied',
    },
    {
        'Model': 'Decision Tree',
        'Accuracy': f"{dt_clf_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, dt_clf_pred, average='weighted')*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    dt_clf_pred, average='weighted')*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        dt_clf_pred, average='weighted')*100:.2f}%",
        'Notes': 'Single tree; no scaling required',
    },
    {
        'Model': 'K-Nearest Neighbors',
        'Accuracy': f"{knn_clf_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, knn_clf_pred, average='weighted')*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    knn_clf_pred, average='weighted')*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        knn_clf_pred, average='weighted')*100:.2f}%",
        'Notes': 'k=5; StandardScaler applied',
    },
    {
        'Model': 'SVM (RBF kernel)',
        'Accuracy': f"{svm_clf_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, svm_clf_pred, average='weighted')*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    svm_clf_pred, average='weighted')*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        svm_clf_pred, average='weighted')*100:.2f}%",
        'Notes': 'RBF kernel C=1.0; StandardScaler applied',
    },
    {
        'Model': 'Random Forest',
        'Accuracy': f"{rf_clf_acc*100:.2f}%",
        'Avg Precision': f"{precision_score(y_te_clf, rf_clf_pred, average='weighted')*100:.2f}%",
        'Avg Recall':    f"{recall_score(y_te_clf,    rf_clf_pred, average='weighted')*100:.2f}%",
        'Avg F1':        f"{f1_score(y_te_clf,        rf_clf_pred, average='weighted')*100:.2f}%",
        'Notes': '500 trees; no scaling required',
    },
])
clf_results.to_csv('results/tables/classification_results.csv', index=False)
print("  Saved: results/tables/classification_results.csv")

reg_results = pd.DataFrame([
    {'Model': 'Threshold-only (Baseline 1)', 'MAE (s)': 'N/A',
     'RMSE (s)': 'N/A', 'Target (<0.6 s)': 'N/A',
     'Notes': 'Cannot predict TtC'},
    {'Model': 'Trend-based (Baseline 2)',
     'MAE (s)': f"{trend_ttc_mae:.4f}",
     'RMSE (s)': f"{trend_ttc_rmse:.4f}",
     'Target (<0.6 s)': '✓' if trend_ttc_mae < 0.6 else '✗',
     'Notes': 'TtC = (thresh − d) / velocity'},
    {'Model': 'Linear Regression',
     'MAE (s)': f"{lr_reg_mae:.4f}",
     'RMSE (s)': f"{lr_reg_rmse:.4f}",
     'Target (<0.6 s)': '✓' if lr_reg_mae < 0.6 else '✗',
     'Notes': 'Baseline ML regressor; StandardScaler applied'},
    {'Model': 'Random Forest',
     'MAE (s)': f"{rf_reg_mae:.4f}",
     'RMSE (s)': f"{rf_reg_rmse:.4f}",
     'Target (<0.6 s)': '✓' if rf_reg_mae < 0.6 else '✗',
     'Notes': '500 trees; no scaling required'},
])
reg_results.to_csv('results/tables/ttc_regression_results.csv', index=False)
print("  Saved: results/tables/ttc_regression_results.csv")

# ─────────────────────────────────────────────────────────────
# 22. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
n_day1_appr = len(raw_data_files[LABEL_APPROACHING][0])
n_day2_appr = sum(len(a) for a in raw_data_files[LABEL_APPROACHING][1:])
n_day1_away = len(raw_data_files[LABEL_AWAY][0])
n_day2_away = sum(len(a) for a in raw_data_files[LABEL_AWAY][1:])
n_day1_stat = sum(len(a) for a in raw_data_files[LABEL_STATIONARY][:2])
n_day2_stat = sum(len(a) for a in raw_data_files[LABEL_STATIONARY][2:])

print("\n" + "=" * 65)
print("AWY-1  FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset — Approaching : {len(raw_data[LABEL_APPROACHING]):,} readings "
      f"(Day1={n_day1_appr:,} + Day2={n_day2_appr:,})")
print(f"          — Away        : {len(raw_data[LABEL_AWAY]):,} readings "
      f"(Day1={n_day1_away:,} + Day2={n_day2_away:,})")
print(f"          — Stationary  : {len(raw_data[LABEL_STATIONARY]):,} readings "
      f"(Day1={n_day1_stat:,} + Day2={n_day2_stat:,})")
print(f"  Zone threshold         : {ZONE_THRESHOLD} m ({ZONE_THRESHOLD*39.37:.1f} in)")
print(f"  Window / stride (train): {WINDOW_SIZE} / {STRIDE_TRAIN} readings")
print()
print("  CLASSIFICATION ACCURACY")
print(f"    Threshold-only (B1)  : {threshold_acc*100:.2f}%")
print(f"    Trend-based    (B2)  : {trend_acc*100:.2f}%")
print(f"    Logistic Regression  : {lr_clf_acc*100:.2f}%")
print(f"    Decision Tree        : {dt_clf_acc*100:.2f}%")
print(f"    K-Nearest Neighbors  : {knn_clf_acc*100:.2f}%")
print(f"    SVM (RBF)            : {svm_clf_acc*100:.2f}%")
print(f"    Random Forest        : {rf_clf_acc*100:.2f}%")
print()
print("  TtC REGRESSION   (MAE in seconds)")
print(f"    Threshold-only (B1)  : N/A")
print(f"    Trend-based    (B2)  : {trend_ttc_mae:.4f} s")
print(f"    Linear Regression    : {lr_reg_mae:.4f} s")
print(f"    Random Forest        : {rf_reg_mae:.4f} s")
print()
print("  OBJECTIVES")
print(f"    ≥85% accuracy  : {rf_clf_acc*100:.2f}%")
print(f"    MAE < 0.6 s    : {rf_reg_mae:.4f} s")
print(f"MAE  : {lr_reg_mae:.4f} s")
print()
print("  OUTPUT FILES")
print("    results/figures/01_eda_overview.png")
print("    results/figures/02_feature_distributions.png")
print("    results/figures/03_confusion_matrices.png")
print("    results/figures/04_feature_importance.png")
print("    results/figures/05_ttc_scatter.png")
print("    results/figures/06_results_summary.png")
print("    results/figures/07_detailed_performance.png")
print("    results/tables/classification_results.csv")
print("    results/tables/ttc_regression_results.csv")
print("    results/models/rf_classifier.joblib")
print("    results/models/rf_regressor.joblib")
print("=" * 65)
print("Script completed successfully.")