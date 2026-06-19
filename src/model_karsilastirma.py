import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings("ignore")

# ─── 1. Veri Yükleme ─────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.getcwd(), "verilerim", "comprehensive_market_data_200_plus_features.csv")
df = pd.read_csv(DATA_PATH)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

# ─── 2. Hedef ve Özellik Sütunlarını Belirle ──────────────────────────────
target_symbol = "BTC-USD"   # İstediğiniz sembolle değiştirin

if target_symbol not in df.columns:
    raise ValueError(f"'{target_symbol}' sütunu bulunamadı. Mevcut sütunlar: {df.columns[:10]}")

feature_cols = [c for c in df.columns if c != target_symbol]
X = df[feature_cols].copy()
y = df[target_symbol].copy()

# ─── 3. Eksik Değerleri Temizle ─────────────────────────────────────────────
X = X.ffill().bfill()
y = y.ffill().bfill()

# ─── 4. Eğitim / Test Ayrımı (Metindeki gibi: %83 / %17, random_state=35) ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.17, random_state=35, shuffle=False
)

# ─── 5. Modelleri Tanımla (DÜZELTİLDİ) ─────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=3000,
        random_state=42,
        n_jobs=-1
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=50,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=7000,
        learning_rate=0.005,
        max_depth=7,
        subsample=0.9,
        min_samples_split=3,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        n_iter_no_change=10,        # Erken durdurma için
        validation_fraction=0.2     # Erken durdurma için doğrulama oranı
    )
}

# ─── 6. Modelleri Eğit ve Metrikleri Hesapla ──────────────────────────────
results = {"Model": [], "Set": [], "R²": [], "MSE": [], "MAE": []}

for name, model in models.items():
    print(f"\n{name} eğitiliyor...")
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    train_r2  = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_r2  = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results["Model"].extend([name, name])
    results["Set"].extend(["Eğitim", "Test"])
    results["R²"].extend([train_r2, test_r2])
    results["MSE"].extend([train_mse, test_mse])
    results["MAE"].extend([train_mae, test_mae])
    
    print(f"  Eğitim  - R²: {train_r2:.4f}, MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    print(f"  Test    - R²: {test_r2:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

# ─── 7. Karşılaştırma Grafiği (Şekil 3) ────────────────────────────────────
df_results = pd.DataFrame(results)
df_plot = df_results[df_results["Model"].isin(["Random Forest", "Gradient Boosting"])]

fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=130)
fig.patch.set_facecolor("#0d1117")

for ax, metric in zip(axes, ["R²", "MSE", "MAE"]):
    ax.set_facecolor("#161b22")
    
    x_labels = ["RF\nEğitim", "RF\nTest", "GB\nEğitim", "GB\nTest"]
    values = [
        df_plot[(df_plot["Model"]=="Random Forest") & (df_plot["Set"]=="Eğitim")][metric].values[0],
        df_plot[(df_plot["Model"]=="Random Forest") & (df_plot["Set"]=="Test")][metric].values[0],
        df_plot[(df_plot["Model"]=="Gradient Boosting") & (df_plot["Set"]=="Eğitim")][metric].values[0],
        df_plot[(df_plot["Model"]=="Gradient Boosting") & (df_plot["Set"]=="Test")][metric].values[0],
    ]
    colors = ["#58a6ff", "#1f6feb", "#f0883e", "#d2691e"]
    bars = ax.bar(x_labels, values, color=colors, edgecolor="#21262d", linewidth=1.2)
    
    ax.set_title(metric, color="#f0f6fc", fontsize=12, fontweight="600")
    ax.set_ylabel(metric, color="#8b949e", fontsize=10)
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines[:].set_color("#21262d")
    ax.grid(axis="y", color="#21262d", linestyle="--", linewidth=0.5)
    
    for bar, val in zip(bars, values):
        offset = 0.05 * abs(val) if val != 0 else 0.1
        y_pos = bar.get_height() + offset if val >= 0 else bar.get_height() - offset
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:.4f}", ha="center", va=va,
                fontsize=8, color="#f0f6fc", fontweight="bold")

fig.suptitle(f"Random Forest vs Gradient Boosting – Eğitim ve Test Metrikleri\n(Hedef: {target_symbol})", 
             color="#f0f6fc", fontsize=14, fontweight="700", y=1.02)
fig.tight_layout()

# ─── 8. Kaydet ve Göster ────────────────────────────────────────────────────
plt.savefig("sekil3_rf_gb_karsilastirma.png", dpi=300, bbox_inches="tight")
plt.show()

# ─── 9. Tablo ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("📊 Tüm Modellerin Performans Metrikleri (Eğitim ve Test Setleri):")
print("="*60)
print(df_results.round(4).to_string(index=False))

# ─── 10. Yığınlama Denemesi Notu ───────────────────────────────────────────
print("\n" + "="*60)
print("🧪 Yığınlama (Stacking) Denemesi Hakkında Not:")
print("="*60)
print("Metinde belirtildiği gibi, Random Forest tahminlerini girdi alan, ")
print("ikinci seviyede LogisticRegression meta-öğrenicisi kullanılan bir ")
print("stacking yaklaşımı test edilmiştir. Ancak regresyon hedefi için ")
print("sınıflandırma modeli (LogisticRegression) kullanılması metodolojik ")
print("tutarsızlık oluşturduğundan bu yöntem nihai model seçimine dahil ")
print("edilmemiş ve Şekil 3'te yer almamıştır.")