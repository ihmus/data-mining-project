import os
import re
import pickle
import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Finansal Tahmin Paneli", page_icon="📈", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label { color: #8b949e !important; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }
.main .block-container { background: #0d1117; padding-top: 1.5rem; }
body { background-color: #0d1117; }
.metric-card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.1rem 1.4rem; text-align: center; }
.metric-card .metric-label { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: #8b949e; margin-bottom: 0.35rem; }
.metric-card .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #58a6ff; }
.metric-card .metric-sub { font-size: 0.72rem; color: #8b949e; margin-top: 0.2rem; }
.err-box { background: #1f0a0a; border-left: 3px solid #f85149; padding: 0.8rem 1rem; border-radius: 4px; color: #f85149; font-size: 0.88rem; margin: 0.5rem 0; }
.warn-box { background: #1a1500; border-left: 3px solid #e3b341; padding: 0.8rem 1rem; border-radius: 4px; color: #e3b341; font-size: 0.88rem; margin: 0.5rem 0; }
.info-box { background: #0c1929; border-left: 3px solid #58a6ff; padding: 0.8rem 1rem; border-radius: 4px; color: #58a6ff; font-size: 0.88rem; margin: 0.5rem 0; }
h1.page-title { font-size: 1.6rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.02em; margin-bottom: 0; }
.page-subtitle { font-size: 0.85rem; color: #8b949e; margin-top: 0.2rem; margin-bottom: 1.5rem; }
h2.section-title { font-size: 1.1rem; font-weight: 600; color: #f0f6fc; margin-bottom: 0.3rem; margin-top: 0.5rem; }
hr.section-divider { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }
.stDownloadButton > button { background: #238636 !important; color: #fff !important; border: none !important; border-radius: 6px !important; font-weight: 600 !important; font-size: 0.85rem !important; }
.stDownloadButton > button:hover { background: #2ea043 !important; }
.sim-card-profit { background: #0d1f0d; border: 1px solid #238636; border-radius: 8px; padding: 1.1rem 1.4rem; text-align: center; }
.sim-card-loss { background: #1f0a0a; border: 1px solid #f85149; border-radius: 8px; padding: 1.1rem 1.4rem; text-align: center; }
.sim-card-neutral { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.1rem 1.4rem; text-align: center; }
/* İşlem tablosu */
.islem-tablo { width: 100%; border-collapse: collapse; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; }
.islem-tablo th { background: #161b22; color: #8b949e; padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid #21262d; font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; }
.islem-tablo td { padding: 0.45rem 0.8rem; border-bottom: 1px solid #161b22; color: #c9d1d9; }
.islem-tablo tr:hover td { background: #161b22; }
.tag-alim { background: #0d2a0d; color: #3fb950; border-radius: 3px; padding: 2px 7px; font-size: 0.75rem; font-weight: 600; }
.tag-satim { background: #2a0d0d; color: #f85149; border-radius: 3px; padding: 2px 7px; font-size: 0.75rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sabitler ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.getcwd(), "verilerim", "comprehensive_market_data_200_plus_features.csv")
MODELS_ROOT = os.path.join(os.getcwd(), "models")
GUN_SECENEKLERI           = [3, 7, 12, 15, 30]
TRADE_HORIZON_SECENEKLERI = [1, 2, 3, 4, 5, 6, 7]
ISLEM_ESIGI = 0.005   # %0.5

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def sembol_klasor_adi(s: str) -> str:
    return re.sub(r"[-=]", "_", s)

def model_yolu(sembol: str, gun: int) -> str:
    return os.path.join(MODELS_ROOT, sembol_klasor_adi(sembol), str(gun))

def model_dosya_adi(sembol: str, gun: int) -> str:
    return f"{sembol_klasor_adi(sembol)}_predict_{gun}"

@st.cache_resource(show_spinner=False)
def load_model(model_name: str, models_path: str):
    fp = os.path.join(models_path, f"{model_name}.pkl")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Model dosyası bulunamadı:\n{fp}")
    with open(fp, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV dosyası bulunamadı:\n{path}")
    df = pd.read_csv(path)
    for col in df.columns:
        if col == "Date":
            continue
        fv = df[col].first_valid_index()
        if fv is not None and fv > 0:
            df.loc[:fv - 1, col] = 0
        elif fv is None:
            df[col] = 0
        df[col] = df[col].astype(float).interpolate(method="linear").ffill()
    return df

def yuzde_degisim(dizi) -> list:
    """Dizideki günlük % değişimleri hesaplar; ilk eleman için NaN döner."""
    sonuc = [np.nan]
    for i in range(1, len(dizi)):
        prev = dizi[i - 1]
        sonuc.append(((dizi[i] - prev) / prev * 100) if prev != 0 else np.nan)
    return sonuc

def yon_dogrulugu(y_gercek, y_pred) -> dict:
    """
    Gerçek ve tahmin dizileri için yön (artış/düşüş) tahmini başarısını ölçer.
    Her gün için: gerçek yön == tahmin yönü mü?
    """
    gercek_yon  = np.diff(y_gercek)   # pozitif = artış, negatif = düşüş
    tahmin_yon  = np.diff(y_pred)
    eslesme     = np.sign(gercek_yon) == np.sign(tahmin_yon)
    dogru_sayi  = int(eslesme.sum())
    toplam      = len(eslesme)
    return {
        "dogru": dogru_sayi,
        "toplam": toplam,
        "oran": (dogru_sayi / toplam * 100) if toplam > 0 else 0.0,
    }

# ── Tahmin fonksiyonu ─────────────────────────────────────────────────────────

def tahmin_yap(df: pd.DataFrame, sembol: str, gun: int, test_modu: bool):
    """
    (dates, y_pred, y_gercek|None, metrics, hata|None) döndürür.
    metrics artık 'yon' anahtarını da içerir (test modunda).
    """
    if sembol not in df.columns:
        return None, None, None, {}, f"'{sembol}' sütunu CSV'de bulunamadı."

    m_path = model_yolu(sembol, gun)
    m_name = model_dosya_adi(sembol, gun)
    try:
        model = load_model(m_name, m_path)
    except FileNotFoundError as e:
        return None, None, None, {}, str(e)
    except Exception as e:
        return None, None, None, {}, f"Model yüklenirken hata: {e}"

    try:
        ftrs = model.feature_names_in_.tolist()
    except AttributeError:
        return None, None, None, {}, "Model 'feature_names_in_' özelliğini desteklemiyor."

    missing = [f for f in ftrs if f not in df.columns]
    if missing:
        return None, None, None, {}, f"Modelde {len(missing)} eksik özellik: {missing[:5]}…"

    n = len(df)

    # ── Test modu ─────────────────────────────────────────────────────────────
    if test_modu:
        if gun > n:
            return None, None, None, {}, f"Seçilen gün ({gun}) > CSV satır sayısı ({n})."
        test_df  = df.iloc[-gun:]
        y_gercek = test_df[sembol].to_numpy()
        try:
            y_pred = model.predict(test_df[ftrs])
        except Exception as e:
            return None, None, None, {}, f"Tahmin hatası: {e}"
        dates = [str(d).split(" ")[0] for d in df["Date"].values[-gun:]]

        metrics = {}
        if len(y_gercek) > 1:
            metrics["R²"]  = r2_score(y_gercek, y_pred)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_gercek, y_pred))
            metrics["MAE"]  = float(np.mean(np.abs(y_gercek - y_pred)))
            metrics["yon"]  = yon_dogrulugu(y_gercek, y_pred)
        return dates, y_pred, y_gercek, metrics, None

    # ── İleriye dönük tahmin ──────────────────────────────────────────────────
    else:
        last_row = df[ftrs].iloc[[-1]]
        repeated = pd.concat([last_row] * gun, ignore_index=True)
        try:
            y_pred = model.predict(repeated)
        except Exception as e:
            return None, None, None, {}, f"Tahmin hatası: {e}"
        try:
            last_date    = pd.to_datetime(df["Date"].iloc[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=gun)
            dates        = [d.strftime("%Y-%m-%d") for d in future_dates]
        except Exception:
            dates = [f"T+{i+1}" for i in range(gun)]
        return dates, y_pred, None, {}, None

# ── Tahmin grafiği ────────────────────────────────────────────────────────────

def _etiket_ekle(ax, x_arr, y_arr, delta_arr, base_color, offset_y=6, fontsize=6.5):
    """
    Her noktaya fiyat + % değişim etiketi koyar.
    Pozitif delta → yeşil, negatif → kırmızı, NaN → base_color.
    """
    for i, (xi, yi, delta) in enumerate(zip(x_arr, y_arr, delta_arr)):
        if np.isnan(delta):
            label = f"{yi:.2f}"
            color = base_color
        else:
            sign  = "+" if delta >= 0 else ""
            label = f"{yi:.2f}\n{sign}{delta:.1f}%"
            color = "#3fb950" if delta >= 0 else "#f85149"
        ax.annotate(
            label,
            (xi, yi),
            fontsize=fontsize,
            color=color,
            ha="center",
            va="bottom",
            xytext=(0, offset_y),
            textcoords="offset points",
            multialignment="center",
        )

def grafik_olustur(sembol, gun, dates, y_pred, y_gercek, metrics, test_modu) -> plt.Figure:
    """
    Ana tahmin grafiği.
    - Test modu: Gerçek + Tahmin çizgisi; her ikisi için renkli % değişim etiketleri.
      Alt panelde: Gerçek vs Tahmin yüzde değişim çubuğu karşılaştırması.
    - İleri tahmin modu: Tahmin çizgisi + renkli % değişim etiketleri.
    """
    x = np.arange(len(dates))

    if test_modu and y_gercek is not None:
        # 3 panelli grafik: fiyat | yön karşılaştırma çubuğu | yön farkı
        fig, (ax_main, ax_bar) = plt.subplots(
            2, 1, figsize=(13, 8), dpi=130,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
        fig.patch.set_facecolor("#0d1117")

        # ── Üst panel: fiyat çizgileri ────────────────────────────────────────
        ax_main.set_facecolor("#161b22")
        ax_main.plot(x, y_gercek, color="#58a6ff", linewidth=2.2, label="Gerçek", zorder=3)
        ax_main.fill_between(x, y_gercek, alpha=0.07, color="#58a6ff")
        ax_main.plot(x, y_pred, color="#f0883e", linewidth=1.8, linestyle="--", label="Tahmin", zorder=4)

        # Renkli % değişim etiketleri – gerçek değerler (yukarıya)
        delta_gercek = yuzde_degisim(y_gercek)
        _etiket_ekle(ax_main, x, y_gercek, delta_gercek, "#58a6ff", offset_y=7)

        # Renkli % değişim etiketleri – tahmin değerleri (aşağıya)
        delta_pred = yuzde_degisim(y_pred)
        for i, (xi, yi, delta) in enumerate(zip(x, y_pred, delta_pred)):
            if np.isnan(delta):
                label, color = f"{yi:.2f}", "#f0883e"
            else:
                sign  = "+" if delta >= 0 else ""
                label = f"{yi:.2f}\n{sign}{delta:.1f}%"
                color = "#3fb950" if delta >= 0 else "#f85149"
            ax_main.annotate(
                label, (xi, yi), fontsize=6.2, color=color,
                ha="center", va="top",
                xytext=(0, -7), textcoords="offset points",
                multialignment="center",
            )

        r2_str   = f"R²={metrics.get('R²', float('nan')):.4f}"
        rmse_str = f"RMSE={metrics.get('RMSE', float('nan')):.4f}"
        yon      = metrics.get("yon", {})
        yon_str  = f"Yön Doğruluğu={yon.get('oran', 0):.1f}%" if yon else ""
        ax_main.set_title(
            f"{sembol}  ·  {gun} Günlük Test   {r2_str}  |  {rmse_str}  |  {yon_str}",
            color="#f0f6fc", fontsize=10.5, fontweight="600", pad=12,
        )
        ax_main.set_xticks(x)
        ax_main.set_xticklabels(dates, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
        ax_main.tick_params(axis="y", colors="#8b949e", labelsize=8)
        ax_main.spines[:].set_color("#21262d")
        ax_main.grid(True, color="#21262d", linewidth=0.6, linestyle="--")
        ax_main.set_ylabel("Fiyat", color="#8b949e", fontsize=9)
        ax_main.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", loc="upper left")

        # ── Alt panel: Gerçek vs Tahmin % değişim bar karşılaştırması ─────────
        ax_bar.set_facecolor("#161b22")
        # İlk eleman NaN, 1. indeksten başla
        bar_x       = x[1:]
        bar_gercek  = np.array(delta_gercek[1:], dtype=float)
        bar_pred    = np.array(delta_pred[1:],   dtype=float)

        width = 0.38
        bar_x_g = bar_x - width / 2
        bar_x_p = bar_x + width / 2

        # Gerçek değişim çubukları – pozitif/negatif renklendir
        for bx, bv in zip(bar_x_g, bar_gercek):
            color = "#58a6ff" if bv >= 0 else "#7d8590"
            ax_bar.bar(bx, bv, width=width, color=color, alpha=0.85, zorder=3)

        # Tahmin değişim çubukları
        for bx, bv in zip(bar_x_p, bar_pred):
            color = "#f0883e" if bv >= 0 else "#6e3010"
            ax_bar.bar(bx, bv, width=width, color=color, alpha=0.85, zorder=3)

        ax_bar.axhline(0, color="#484f58", linewidth=0.8)
        ax_bar.set_xticks(x[1:])
        ax_bar.set_xticklabels(dates[1:], rotation=45, ha="right", fontsize=7, color="#8b949e")
        ax_bar.tick_params(axis="y", colors="#8b949e", labelsize=7.5)
        ax_bar.spines[:].set_color("#21262d")
        ax_bar.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")
        ax_bar.set_ylabel("% Değişim", color="#8b949e", fontsize=8.5)
        ax_bar.set_title("Günlük % Değişim Karşılaştırması  (🔵 Gerçek · 🟠 Tahmin)", color="#f0f6fc", fontsize=9.5, fontweight="600", pad=6)

    else:
        # ── İleriye dönük tek panel ────────────────────────────────────────────
        fig, ax_main = plt.subplots(figsize=(12, 5), dpi=130)
        fig.patch.set_facecolor("#0d1117")
        ax_main.set_facecolor("#161b22")

        ax_main.plot(x, y_pred, color="#3fb950", linewidth=2.2, label="İleriye Tahmin", zorder=3)
        ax_main.fill_between(x, y_pred, alpha=0.08, color="#3fb950")

        delta_pred = yuzde_degisim(y_pred)
        _etiket_ekle(ax_main, x, y_pred, delta_pred, "#3fb950", offset_y=6)

        ax_main.set_title(
            f"{sembol}  ·  {gun} Günlük İleriye Tahmin",
            color="#f0f6fc", fontsize=11, fontweight="600", pad=12,
        )
        ax_main.set_xticks(x)
        ax_main.set_xticklabels(dates, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
        ax_main.tick_params(axis="y", colors="#8b949e", labelsize=8)
        ax_main.spines[:].set_color("#21262d")
        ax_main.grid(True, color="#21262d", linewidth=0.6, linestyle="--")
        ax_main.set_xlabel("Tarih", color="#8b949e", fontsize=9)
        ax_main.set_ylabel("Fiyat", color="#8b949e", fontsize=9)
        ax_main.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", loc="upper left")

    fig.tight_layout(pad=1.8)
    return fig

def tahmin_grafik_sadece(sembol, gun, dates, y_pred, islem_listesi) -> plt.Figure:
    """
    Sadece model tahminlerini ve al/sat işaretlerini gösteren grafik.
    """
    x = np.arange(len(dates))
    fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(x, y_pred, color="#f0883e", linewidth=2.2, label="Model Tahmini", zorder=3)
    ax.fill_between(x, y_pred, alpha=0.08, color="#f0883e")

    # Günlük % değişim etiketleri
    delta_pred = yuzde_degisim(y_pred)
    for i, (xi, yi, delta) in enumerate(zip(x, y_pred, delta_pred)):
        if np.isnan(delta):
            label, color = f"{yi:.2f}", "#f0883e"
        else:
            sign = "+" if delta >= 0 else ""
            label = f"{yi:.2f}\n{sign}{delta:.1f}%"
            color = "#3fb950" if delta >= 0 else "#f85149"
        ax.annotate(label, (xi, yi), fontsize=6.5, color=color,
                    ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points", multialignment="center")

    # Al/Sat işaretleri
    for islem in islem_listesi:
        idx = islem["idx"]
        if idx >= len(y_pred):
            continue
        tahmin_degeri = y_pred[idx]
        if islem["tur"] == "ALIM":
            ax.scatter(idx, tahmin_degeri, color="#3fb950", s=100, zorder=10,
                       marker="^", edgecolor="white", linewidth=0.5)
            ax.annotate(f"AL\n${tahmin_degeri:.2f}", (idx, tahmin_degeri),
                        fontsize=7, color="#3fb950", ha="center", va="bottom",
                        xytext=(0, 10), textcoords="offset points")
        else:
            ax.scatter(idx, tahmin_degeri, color="#f85149", s=100, zorder=10,
                       marker="v", edgecolor="white", linewidth=0.5)
            ax.annotate(f"SAT\n${tahmin_degeri:.2f}", (idx, tahmin_degeri),
                        fontsize=7, color="#f85149", ha="center", va="top",
                        xytext=(0, -10), textcoords="offset points")

    ax.set_title(f"{sembol}  ·  {gun} Günlük Model Tahminleri ve Sinyaller",
                 color="#f0f6fc", fontsize=11, fontweight="600", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
    ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    ax.grid(True, color="#21262d", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Tarih", color="#8b949e", fontsize=9)
    ax.set_ylabel("Fiyat", color="#8b949e", fontsize=9)
    ax.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9")

    fig.tight_layout()
    return fig

# ── Alım-Satım Simülatörü ─────────────────────────────────────────────────────

def run_simulation(
    df: pd.DataFrame,
    sembol: str,
    model_gun: int,
    trade_horizon: int,
    test_days: int,
    initial_balance: float,
) -> dict:
    """
    Adım adım alım-satım simülasyonu (test penceresi üzerinde).

    Karar mantığı her t gününde:
      - tahmin > mevcut*(1+eşik)  → ALIM  (nakit yoksa bekle)
      - tahmin < mevcut*(1-eşik)  → SATIM (varlık yoksa bekle)
      - aksi                       → BEKLE

    Döndürülen dict içinde:
      tarihler, portfoy_gecmisi, islem_listesi (genişletilmiş),
      metrikler ve tabloya hazır veri.
    """
    if sembol not in df.columns:
        return {"hata": f"'{sembol}' sütunu bulunamadı."}
    n_total = len(df)
    if test_days > n_total:
        return {"hata": f"Test günü ({test_days}) > toplam veri ({n_total})."}

    test_df = df.iloc[-test_days:].reset_index(drop=True)
    n_test  = len(test_df)

    if trade_horizon >= n_test:
        return {"hata": f"İşlem ufku ({trade_horizon}) >= test verisi ({n_test})."}

    # Model yükle
    try:
        model = load_model(model_dosya_adi(sembol, model_gun), model_yolu(sembol, model_gun))
    except FileNotFoundError as e:
        return {"hata": str(e)}
    except Exception as e:
        return {"hata": f"Model yüklenemedi: {e}"}

    try:
        ftrs = model.feature_names_in_.tolist()
    except AttributeError:
        return {"hata": "Model feature_names_in_ desteklemiyor."}

    missing = [f for f in ftrs if f not in test_df.columns]
    if missing:
        return {"hata": f"Eksik özellikler ({len(missing)}): {missing[:5]}…"}

    # ── Simülasyon değişkenleri ───────────────────────────────────────────────
    bakiye             = float(initial_balance)
    varlik_adet        = 0.0
    pozisyon           = "NAKIT"
    satin_alma_maliyet = 0.0   # son alım toplam maliyeti
    satin_alma_fiyat   = 0.0   # son alım birim fiyatı

    portfoy_gecmisi = []
    tarihler        = []
    # İşlem listesi (genişletilmiş): dict listesi
    islem_listesi   = []

    for t in range(n_test - trade_horizon):
        mevcut_fiyat  = float(test_df[sembol].iloc[t])
        tarih         = str(test_df["Date"].iloc[t]).split(" ")[0]
        X_now         = test_df[ftrs].iloc[[t]]

        try:
            tahmin_fiyat = float(model.predict(X_now)[0])
        except Exception:
            tahmin_fiyat = mevcut_fiyat

        varlik_degeri  = varlik_adet * mevcut_fiyat
        portfoy_degeri = bakiye + varlik_degeri
        tarihler.append(tarih)
        portfoy_gecmisi.append(portfoy_degeri)

        ust = mevcut_fiyat * (1 + ISLEM_ESIGI)
        alt = mevcut_fiyat * (1 - ISLEM_ESIGI)

        if tahmin_fiyat > ust and pozisyon == "NAKIT" and bakiye > 0:
            # ALIM
            varlik_adet        = bakiye / mevcut_fiyat
            satin_alma_maliyet = bakiye
            satin_alma_fiyat   = mevcut_fiyat
            bakiye             = 0.0
            pozisyon           = "VARLIK"
            islem_listesi.append({
                "idx": t, "tarih": tarih,
                "tur": "ALIM", "fiyat": mevcut_fiyat,
                "tahmin": tahmin_fiyat,
                "portfoy": portfoy_degeri,
                "kar_zarar": None,   # satışta belirlenecek
            })

        elif tahmin_fiyat < alt and pozisyon == "VARLIK" and varlik_adet > 0:
            # SATIM
            satis_geliri = varlik_adet * mevcut_fiyat
            kar_zarar    = satis_geliri - satin_alma_maliyet
            bakiye       = satis_geliri
            varlik_adet  = 0.0
            pozisyon     = "NAKIT"
            islem_listesi.append({
                "idx": t, "tarih": tarih,
                "tur": "SATIM", "fiyat": mevcut_fiyat,
                "tahmin": tahmin_fiyat,
                "portfoy": portfoy_degeri,
                "kar_zarar": kar_zarar,
                "alim_fiyat": satin_alma_fiyat,
            })

    # Son portföy değeri
    son_fiyat       = float(test_df[sembol].iloc[n_test - trade_horizon])
    son_portfoy     = bakiye + varlik_adet * son_fiyat
    net_kar         = son_portfoy - initial_balance
    getiri_yuzdesi  = net_kar / initial_balance * 100

    alim_sayisi  = sum(1 for i in islem_listesi if i["tur"] == "ALIM")
    satim_sayisi = sum(1 for i in islem_listesi if i["tur"] == "SATIM")

    satislar      = [i for i in islem_listesi if i["tur"] == "SATIM"]
    kazanilan     = sum(1 for i in satislar if (i["kar_zarar"] or 0) > 0)
    kazanma_orani = (kazanilan / len(satislar) * 100) if satislar else 0.0

    # Buy & Hold
    ilk_fiyat  = float(test_df[sembol].iloc[0])
    bh_son     = (initial_balance / ilk_fiyat) * son_fiyat
    bh_getiri  = (bh_son - initial_balance) / initial_balance * 100

    # Drawdown hesaplama
    peak = max(portfoy_gecmisi) if portfoy_gecmisi else initial_balance
    drawdown = (peak - son_portfoy) / peak * 100 if peak > 0 else 0.0

    return {
        "hata": None,
        "tarihler": tarihler,
        "portfoy_gecmisi": portfoy_gecmisi,
        "islem_listesi": islem_listesi,
        "son_portfoy": son_portfoy,
        "net_kar": net_kar,
        "getiri_yuzdesi": getiri_yuzdesi,
        "alim_sayisi": alim_sayisi,
        "satim_sayisi": satim_sayisi,
        "toplam_islem": alim_sayisi + satim_sayisi,
        "kazanma_orani": kazanma_orani,
        "bh_getiri": bh_getiri,
        "bh_son_deger": bh_son,
        "gercek_fiyatlar": test_df[sembol].iloc[:n_test - trade_horizon].tolist(),
        "initial_balance": initial_balance,
        "drawdown": drawdown,
    }

# ── Periyot Karşılaştırma ────────────────────────────────────────────────────

def periyot_karsilastirma(df: pd.DataFrame, sembol: str, test_days: int,
                          trade_horizon: int, initial_balance: float) -> dict:
    """
    Tüm periyotlar (3,7,12,15,30) için simülasyon çalıştırır.
    Sonuçları ve grafik için verileri döndürür.
    """
    periods = [3, 7, 12, 15, 30]
    results = {}
    errors = []

    for period in periods:
        # Test için gün sayısı period'dan az olmamalı, en az period kadar olmalı
        if test_days < period:
            errors.append(f"Test günü ({test_days}) {period} günlük model için yetersiz. En az {period} olmalı.")
            continue

        sim = run_simulation(
            df=df,
            sembol=sembol,
            model_gun=period,
            trade_horizon=trade_horizon,
            test_days=test_days,
            initial_balance=initial_balance
        )
        if sim.get("hata"):
            errors.append(f"{period} gün: {sim['hata']}")
            results[period] = None
        else:
            results[period] = {
                "net_kar": sim["net_kar"],
                "getiri_yuzdesi": sim["getiri_yuzdesi"],
                "islem_sayisi": sim["toplam_islem"],
                "kazanma_orani": sim["kazanma_orani"],
            }

    return {"results": results, "errors": errors}

def karsilastirma_grafik_olustur(results: dict, baslangic_bakiye: float, sembol: str = "", test_days: int = 0) -> plt.Figure:
    """
    Periyot bazında net kar/zarar çubuk grafiği oluşturur.
    """
    periods = sorted([p for p, v in results.items() if v is not None])
    if not periods:
        return None

    net_karlar = [results[p]["net_kar"] for p in periods]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    colors = ["#3fb950" if x >= 0 else "#f85149" for x in net_karlar]
    bars = ax.bar(periods, net_karlar, color=colors, edgecolor="#21262d", linewidth=1.2)

    # Değer etiketleri
    for bar, val in zip(bars, net_karlar):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.05 * max(abs(min(net_karlar)), abs(max(net_karlar))) + 1),
                f"{val:,.2f} $",
                ha="center", va="bottom", color="#c9d1d9", fontsize=9, fontweight="bold")

    ax.axhline(0, color="#484f58", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Tahmin Periyodu (gün)", color="#8b949e", fontsize=10)
    ax.set_ylabel("Net Kar / Zarar (USD)", color="#8b949e", fontsize=10)
    ax.set_title(f"Farklı Periyotlar İçin Simülasyon Sonuçları\n(Sembol: {sembol}, Test Günü: {test_days})",
                 color="#f0f6fc", fontsize=11, fontweight="600")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    ax.grid(axis="y", color="#21262d", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig

# ── Simülasyon grafiği ────────────────────────────────────────────────────────

def sim_grafik_olustur(sim_sonuc: dict) -> plt.Figure:
    """
    3 panelli simülasyon grafiği:
      - Üst   : Portföy değeri + alım(▲)/satım(▼) işaretleri + fiyat/etiket
      - Orta  : Gerçek fiyat çizgisi + al/sat noktaları
      - Alt   : Strateji vs Buy&Hold bar karşılaştırması
    """
    tarihler    = sim_sonuc["tarihler"]
    portfoy     = sim_sonuc["portfoy_gecmisi"]
    islemler    = sim_sonuc["islem_listesi"]
    fiyatlar    = sim_sonuc["gercek_fiyatlar"]
    init_bal    = sim_sonuc["initial_balance"]
    bh_son      = sim_sonuc["bh_son_deger"]
    son_portfoy = sim_sonuc["son_portfoy"]
    n           = len(tarihler)

    fig = plt.figure(figsize=(13, 11), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    gs  = fig.add_gridspec(3, 1, height_ratios=[2.8, 2, 1], hspace=0.55)
    ax1 = fig.add_subplot(gs[0])   # portföy değeri
    ax2 = fig.add_subplot(gs[1])   # gerçek fiyat + sinyaller
    ax3 = fig.add_subplot(gs[2])   # strateji karşılaştırma

    x = np.arange(n)

    # Tarih eti̇keti̇ ayarı (max 10 göster)
    step          = max(1, n // 10)
    tick_pos      = list(range(0, n, step))
    tick_labels   = [tarihler[i] for i in tick_pos]

    def _ax_style(ax, ylabel=""):
        ax.set_facecolor("#161b22")
        ax.spines[:].set_color("#21262d")
        ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
        ax.grid(True, color="#21262d", linewidth=0.6, linestyle="--")
        if ylabel:
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=8.5)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7, color="#8b949e")

    # ── Panel 1: Portföy değeri ───────────────────────────────────────────────
    ax1.plot(x, portfoy, color="#58a6ff", linewidth=2.2, label="Portföy Değeri", zorder=3)
    ax1.fill_between(x, portfoy, alpha=0.07, color="#58a6ff")
    ax1.axhline(y=init_bal, color="#484f58", linewidth=1.0, linestyle="--", label="Başlangıç")
    _ax_style(ax1, "Portföy (USD)")
    ax1.set_title("Portföy Değeri  ·  ▲ Alım (Yeşil)  ▼ Satım (Kırmızı)", color="#f0f6fc", fontsize=10, fontweight="600", pad=8)

    for ish in islemler:
        idx = ish["idx"]
        if idx >= n:
            continue
        pv  = portfoy[idx]
        if ish["tur"] == "ALIM":
            ax1.scatter([idx], [pv], color="#3fb950", s=80, zorder=6, marker="^")
            ax1.annotate(
                f"AL\n${ish['fiyat']:,.2f}",
                (idx, pv), fontsize=6.5, color="#3fb950",
                ha="center", va="top", xytext=(0, -16),
                textcoords="offset points", multialignment="center",
            )
        else:
            kz   = ish.get("kar_zarar") or 0
            renk = "#3fb950" if kz >= 0 else "#f85149"
            ax1.scatter([idx], [pv], color="#f85149", s=80, zorder=6, marker="v")
            ax1.annotate(
                f"SAT\n${ish['fiyat']:,.2f}\n{'+' if kz>=0 else ''}{kz:,.2f}$",
                (idx, pv), fontsize=6.5, color=renk,
                ha="center", va="bottom", xytext=(0, 12),
                textcoords="offset points", multialignment="center",
            )

    ax1.legend(fontsize=8, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", loc="upper left")

    # ── Panel 2: Gerçek fiyat + al/sat sinyalleri ────────────────────────────
    fiyat_arr = np.array(fiyatlar, dtype=float)
    ax2.plot(x[:len(fiyat_arr)], fiyat_arr, color="#c9d1d9", linewidth=1.6, label="Gerçek Fiyat", zorder=3)
    ax2.fill_between(x[:len(fiyat_arr)], fiyat_arr, alpha=0.05, color="#c9d1d9")
    _ax_style(ax2, "Fiyat")
    ax2.set_title("Gerçek Fiyat  ·  AL / SAT Sinyalleri", color="#f0f6fc", fontsize=10, fontweight="600", pad=8)

    for ish in islemler:
        idx = ish["idx"]
        if idx >= len(fiyat_arr):
            continue
        fp = fiyat_arr[idx]
        if ish["tur"] == "ALIM":
            ax2.scatter([idx], [fp], color="#3fb950", s=90, zorder=5, marker="^")
            ax2.annotate(
                f"AL\n${fp:,.2f}", (idx, fp), fontsize=6.5, color="#3fb950",
                ha="center", va="top", xytext=(0, -14),
                textcoords="offset points", multialignment="center",
            )
        else:
            kz   = ish.get("kar_zarar") or 0
            renk = "#3fb950" if kz >= 0 else "#f85149"
            ax2.scatter([idx], [fp], color="#f85149", s=90, zorder=5, marker="v")
            ax2.annotate(
                f"SAT\n${fp:,.2f}", (idx, fp), fontsize=6.5, color=renk,
                ha="center", va="bottom", xytext=(0, 10),
                textcoords="offset points", multialignment="center",
            )

    ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#21262d", labelcolor="#c9d1d9", loc="upper left")

    # ── Panel 3: Bar karşılaştırması ──────────────────────────────────────────
    ax3.set_facecolor("#161b22")
    kategoriler = ["Başlangıç", "Al-Sat Strateji", "Buy & Hold"]
    degerler    = [init_bal, son_portfoy, bh_son]
    renkler     = ["#484f58", "#3fb950" if son_portfoy >= init_bal else "#f85149", "#f0883e"]
    bars        = ax3.bar(kategoriler, degerler, color=renkler, width=0.38, zorder=3)
    for bar, val in zip(bars, degerler):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(degerler) * 0.01,
            f"${val:,.2f}",
            ha="center", va="bottom", fontsize=8.5, color="#c9d1d9", fontfamily="monospace",
        )
    ax3.spines[:].set_color("#21262d")
    ax3.tick_params(colors="#8b949e", labelsize=8)
    ax3.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--")
    ax3.set_title("Strateji Karşılaştırması", color="#f0f6fc", fontsize=9.5, fontweight="600", pad=6)
    ax3.set_ylabel("Değer (USD)", color="#8b949e", fontsize=8.5)

    fig.tight_layout()
    return fig

def fig_to_bytes(fig: plt.Figure, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

# ── İşlem tablosu HTML ────────────────────────────────────────────────────────

def islem_tablosu_html(islem_listesi: list) -> str:
    """Simülasyon işlemlerini HTML tablo olarak döndürür."""
    if not islem_listesi:
        return "<p style='color:#484f58;font-size:0.85rem;'>Hiçbir işlem gerçekleşmedi.</p>"

    satirlar = ""
    for ish in islem_listesi:
        tur_html = (
            f"<span class='tag-alim'>▲ ALIM</span>"
            if ish["tur"] == "ALIM"
            else f"<span class='tag-satim'>▼ SATIM</span>"
        )
        kz = ish.get("kar_zarar")
        if kz is None:
            kz_html = "<span style='color:#484f58;'>—</span>"
        else:
            renk   = "#3fb950" if kz >= 0 else "#f85149"
            kz_html = f"<span style='color:{renk};'>{'+' if kz>=0 else ''}{kz:,.2f} $</span>"

        al_fiyat = ish.get("alim_fiyat")
        al_html  = f"${al_fiyat:,.2f}" if al_fiyat else "—"

        satirlar += (
            f"<tr>"
            f"<td>{ish['tarih']}</td>"
            f"<td>{tur_html}</td>"
            f"<td>${ish['fiyat']:,.4f}</td>"
            f"<td>{al_html}</td>"
            f"<td>${ish['tahmin']:,.4f}</td>"
            f"<td>{kz_html}</td>"
            f"<td>${ish['portfoy']:,.2f}</td>"
            f"</tr>"
        )

    return (
        "<div style='overflow-x:auto;'>"
        "<table class='islem-tablo'>"
        "<thead><tr>"
        "<th>Tarih</th><th>İşlem</th><th>Fiyat</th>"
        "<th>Alım Fiyatı</th><th>Tahmin</th><th>Kar/Zarar</th><th>Portföy</th>"
        "</tr></thead>"
        f"<tbody>{satirlar}</tbody>"
        "</table></div>"
    )

# ── Session state başlatma ────────────────────────────────────────────────────
for key, val in [
    ("result_ready", False), ("fig", None), ("metrics", {}), ("hata", None),
    ("sembol", "BTC-USD"), ("gun", 7), ("test_modu", True),
    ("sim_result_ready", False), ("sim_fig", None), ("sim_sonuc", None),
    ("tahmin_fig", None), ("tahmin_fig_sadece", None),
    ("karsilastirma_fig", None), ("karsilastirma_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#f0f6fc;margin-bottom:0.5rem;'>📈 Tahmin Paneli</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

    sembol_input = st.text_input(
        "Sembol", value=st.session_state["sembol"],
        placeholder="örn: BTC-USD", help="Yahoo Finance formatında sembol girin.",
    )
    gun_input = st.selectbox(
        "Tahmin Günü", options=GUN_SECENEKLERI,
        index=GUN_SECENEKLERI.index(st.session_state["gun"])
               if st.session_state["gun"] in GUN_SECENEKLERI else 1,
    )
    test_modu_input = st.checkbox(
        "Test / Simülasyon Modu", value=st.session_state["test_modu"],
        help=(
            "✅ Açık: CSV'nin son N satırını test seti olarak kullanır, "
            "gerçek değerlerle karşılaştırır.\n\n"
            "⬜ Kapalı: Bugünden itibaren N gün ileriye tahmin üretir."
        ),
    )
    st.markdown("<hr style='border-color:#21262d;margin:1rem 0;'>", unsafe_allow_html=True)
    calistir = st.button("▶ Tahmini Çalıştır", use_container_width=True, type="primary")

    # ── Simülatör bölümü ─────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#21262d;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.9rem;font-weight:600;color:#f0f6fc;margin-bottom:0.6rem;'>🤖 Alım-Satım Simülatörü</div>",
        unsafe_allow_html=True,
    )
    baslangic_bakiye = st.number_input(
        "Başlangıç Bakiyesi (USD)", min_value=100.0, max_value=10_000_000.0,
        value=10_000.0, step=500.0, format="%.2f",
        help="Simülasyona başlamak için kullanılacak nakit.",
    )
    trade_horizon_input = st.selectbox(
        "İşlem Ufku (Gün)", options=TRADE_HORIZON_SECENEKLERI, index=1,
        help="Model bu kadar gün sonrasını tahmin ederek alım/satım kararı alır.",
    )
    sim_calistir = st.button(
        "📊 Simülasyonu Çalıştır", use_container_width=True,
        disabled=not test_modu_input, type="secondary",
        help="Simülatör yalnızca Test Modu aktifken çalışır.",
    )
    if not test_modu_input:
        st.markdown(
            "<div style='font-size:0.75rem;color:#484f58;margin-top:0.3rem;'>"
            "⚠️ Simülatör için Test Modunu açın.</div>",
            unsafe_allow_html=True,
        )

    # ── Periyot Karşılaştırma Butonu ──────────────────────────────────────
    st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0;'>", unsafe_allow_html=True)
    kar_buton = st.button(
        "📊 Tüm Periyotları Karşılaştır",
        use_container_width=True,
        disabled=not test_modu_input,
        help="3, 7, 12, 15, 30 günlük modelleri aynı test verisiyle simüle eder ve karşılaştırma grafiği oluşturur."
    )
    if not test_modu_input:
        st.markdown(
            "<div style='font-size:0.75rem;color:#484f58;margin-top:0.3rem;'>⚠️ Karşılaştırma için Test Modunu açın.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#484f58;margin-top:1rem;line-height:1.6;'>"
        "Model dizini:<br><code style='color:#6e7681;'>models/{SEMBOL}/{GÜN}/</code>"
        "<br>Veri dosyası:<br><code style='color:#6e7681;'>verilerim/comprehensive_market_data.csv</code>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Ana alan ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 class='page-title'>Finansal Tahmin Paneli</h1>"
    "<p class='page-subtitle'>ML modeli ile fiyat tahmini · Test & İleriye Dönük Mod · Alım-Satım Simülatörü</p>",
    unsafe_allow_html=True,
)

# Veri yükle
try:
    df = load_data(DATA_PATH)
    data_tamam = True
except FileNotFoundError as e:
    st.markdown(f"<div class='err-box'>🚨 {e}</div>", unsafe_allow_html=True)
    data_tamam = False
except Exception as e:
    st.markdown(f"<div class='err-box'>🚨 CSV yüklenirken hata: {e}</div>", unsafe_allow_html=True)
    data_tamam = False

if data_tamam:
    ci1, ci2, ci3 = st.columns(3)
    try:
        son_tarih = pd.to_datetime(df["Date"].iloc[-1]).strftime("%d %b %Y")
    except Exception:
        son_tarih = str(df["Date"].iloc[-1])
    for col, label, val, sub in [
        (ci1, "Toplam Kayıt",  f"{len(df):,}",          "satır"),
        (ci2, "Kolon Sayısı",  f"{len(df.columns):,}",  "özellik"),
        (ci3, "Son Tarih",     son_tarih,                "CSV'deki son gün"),
    ]:
        with col:
            fs = "font-size:1.2rem;" if label == "Son Tarih" else ""
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{label}</div>"
                f"<div class='metric-value' style='{fs}'>{val}</div>"
                f"<div class='metric-sub'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── Tahmin çalıştır ───────────────────────────────────────────────────────────
if calistir and data_tamam:
    sembol    = sembol_input.strip().upper()
    gun       = gun_input
    test_modu = test_modu_input
    st.session_state.update({"sembol": sembol, "gun": gun, "test_modu": test_modu})

    if test_modu and gun > len(df):
        st.markdown(
            f"<div class='warn-box'>⚠️ Seçilen gün ({gun}) > CSV satır sayısı ({len(df)}).</div>",
            unsafe_allow_html=True,
        )
        st.session_state["result_ready"] = False
    else:
        with st.spinner("Model yükleniyor ve tahmin yapılıyor…"):
            dates, y_pred, y_gercek, metrics, hata = tahmin_yap(df, sembol, gun, test_modu)
        if hata:
            st.session_state.update({"hata": hata, "result_ready": False})
        else:
            fig = grafik_olustur(sembol, gun, dates, y_pred, y_gercek, metrics, test_modu)
            st.session_state.update({"fig": fig, "metrics": metrics, "hata": None, "result_ready": True})

# ── Simülasyon çalıştır ───────────────────────────────────────────────────────
if sim_calistir and data_tamam and test_modu_input:
    sembol = sembol_input.strip().upper()
    st.session_state.update({"sembol": sembol, "gun": gun_input, "test_modu": True})

    with st.spinner("Simülasyon çalışıyor…"):
        sim_sonuc = run_simulation(
            df=df, sembol=sembol, model_gun=gun_input,
            trade_horizon=trade_horizon_input, test_days=gun_input,
            initial_balance=baslangic_bakiye,
        )

    if sim_sonuc.get("hata"):
        st.session_state.update({"hata": sim_sonuc["hata"], "sim_result_ready": False})
    else:
        # 1. Mevcut simülasyon grafiği (portföy, gerçek fiyat, karşılaştırma)
        sim_fig = sim_grafik_olustur(sim_sonuc)

        # 2. Tahmin grafiğini oluştur ve üzerine sinyalleri ekle (gerçek + tahmin birlikte)
        dates, y_pred, y_gercek, metrics, hata = tahmin_yap(df, sembol, gun_input, True)
        if hata:
            tahmin_fig = None
            tahmin_fig_sadece = None
        else:
            # Gerçek + tahmin grafiği
            tahmin_fig = grafik_olustur(sembol, gun_input, dates, y_pred, y_gercek, metrics, True)
            # İşaret ekle (tahmin çizgisi üzerine)
            ax = tahmin_fig.axes[0]
            islem_listesi = sim_sonuc["islem_listesi"]
            for islem in islem_listesi:
                idx = islem["idx"]
                if idx < len(y_pred):
                    tahmin_degeri = y_pred[idx]
                    if islem["tur"] == "ALIM":
                        ax.scatter(idx, tahmin_degeri, color="#3fb950", s=100, zorder=10, marker="^", edgecolor="white", linewidth=0.5)
                        ax.annotate(f"AL\n${tahmin_degeri:.2f}", (idx, tahmin_degeri),
                                    fontsize=7, color="#3fb950", ha="center", va="bottom",
                                    xytext=(0, 10), textcoords="offset points")
                    else:
                        ax.scatter(idx, tahmin_degeri, color="#f85149", s=100, zorder=10, marker="v", edgecolor="white", linewidth=0.5)
                        ax.annotate(f"SAT\n${tahmin_degeri:.2f}", (idx, tahmin_degeri),
                                    fontsize=7, color="#f85149", ha="center", va="top",
                                    xytext=(0, -10), textcoords="offset points")
            tahmin_fig.canvas.draw()

            # Sadece tahmin grafiği (gerçek yok)
            tahmin_fig_sadece = tahmin_grafik_sadece(sembol, gun_input, dates, y_pred, sim_sonuc["islem_listesi"])

        st.session_state.update({
            "sim_fig": sim_fig,
            "sim_sonuc": sim_sonuc,
            "tahmin_fig": tahmin_fig,
            "tahmin_fig_sadece": tahmin_fig_sadece,
            "hata": None,
            "sim_result_ready": True,
        })

# ── Karşılaştırma ─────────────────────────────────────────────────────────────
if kar_buton and data_tamam and test_modu_input:
    sembol = sembol_input.strip().upper()
    test_gunu = max(gun_input, 30)
    st.session_state.update({"sembol": sembol, "gun": gun_input, "test_modu": True})

    with st.spinner("Tüm periyotlar için simülasyon çalıştırılıyor..."):
        comp = periyot_karsilastirma(
            df=df,
            sembol=sembol,
            test_days=test_gunu,
            trade_horizon=trade_horizon_input,
            initial_balance=baslangic_bakiye
        )

        if comp["errors"]:
            for err in comp["errors"]:
                st.warning(err)

        if comp["results"]:
            fig = karsilastirma_grafik_olustur(comp["results"], baslangic_bakiye, sembol, test_gunu)
            if fig:
                st.session_state.update({
                    "karsilastirma_fig": fig,
                    "karsilastirma_results": comp["results"],
                    "hata": None
                })
            else:
                st.error("Hiçbir periyotta geçerli sonuç elde edilemedi.")
        else:
            st.error("Hiçbir periyot çalıştırılamadı. Hata mesajlarını kontrol edin.")

# ── Hata göster ───────────────────────────────────────────────────────────────
if st.session_state.get("hata"):
    st.markdown(f"<div class='err-box'>🚨 {st.session_state['hata']}</div>", unsafe_allow_html=True)

# ── Tahmin sonuçları ──────────────────────────────────────────────────────────
if st.session_state.get("result_ready") and st.session_state.get("fig"):
    fig      = st.session_state["fig"]
    metrics  = st.session_state["metrics"]
    sembol   = st.session_state["sembol"]
    gun      = st.session_state["gun"]

    # Metrik kartları
    if metrics:
        yon = metrics.get("yon", {})
        cols_met = st.columns(4 if yon else 3)
        met_items = [
            ("R² Skoru", "R²", ".4f"),
            ("RMSE",     "RMSE", ".4f"),
            ("MAE",      "MAE",  ".4f"),
        ]
        for col, (label, key, fmt) in zip(cols_met[:3], met_items):
            val   = metrics.get(key, float("nan"))
            color = "#3fb950" if (key=="R²" and val>0.8) else "#f0883e" if (key=="R²" and val>0.5) else "#58a6ff"
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value' style='color:{color};'>{val:{fmt}}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        if yon and len(cols_met) == 4:
            yon_oran = yon.get("oran", 0)
            yon_renk = "#3fb950" if yon_oran >= 60 else "#f0883e" if yon_oran >= 45 else "#f85149"
            with cols_met[3]:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>Yön Doğruluğu</div>"
                    f"<div class='metric-value' style='color:{yon_renk};'>{yon_oran:.1f}%</div>"
                    f"<div class='metric-sub'>{yon.get('dogru',0)}/{yon.get('toplam',0)} gün doğru yön</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    st.pyplot(fig, use_container_width=True)

    dl1, dl2, dl3 = st.columns([1, 1, 4])
    with dl1:
        st.download_button(
            "⬇ PNG İndir", fig_to_bytes(fig, "png"),
            f"{sembol_klasor_adi(sembol)}_{gun}gun_tahmin.png", "image/png",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "⬇ PDF İndir", fig_to_bytes(fig, "pdf"),
            f"{sembol_klasor_adi(sembol)}_{gun}gun_tahmin.pdf", "application/pdf",
            use_container_width=True,
        )
    with dl3:
        if st.button("✕ Grafiği Kapat"):
            st.session_state.update({"result_ready": False, "fig": None, "metrics": {}})
            st.rerun()

elif not calistir and data_tamam and not st.session_state.get("result_ready") and not st.session_state.get("sim_result_ready") and not st.session_state.get("karsilastirma_fig"):
    st.markdown(
        "<div style='text-align:center;padding:4rem 2rem;color:#484f58;'>"
        "<div style='font-size:3rem;margin-bottom:1rem;'>📊</div>"
        "<div style='font-size:1rem;font-weight:600;color:#6e7681;'>Tahmin bekleniyor</div>"
        "<div style='font-size:0.85rem;margin-top:0.5rem;'>Sol panelden sembol ve gün sayısını seçip "
        "<b>▶ Tahmini Çalıştır</b> veya <b>📊 Simülasyonu Çalıştır</b>'a tıklayın.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Simülasyon sonuçları ──────────────────────────────────────────────────────
if st.session_state.get("sim_result_ready") and st.session_state.get("sim_fig"):
    sim_sonuc = st.session_state["sim_sonuc"]
    sim_fig   = st.session_state["sim_fig"]

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>🤖 Alım-Satım Simülatörü Sonuçları</h2>", unsafe_allow_html=True)

    net_kar        = sim_sonuc["net_kar"]
    getiri_yuz     = sim_sonuc["getiri_yuzdesi"]
    bh_getiri      = sim_sonuc["bh_getiri"]
    kazanma_orani  = sim_sonuc["kazanma_orani"]
    alim_sayisi    = sim_sonuc["alim_sayisi"]
    satim_sayisi   = sim_sonuc["satim_sayisi"]
    toplam_islem   = sim_sonuc["toplam_islem"]
    son_portfoy    = sim_sonuc["son_portfoy"]
    init_bal       = sim_sonuc["initial_balance"]
    drawdown       = sim_sonuc.get("drawdown", 0.0)

    kar_sinif = "sim-card-profit" if net_kar >= 0 else "sim-card-loss"
    kar_renk  = "#3fb950" if net_kar >= 0 else "#f85149"
    bh_renk   = "#3fb950" if bh_getiri >= 0 else "#f85149"
    kz_renk   = "#3fb950" if kazanma_orani >= 50 else "#f0883e"
    fark       = getiri_yuz - bh_getiri

    # 5 metrik kartı (drawdown eklendi)
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.markdown(
            f"<div class='{kar_sinif}'>"
            f"<div class='metric-label'>Net Kar / Zarar</div>"
            f"<div class='metric-value' style='color:{kar_renk};'>{'+' if net_kar>=0 else ''}{net_kar:,.2f} $</div>"
            f"<div class='metric-sub'>{getiri_yuz:+.2f}% getiri</div></div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='sim-card-neutral'>"
            f"<div class='metric-label'>Buy & Hold Getirisi</div>"
            f"<div class='metric-value' style='color:{bh_renk};'>{bh_getiri:+.2f}%</div>"
            f"<div class='metric-sub'>Pasif strateji</div></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='sim-card-neutral'>"
            f"<div class='metric-label'>Toplam İşlem</div>"
            f"<div class='metric-value' style='color:#58a6ff;'>{toplam_islem}</div>"
            f"<div class='metric-sub'>▲ {alim_sayisi} alım  ▼ {satim_sayisi} satım</div></div>",
            unsafe_allow_html=True,
        )
    with s4:
        st.markdown(
            f"<div class='sim-card-neutral'>"
            f"<div class='metric-label'>Kazanma Oranı</div>"
            f"<div class='metric-value' style='color:{kz_renk};'>{kazanma_orani:.1f}%</div>"
            f"<div class='metric-sub'>Kârlı satış yüzdesi</div></div>",
            unsafe_allow_html=True,
        )
    with s5:
        drawdown_renk = "#f0883e"
        st.markdown(
            f"<div class='sim-card-neutral'>"
            f"<div class='metric-label'>Maks. Geri Çekilme</div>"
            f"<div class='metric-value' style='color:{drawdown_renk};'>{drawdown:.2f}%</div>"
            f"<div class='metric-sub'>Zirveden düşüş</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

    # Karşılaştırma bilgi kutusu
    if fark > 0:
        st.markdown(f"<div class='info-box'>✅ Strateji, Buy & Hold'u <b>{fark:+.2f}%</b> puan geride bıraktı.</div>", unsafe_allow_html=True)
    elif fark < 0:
        st.markdown(f"<div class='warn-box'>⚠️ Buy & Hold, stratejiden <b>{abs(fark):.2f}%</b> puan daha iyi performans gösterdi.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>🔄 Strateji ve Buy & Hold eşit performans gösterdi.</div>", unsafe_allow_html=True)

    # Simülasyon grafiği (portföy, gerçek fiyat, karşılaştırma)
    st.pyplot(sim_fig, use_container_width=True)

    # ── Model Tahmin Grafiği (Gerçek + Tahmin) ──────────────────────────────
    if st.session_state.get("tahmin_fig"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-title'>📈 Gerçek Fiyat ve Model Tahmini (Sinyallerle)</h2>", unsafe_allow_html=True)
        st.pyplot(st.session_state["tahmin_fig"], use_container_width=True)

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            st.download_button(
                "⬇ PNG İndir", fig_to_bytes(st.session_state["tahmin_fig"], "png"),
                f"{sembol_klasor_adi(st.session_state['sembol'])}_tahmin_sinyal.png", "image/png",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ PDF İndir", fig_to_bytes(st.session_state["tahmin_fig"], "pdf"),
                f"{sembol_klasor_adi(st.session_state['sembol'])}_tahmin_sinyal.pdf", "application/pdf",
                use_container_width=True,
            )

    # ── Sadece Model Tahmin Grafiği (Gerçek yok) ────────────────────────────
    if st.session_state.get("tahmin_fig_sadece"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("<h2 class='section-title'>📈 Yalnızca Model Tahminleri (Sinyallerle)</h2>", unsafe_allow_html=True)
        st.pyplot(st.session_state["tahmin_fig_sadece"], use_container_width=True)

        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            st.download_button(
                "⬇ PNG İndir", fig_to_bytes(st.session_state["tahmin_fig_sadece"], "png"),
                f"{sembol_klasor_adi(st.session_state['sembol'])}_sadece_tahmin.png", "image/png",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "⬇ PDF İndir", fig_to_bytes(st.session_state["tahmin_fig_sadece"], "pdf"),
                f"{sembol_klasor_adi(st.session_state['sembol'])}_sadece_tahmin.pdf", "application/pdf",
                use_container_width=True,
            )

    # ── İşlem detay tablosu (Açılabilir) ─────────────────────────────────────
    if sim_sonuc.get("islem_listesi"):
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        
        # ALIM ve SATIM tarihlerinin özet tablosu
        alim_tarihleri = [i["tarih"] for i in sim_sonuc["islem_listesi"] if i["tur"] == "ALIM"]
        satim_tarihleri = [i["tarih"] for i in sim_sonuc["islem_listesi"] if i["tur"] == "SATIM"]
        
        if alim_tarihleri or satim_tarihleri:
            st.markdown("<h3 style='color:#f0f6fc;font-size:0.95rem;font-weight:600;margin-bottom:0.3rem;'>📅 Alım ve Satım Tarihleri Özeti</h3>", unsafe_allow_html=True)
            cols = st.columns(2)
            with cols[0]:
                alim_str = ", ".join(alim_tarihleri) if alim_tarihleri else "Hiç alım yok"
                st.markdown(f"<div style='background:#0d2a0d;border-left:3px solid #3fb950;padding:0.5rem 0.8rem;border-radius:4px;color:#c9d1d9;font-size:0.85rem;'><span style='color:#3fb950;font-weight:600;'>▲ ALIM:</span> {alim_str}</div>", unsafe_allow_html=True)
            with cols[1]:
                satim_str = ", ".join(satim_tarihleri) if satim_tarihleri else "Hiç satım yok"
                st.markdown(f"<div style='background:#2a0d0d;border-left:3px solid #f85149;padding:0.5rem 0.8rem;border-radius:4px;color:#c9d1d9;font-size:0.85rem;'><span style='color:#f85149;font-weight:600;'>▼ SATIM:</span> {satim_str}</div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)
        
        with st.expander("📋 Tüm İşlem Detaylarını Göster (Aç/Kapat)"):
            st.markdown(islem_tablosu_html(sim_sonuc["islem_listesi"]), unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # İndirme butonları (simülasyon grafiği için)
    sdl1, sdl2, sdl3 = st.columns([1, 1, 4])
    with sdl1:
        st.download_button(
            "⬇ PNG İndir", fig_to_bytes(sim_fig, "png"),
            f"{sembol_klasor_adi(st.session_state['sembol'])}_simulasyon.png", "image/png",
            use_container_width=True,
        )
    with sdl2:
        st.download_button(
            "⬇ PDF İndir", fig_to_bytes(sim_fig, "pdf"),
            f"{sembol_klasor_adi(st.session_state['sembol'])}_simulasyon.pdf", "application/pdf",
            use_container_width=True,
        )
    with sdl3:
        if st.button("✕ Simülasyonu Kapat"):
            st.session_state.update({
                "sim_result_ready": False,
                "sim_fig": None,
                "sim_sonuc": None,
                "tahmin_fig": None,
                "tahmin_fig_sadece": None,
            })
            st.rerun()

# ── Karşılaştırma Sonuçları ──────────────────────────────────────────────────
if st.session_state.get("karsilastirma_fig"):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>📊 Periyot Karşılaştırması</h2>", unsafe_allow_html=True)

    fig = st.session_state["karsilastirma_fig"]
    st.pyplot(fig, use_container_width=True)

    # Sonuçları tablo olarak göster
    results = st.session_state.get("karsilastirma_results", {})
    if results:
        df_tablo = pd.DataFrame.from_dict(results, orient="index")
        df_tablo.index.name = "Periyot"
        df_tablo = df_tablo.rename(columns={
            "net_kar": "Net Kar (USD)",
            "getiri_yuzdesi": "Getiri (%)",
            "islem_sayisi": "İşlem Sayısı",
            "kazanma_orani": "Kazanma Oranı (%)"
        })
        st.dataframe(df_tablo.style.format({
            "Net Kar (USD)": "{:,.2f}",
            "Getiri (%)": "{:+.2f}",
            "İşlem Sayısı": "{:.0f}",
            "Kazanma Oranı (%)": "{:.1f}"
        }).background_gradient(subset=["Net Kar (USD)"], cmap="RdYlGn", vmax=df_tablo["Net Kar (USD)"].max(), vmin=df_tablo["Net Kar (USD)"].min()))

    # İndirme butonları
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        st.download_button(
            "⬇ PNG İndir", fig_to_bytes(fig, "png"),
            f"{sembol_klasor_adi(st.session_state['sembol'])}_periyot_karsilastirma.png", "image/png",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇ PDF İndir", fig_to_bytes(fig, "pdf"),
            f"{sembol_klasor_adi(st.session_state['sembol'])}_periyot_karsilastirma.pdf", "application/pdf",
            use_container_width=True,
        )

    if st.button("✕ Karşılaştırmayı Kapat"):
        st.session_state.update({"karsilastirma_fig": None, "karsilastirma_results": None})
        st.rerun()
