import os
import re
import pickle
import io

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_squared_error

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finansal Tahmin Paneli",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stCheckbox label { color: #8b949e !important; font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }

    /* Main background */
    .main .block-container { background: #0d1117; padding-top: 1.5rem; }
    body { background-color: #0d1117; }

    /* Metric cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.35rem;
    }
    .metric-card .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #58a6ff;
    }
    .metric-card .metric-sub {
        font-size: 0.72rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* Error / warning boxes */
    .err-box {
        background: #1f0a0a;
        border-left: 3px solid #f85149;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #f85149;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #1a1500;
        border-left: 3px solid #e3b341;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #e3b341;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }

    /* Page title */
    h1.page-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin-bottom: 0;
    }
    .page-subtitle {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Stagger divider */
    hr.section-divider {
        border: none;
        border-top: 1px solid #21262d;
        margin: 1.5rem 0;
    }

    /* Button overrides */
    .stDownloadButton > button {
        background: #238636 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        background: #2ea043 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.getcwd(), "datas", "comprehensive_market_data_200_plus_features.csv")
MODELS_ROOT = os.path.join(os.getcwd(), "models")
GUN_SECENEKLERI = [3, 7, 12, 15, 30]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sembol_klasor_adi(sembol: str) -> str:
    """BTC-USD → BTC_USD"""
    return re.sub(r"[-=]", "_", sembol)


def model_yolu(sembol: str, gun: int) -> str:
    return os.path.join(MODELS_ROOT, sembol_klasor_adi(sembol), str(gun))


def model_dosya_adi(sembol: str, gun: int) -> str:
    return f"{sembol_klasor_adi(sembol)}_predict_{gun}"


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, models_path: str):
    file_path = os.path.join(models_path, f"{model_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı:\n{file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV dosyası bulunamadı:\n{path}")
    df = pd.read_csv(path)
    for col in df.columns:
        if col == "Date":
            continue
        first_valid = df[col].first_valid_index()
        if first_valid is not None:
            if first_valid > 0:
                df.loc[:first_valid - 1, col] = 0
        else:
            df[col] = 0
        df[col] = df[col].astype(float).interpolate(method="linear").ffill()
    return df


def tahmin_yap(
    df: pd.DataFrame,
    sembol: str,
    gun: int,
    test_modu: bool,
):
    """
    Returns (dates, y_pred, y_gercek_or_None, metrics_dict, hata_str_or_None)
    """
    if sembol not in df.columns:
        return None, None, None, {}, f"'{sembol}' sütunu CSV'de bulunamadı. Mevcut sütunları kontrol edin."

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
        return None, None, None, {}, f"Modelin gerektirdiği {len(missing)} özellik CSV'de eksik: {missing[:5]}…"

    n = len(df)

    # ── TEST MODU ─────────────────────────────────────────────────────────────
    if test_modu:
        if gun > n:
            return None, None, None, {}, (
                f"Seçilen gün sayısı ({gun}), CSV'deki toplam satır sayısından ({n}) büyük."
            )

        test_df = df.iloc[-gun:]
        X_test = test_df[ftrs]
        y_gercek = test_df[sembol].to_numpy()

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            return None, None, None, {}, f"Tahmin sırasında hata: {e}"

        dates_raw = df["Date"].values[-gun:]
        dates = [str(d).split(" ")[0] for d in dates_raw]

        metrics = {}
        if len(y_gercek) == len(y_pred) and len(y_gercek) > 1:
            metrics["R²"] = r2_score(y_gercek, y_pred)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_gercek, y_pred))
            metrics["MAE"] = float(np.mean(np.abs(y_gercek - y_pred)))

        return dates, y_pred, y_gercek, metrics, None

    # ── TAHMİN MODU (ileriye dönük) ───────────────────────────────────────────
    else:
        # Son satırı baz alarak gelecek `gun` adım tahmini
        # Strateji: son satırı `gun` kez tekrarlayıp tahmin üret
        # (Gerçek gelecek değeri yoktur)
        last_row = df[ftrs].iloc[[-1]]
        repeated = pd.concat([last_row] * gun, ignore_index=True)

        try:
            y_pred = model.predict(repeated)
        except Exception as e:
            return None, None, None, {}, f"Tahmin sırasında hata: {e}"

        # Tarih aralığı: son tarihten itibaren
        try:
            last_date = pd.to_datetime(df["Date"].iloc[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=gun)
            dates = [d.strftime("%Y-%m-%d") for d in future_dates]
        except Exception:
            dates = [f"T+{i+1}" for i in range(gun)]

        return dates, y_pred, None, {}, None


def grafik_olustur(
    sembol: str,
    gun: int,
    dates,
    y_pred,
    y_gercek,
    metrics: dict,
    test_modu: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    x = np.arange(len(dates))

    if test_modu and y_gercek is not None:
        ax.plot(x, y_gercek, color="#58a6ff", linewidth=2.2, label="Gerçek Değerler", zorder=3)
        ax.fill_between(x, y_gercek, alpha=0.08, color="#58a6ff")
        ax.plot(x, y_pred, color="#f0883e", linewidth=1.8, linestyle="--", label="Tahmin", zorder=4)

        for i in range(len(y_pred)):
            ax.annotate(
                f"{y_pred[i]:.2f}",
                (x[i], y_pred[i]),
                fontsize=6.5,
                color="#f0883e",
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

        r2_str  = f"R² = {metrics.get('R²', float('nan')):.4f}"
        rmse_str = f"RMSE = {metrics.get('RMSE', float('nan')):.4f}"
        ax.set_title(
            f"{sembol}  ·  {gun} Günlük Test Simülasyonu   {r2_str}  |  {rmse_str}",
            color="#f0f6fc",
            fontsize=11,
            fontweight="600",
            pad=12,
        )
    else:
        ax.plot(x, y_pred, color="#3fb950", linewidth=2.2, label="İleriye Tahmin", zorder=3)
        ax.fill_between(x, y_pred, alpha=0.08, color="#3fb950")

        for i in range(len(y_pred)):
            delta = ((y_pred[i] - y_pred[i - 1]) / y_pred[i - 1] * 100) if i > 0 else 0
            ax.annotate(
                f"{y_pred[i]:.2f}  ({delta:+.1f}%)",
                (x[i], y_pred[i]),
                fontsize=6.5,
                color="#3fb950",
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

        ax.set_title(
            f"{sembol}  ·  {gun} Günlük İleriye Tahmin",
            color="#f0f6fc",
            fontsize=11,
            fontweight="600",
            pad=12,
        )

    # Axes styling
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7.5, color="#8b949e")
    ax.tick_params(axis="y", colors="#8b949e", labelsize=8)
    ax.spines[:].set_color("#21262d")
    ax.grid(True, color="#21262d", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Tarih", color="#8b949e", fontsize=9)
    ax.set_ylabel("Fiyat", color="#8b949e", fontsize=9)

    legend = ax.legend(
        fontsize=8.5,
        facecolor="#161b22",
        edgecolor="#21262d",
        labelcolor="#c9d1d9",
        loc="upper left",
    )

    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ── SESSION STATE INIT ────────────────────────────────────────────────────────
for key, val in [
    ("result_ready", False),
    ("fig", None),
    ("metrics", {}),
    ("hata", None),
    ("sembol", "BTC-USD"),
    ("gun", 7),
    ("test_modu", True),
]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:#f0f6fc;margin-bottom:0.5rem;'>📈 Tahmin Paneli</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#21262d;margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

    sembol_input = st.text_input(
        "Sembol",
        value=st.session_state["sembol"],
        placeholder="örn: BTC-USD",
        help="Yahoo Finance formatında sembol girin.",
    )

    gun_input = st.selectbox(
        "Tahmin Günü",
        options=GUN_SECENEKLERI,
        index=GUN_SECENEKLERI.index(st.session_state["gun"]) if st.session_state["gun"] in GUN_SECENEKLERI else 1,
    )

    test_modu_input = st.checkbox(
        "Test / Simülasyon Modu",
        value=st.session_state["test_modu"],
        help=(
            "✅ Açık: CSV'nin son N satırını test seti olarak ayırır, "
            "mevcut model ile tahmin yapar ve gerçek değerlerle karşılaştırır.\n\n"
            "⬜ Kapalı: Bugünden itibaren N gün ileriye tahmin üretir."
        ),
    )

    st.markdown("<hr style='border-color:#21262d;margin:1rem 0;'>", unsafe_allow_html=True)

    calistir = st.button("▶ Tahmini Çalıştır", use_container_width=True, type="primary")

    st.markdown(
        "<div style='font-size:0.72rem;color:#484f58;margin-top:1.5rem;line-height:1.6;'>"
        "Model dizini:<br><code style='color:#6e7681;'>models/{SEMBOL}/{GÜN}/</code>"
        "<br>Veri dosyası:<br><code style='color:#6e7681;'>optimized_financial_data/birlesik_veri.csv</code>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 class='page-title'>Finansal Tahmin Paneli</h1>"
    "<p class='page-subtitle'>ML modeli ile fiyat tahmini · Test & İleriye Dönük Mod</p>",
    unsafe_allow_html=True,
)

# Load data once
try:
    df = load_data(DATA_PATH)
    data_tamam = True
except FileNotFoundError as e:
    st.markdown(f"<div class='err-box'>🚨 {e}</div>", unsafe_allow_html=True)
    data_tamam = False
except Exception as e:
    st.markdown(f"<div class='err-box'>🚨 CSV yüklenirken beklenmedik hata: {e}</div>", unsafe_allow_html=True)
    data_tamam = False

if data_tamam:
    n_rows = len(df)
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Toplam Kayıt</div>"
            f"<div class='metric-value'>{n_rows:,}</div>"
            f"<div class='metric-sub'>satır</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_info2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Kolon Sayısı</div>"
            f"<div class='metric-value'>{len(df.columns):,}</div>"
            f"<div class='metric-sub'>özellik</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_info3:
        try:
            son_tarih = pd.to_datetime(df["Date"].iloc[-1]).strftime("%d %b %Y")
        except Exception:
            son_tarih = str(df["Date"].iloc[-1])
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Son Tarih</div>"
            f"<div class='metric-value' style='font-size:1.2rem;'>{son_tarih}</div>"
            f"<div class='metric-sub'>CSV'deki son gün</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── RUN ───────────────────────────────────────────────────────────────────────
if calistir and data_tamam:
    sembol = sembol_input.strip().upper()
    gun = gun_input
    test_modu = test_modu_input

    st.session_state["sembol"] = sembol
    st.session_state["gun"] = gun
    st.session_state["test_modu"] = test_modu

    # Validation: test modu gün > satır
    if test_modu and gun > len(df):
        st.markdown(
            f"<div class='warn-box'>⚠️ Seçilen gün ({gun}), CSV satır sayısından ({len(df)}) büyük. Lütfen daha küçük bir değer seçin.</div>",
            unsafe_allow_html=True,
        )
        st.session_state["result_ready"] = False
    else:
        with st.spinner("Model yükleniyor ve tahmin yapılıyor…"):
            dates, y_pred, y_gercek, metrics, hata = tahmin_yap(
                df, sembol, gun, test_modu
            )

        if hata:
            st.session_state["hata"] = hata
            st.session_state["result_ready"] = False
        else:
            fig = grafik_olustur(sembol, gun, dates, y_pred, y_gercek, metrics, test_modu)
            st.session_state["fig"] = fig
            st.session_state["metrics"] = metrics
            st.session_state["hata"] = None
            st.session_state["result_ready"] = True

# ── RESULTS ───────────────────────────────────────────────────────────────────
if st.session_state.get("hata"):
    st.markdown(
        f"<div class='err-box'>🚨 {st.session_state['hata']}</div>",
        unsafe_allow_html=True,
    )

if st.session_state.get("result_ready") and st.session_state.get("fig") is not None:
    fig = st.session_state["fig"]
    metrics = st.session_state["metrics"]
    sembol = st.session_state["sembol"]
    gun = st.session_state["gun"]

    # Metric cards
    if metrics:
        mc1, mc2, mc3 = st.columns(3)
        for col, (label, key, fmt) in zip(
            [mc1, mc2, mc3],
            [("R² Skoru", "R²", ".4f"), ("RMSE", "RMSE", ".4f"), ("MAE", "MAE", ".4f")],
        ):
            val = metrics.get(key, float("nan"))
            color = "#3fb950" if (key == "R²" and val > 0.8) else "#f0883e" if (key == "R²" and val > 0.5) else "#58a6ff"
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-label'>{label}</div>"
                    f"<div class='metric-value' style='color:{color};'>{val:{fmt}}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Chart
    st.pyplot(fig, use_container_width=True)

    # Download buttons
    dl1, dl2, dl3 = st.columns([1, 1, 4])
    with dl1:
        png_bytes = fig_to_bytes(fig, "png")
        st.download_button(
            label="⬇ PNG İndir",
            data=png_bytes,
            file_name=f"{sembol_klasor_adi(sembol)}_{gun}gun_tahmin.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl2:
        pdf_bytes = fig_to_bytes(fig, "pdf")
        st.download_button(
            label="⬇ PDF İndir",
            data=pdf_bytes,
            file_name=f"{sembol_klasor_adi(sembol)}_{gun}gun_tahmin.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl3:
        if st.button("✕ Grafiği Kapat", use_container_width=False):
            st.session_state["result_ready"] = False
            st.session_state["fig"] = None
            st.session_state["metrics"] = {}
            st.rerun()

elif not calistir and data_tamam and not st.session_state.get("result_ready"):
    st.markdown(
        "<div style='text-align:center;padding:4rem 2rem;color:#484f58;'>"
        "<div style='font-size:3rem;margin-bottom:1rem;'>📊</div>"
        "<div style='font-size:1rem;font-weight:600;color:#6e7681;'>Tahmin bekleniyor</div>"
        "<div style='font-size:0.85rem;margin-top:0.5rem;'>Sol panelden sembol ve gün sayısını seçip <b>▶ Tahmini Çalıştır</b>'a tıklayın.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
