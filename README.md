# 📈 Kripto Para Fiyat Tahmini
### Çok Kaynaklı API Entegrasyonu ile Kapsamlı Finansal Veri Toplama ve Makine Öğrenmesi Tabanlı Fiyat Tahmini

> **Lisans — Veri Madenciliği Dersi Projesi**  
> Durum: `Aşama 1–3 Tamamlandı` · `Aşama 4  Devam Ediyor`

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Araştırma Sorusu](#-araştırma-sorusu)
- [Proje Aşamaları](#-proje-aşamaları)
- [Veri Kaynakları](#-veri-kaynakları)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Desteklenen Semboller](#-desteklenen-semboller)
- [Teknik Detaylar](#-teknik-detaylar)
- [Proje Yapısı](#-proje-yapısı)
- [Sonuçlar](#-sonuçlar)
- [Katkıda Bulunma](#-katkıda-bulunma)

---

## 🎯 Proje Hakkında

Bu proje, kripto para piyasalarının yüksek volatiliteli yapısını veri madenciliği yöntemleriyle analiz ederek güvenilir fiyat tahmin modelleri geliştirmeyi hedeflemektedir.

**Yahoo Finance**, **Binance** ve **CoinGecko** API'lerinden otomatik olarak çekilen 2000 günlük (≈5.5 yıl) tarihsel veri üzerinde LSTM ve ensemble yöntemleri (Random Forest / XGBoost) karşılaştırmalı olarak uygulanmaktadır.

### Neden Bu Problem?

| Faktör | Açıklama |
|--------|----------|
| 🕐 7/24 Piyasa | Geleneksel piyasalardan farklı olarak kesintisiz işlem |
| 📊 Veri Erişimi | Açık API'ler sayesinde yüksek kaliteli tarihsel veri |
| ⚡ Yüksek Volatilite | Tahmin zorluğu nedeniyle akademik ilgi yoğun |
| 🔗 Çok Kaynak | API entegrasyonu ile eksik veri problemi minimize ediliyor |

---

## ❓ Araştırma Sorusu

> *Çoklu finansal API kaynaklarından elde edilen 2000 günlük tarihsel fiyat verisi kullanılarak, kripto para birimlerinin gelecek fiyat hareketleri ne ölçüde tahmin edilebilir? Hangi veri madenciliği modeli bu görev için en yüksek başarımı sağlar?*

**Tahmin Hedefleri:**
- `T+1` — Yarınki kapanış fiyatı tahmini (Regresyon)
- `T+7` — 7 günlük fiyat tahmini (Regresyon)
- `Yön Tahmini` — Fiyat artış/düşüş sınıflandırması (Binary Classification)

---

## 🗺 Proje Aşamaları

```
Aşama 1  ████████████████████  ✅ Problem Tanımı & Hedef Belirleme
Aşama 2  ████████████████████  ✅ Veri Toplama (Multi-API Scraper)
Aşama 3  ████████████████████  ✅ Keşifsel Analiz (EDA)
Aşama 4  ████████░░░░░░░░░░░░  ⏳ Veri Ön İşleme & Feature Engineering
Aşama 5  ░░░░░░░░░░░░░░░░░░░░  🔜 Modelleme (LSTM + RF/XGBoost)
Aşama 6  ░░░░░░░░░░░░░░░░░░░░  🔜 Değerlendirme & Metrikler
Aşama 7  ░░░░░░░░░░░░░░░░░░░░  🔜 Görselleştirme
Aşama 8  ░░░░░░░░░░░░░░░░░░░░  🔜 Raporlama & Sonuç
```

---

## 🌐 Veri Kaynakları

| API | Kapsam | Öncelik | Limit |
|-----|--------|---------|-------|
| **Yahoo Finance** | Hisse, endeks, döviz, emtia, kripto | Birincil | Ücretsiz / rate limit var |
| **Binance API** | Tüm kripto/USDT pariteleri | İkincil | 1000 mum/istek |
| **CoinGecko** | Kripto para birimleri | Yedek | 365 gün / ücretsiz |

### Veri Seti Özellikleri

```
📅 Zaman Aralığı  : ~2019 – 2026 (2000 takvim günü)
📊 Format         : CSV — Date + sembol kolonları (pivot yapı)
🎯 Min. Eşik      : Her sembol için ≥ 1001 gerçek işlem günü
⚡ Gerçek Zamanlı : Her çalıştırmada bugünün anlık fiyatı eklenir
🔧 Eksik Veri     : Hafta sonu / tatil → forward-fill
```

---

## 🏗 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│                OptimizedFinancialScraper                │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Yahoo Finance│  │   Binance    │  │  CoinGecko   │   │
│  │  (Birincil)  │→ │  (İkincil)   │→ │   (Yedek)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│           │                │                │           │
│           └────────────────┴────────────────┘           │
│                            │                            │
│              ┌─────────────▼───────────────┐            │
│              │   _validate_and_clean_data  │            │
│              │  • Duplicate removal        │            │
│              │  • Price > 0 filter         │            │
│              │  • Outlier detection (>300%)│            │
│              │  • OHLC consistency check   │            │
│              └─────────────┬───────────────┘            │
│                            │                            │
│              ┌─────────────▼───────────────┐            │
│              │  _fill_weekend_gaps_enhanced│            │
│              │  • Forward-fill OHLC        │            │
│              │  • Volume → 0 for non-trade │            │
│              └─────────────┬───────────────┘            │
│                            │                            │
│              ┌─────────────▼──────────────┐             │
│              │     CSV Output (Merged)    │             │
│              │  + Real-time price append  │             │
│              └────────────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

### Hata Yönetimi & Dayanıklılık

- **Exponential Backoff** — Her başarısız denemede bekleme süresi katlanır (max 30s)
- **HTTP 429 Yönetimi** — Rate limit algılandığında otomatik bekleme
- **User-Agent Rotasyonu** — Bot tespitini engellemek için dinamik başlık değişimi
- **Batch Processing** — 5'li gruplar, grup arası ~2–3s bekleme
- **Graceful Shutdown** — `Ctrl+C` ile mevcut veriler kaydedilerek güvenli çıkış
- **Bağlantı Havuzu** — `pool_connections=15`, `pool_maxsize=30`

---

## 🚀 Kurulum

### Gereksinimler

```bash
Python >= 3.8
```

### Bağımlılıkları Yükle

```bash
git clone https://github.com/kullanici-adi/kripto-fiyat-tahmini.git
cd kripto-fiyat-tahmini

pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=1.5.0
numpy>=1.23.0
requests>=2.28.0
scikit-learn>=1.2.0
tensorflow>=2.12.0      # LSTM için
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 💻 Kullanım

### 1. Veri Toplama

```bash
python scraper.py
```

Tüm semboller için veri çekilir ve `optimized_financial_data/comprehensive_market_data_2000_days.csv` dosyasına kaydedilir.

**Özelleştirilmiş kullanım:**

```python
from scraper import OptimizedFinancialScraper

scraper = OptimizedFinancialScraper(
    min_days_required=1001,   # Minimum işlem günü
    max_workers=5,            # Paralel worker sayısı
    request_timeout=30        # İstek timeout (saniye)
)

# Belirli semboller için veri çek
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

successful, failed = scraper.download_all_symbols_optimized(
    symbols=symbols,
    target_days=2000,
    output_file="my_data.csv",
    batch_processing=True
)

print(f"Başarılı: {len(successful)} | Başarısız: {len(failed)}")
```

### 2. Kaynak Kapsamını Kontrol Et

```python
coverage = scraper.validate_symbol_coverage(symbols)
# Çıktı: Her sembol için Yahoo/Binance/CoinGecko desteği tablosu
```

### 3. Keşifsel Analiz

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 4. Model Eğitimi *(yakında)*

```bash
python train.py --model lstm --symbol BTC-USD --horizon 1
python train.py --model xgboost --symbol BTC-USD --horizon 7
```

---

## 📦 Desteklenen Semboller

<details>
<summary><b>Hisse Senedi Endeksleri (11)</b></summary>

| Sembol | Tanım |
|--------|-------|
| `^GSPC` | S&P 500 |
| `^DJI` | Dow Jones |
| `^IXIC` | NASDAQ Composite |
| `^NDX` | NASDAQ 100 |
| `^FTSE` | FTSE 100 |
| `^GDAXI` | DAX |
| `^N225` | Nikkei 225 |
| `^STOXX50E` | Euro Stoxx 50 |
| `000001.SS` | Shanghai Composite |
| `^HSI` | Hang Seng |
| `XU100.IS` | BIST 100 |

</details>

<details>
<summary><b>Bireysel Hisseler (10)</b></summary>

`AAPL` `MSFT` `GOOGL` `AMZN` `NVDA` `TSLA` `META` `SPGI` `JPM` `JNJ`

</details>

<details>
<summary><b>Döviz & Emtia (12)</b></summary>

| Sembol     | Tanım        |
|------------|--------------|
| `EURUSD=X` | EUR/USD      |
| `GBPUSD=X` | GBP/USD      |
| `USDJPY=X` | USD/JPY      |
| `GC=F`     | Altın        |
| `SI=F`     | Gümüş        |
| `CL=F`     | Ham Petrol   |
| `NG=F`     | Doğalgaz     |
| `HG=F`     | Bakır        |
| `ZC=F`     | Mısır        |
| `ZS=F`     | Soya Fasulyesi |
| `ZW=F`     | Buğday       |
| `DX-Y.NYB` | Dolar Endeksi |

</details>

<details>
<summary><b>Kripto Para Birimleri (35+)</b></summary>

**Majörler:** `BTC-USD` `ETH-USD` `XRP-USD` `SOL-USD` `BNB-USD` `DOGE-USD` `AVAX-USD`

**DeFi / Layer-1:** `LINK-USD` `DOT-USD` `UNI-USD` `NEAR-USD` `APT-USD` `ARB-USD` `OP-USD`

**Diğerleri:** `BCH-USD` `XLM-USD` `TRX-USD` `ETC-USD` `FIL-USD` `SHIB-USD` `PEPE-USD` `SUI-USD`

**Stablecoin:** `USDT-USD` `USDC-USD` `DAI-USD`

</details>

---

## 🔧 Teknik Detaylar

### Veri Temizleme Pipeline

```python
# Otomatik uygulanan adımlar (_validate_and_clean_data)
1. df.drop_duplicates(subset=['Date'])          # Tekrar eden tarihler
2. pd.to_numeric(df[col], errors='coerce')      # Tip dönüşümü
3. df[df['Close'] > 0]                          # Geçersiz fiyatlar
4. df[abs(df['pct_change']) < 3.0]              # Aykırı değerler (>%300)
5. High = max(High, Open, Close)                 # OHLC tutarlılığı
6. Low  = min(Low,  Open, Close)                 # OHLC tutarlılığı
```

### Planlanan Feature Engineering

```python
# Teknik Göstergeler
- RSI(14), MACD, Bollinger Bands
- EMA(7), EMA(21), EMA(50)

# Gecikmeli Özellikler
- lag_1, lag_3, lag_7  (önceki günlerin kapanış fiyatları)

# Volatilite
- Rolling STD (7, 14, 30 gün)
- Parkinson Volatility

# Return
- log_return = log(P_t / P_{t-1})
```

### Train / Validation / Test Bölme

```
Kronolojik Bölme (data leakage önlenir):

|──────── %70 Train ────────|── %15 Val ──|── %15 Test ──|
2019                      2024          2025           2026
```

### Değerlendirme Metrikleri

| Görev | Metrikler |
|-------|-----------|
| Regresyon | MAE, RMSE, R², MAPE |
| Sınıflandırma | Accuracy, Precision, Recall, F1, ROC-AUC |
| Özel | Directional Accuracy (yön doğruluğu) |

---

## 📁 Proje Yapısı

```
├── data
├── LICENSE
├── models
├── README.md
├── results
│   ├── docs
│   │   └── Ara Rapor.pdf
│   ├── reports
│   └── visuals
└── src
    │
    └── notebooks

```

---

## 📊 Sonuçlar

> ⏳ Modelleme aşaması devam etmektedir. Sonuçlar tamamlandıkça buraya eklenecektir.

| Model | Sembol | Horizon | RMSE | MAE | R² | Dir. Acc. |
|-------|--------|---------|------|-----|----|-----------|
| LSTM | BTC-USD | T+1 | — | — | — | — |
| LSTM | ETH-USD | T+1 | — | — | — | — |
| XGBoost | BTC-USD | T+1 | — | — | — | — |
| Random Forest | BTC-USD | T+1 | — | — | — | — |

---

## 📚 Referanslar

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- [Yahoo Finance API Docs](https://finance.yahoo.com)
- [Binance API Docs](https://binance-docs.github.io/apidocs/)
- [CoinGecko API Docs](https://www.coingecko.com/en/api/documentation)

---

## ⚠️ Sorumluluk Reddi

Bu proje **akademik amaçlıdır**. Herhangi bir finansal tavsiye niteliği taşımamaktadır. Kripto para yatırımları yüksek risk içermektedir.

---

<div align="center">
  <sub>Veri Madenciliği Lisans Projesi · Nisan 2026</sub>
</div>
