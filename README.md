# 📈 Kripto Para Fiyat Tahmini
## Dinamik Model Yükleme ve Yapay Zekâ Destekli Alım–Satım Simülasyonu ile Çok Kaynaklı Veri Toplamayı Birleştiren Entegre Bir Kripto Para Fiyat Tahmin Platformu

> **Veri Madenciliği Dersi Projesi**  
> **Lisans Düzeyi Teknik Rapor Temelli README**  
> Durum: **Tüm Aşamalar Tamamlandı**

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özet](#-özet)
- [Araştırma Sorusu](#-araştırma-sorusu)
- [Projenin Katkıları](#-projenin-katkıları)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Veri Toplama ve Ön İşleme](#-veri-toplama-ve-ön-i̇şleme)
- [Öznitelik Mühendisliği ve Seçimi](#-öznitelik-mühendisliği-ve-seçimi)
- [Model Eğitimi ve Değerlendirme](#-model-eğitimi-ve-değerlendirme)
- [Streamlit Uygulaması ve Simülasyon](#-streamlit-uygulaması-ve-simülasyon)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Desteklenen Varlıklar](#-desteklenen-varlıklar)
- [Proje Yapısı](#-proje-yapısı)
- [Sonuçlar ve Gözlemler](#-sonuçlar-ve-gözlemler)
- [Gelecek Çalışmalar](#-gelecek-çalışmalar)
- [Referanslar](#-referanslar)
- [Teşekkür](#-teşekkür)
- [Sorumluluk Reddi](#-sorumluluk-reddi)

---

## 🎯 Proje Hakkında

Bu proje, çok kaynaklı API entegrasyonu ile finansal zaman serisi verisi toplayan, makine öğrenmesi tabanlı fiyat tahmin modelleri üreten ve bu modelleri kullanıcı etkileşimli bir alım–satım simülasyonuna bağlayan uçtan uca bir sistem sunar.

Sistemin ana hedefi; kripto para fiyatlarını yalnızca tahmin etmek değil, bu tahminleri doğrudan karar verme mekanizmasına dönüştürerek Buy & Hold stratejisi ile karşılaştırmalı sonuçlar üretmektir.

Bu çalışma, üç ana katman üzerine kuruludur:

1. **OptimizedFinancialScraper** ile çok kaynaklı veri toplama,
2. **Lag tabanlı öznitelik üretimi + korelasyon tabanlı öznitelik seçimi** ile modelleme,
3. **Streamlit tabanlı dinamik simülasyon arayüzü** ile sinyal üretimi ve portföy takibi.

---

## 🧾 Özet

Sistem; Yahoo Finance, Binance ve CoinGecko kaynaklarını önceliklendirerek birleştiren dayanıklı bir veri toplama katmanı, gecikmeli (lag) öznitelikler ve SelectKBest / r_regression tabanlı filtre seçim yöntemiyle hazırlanmış model geliştirme katmanı ve eğitilmiş modelleri dinamik olarak yükleyen Streamlit arayüzünden oluşmaktadır.

Raporun final sürümüne göre proje;

- veri toplama,
- veri temizleme,
- eksik veri doldurma,
- öznitelik üretimi,
- öznitelik seçimi,
- model eğitimi,
- simülasyon,
- sonuç raporlama

aşamalarının tamamını kapsamaktadır. fileciteturn1file0L17-L23

---

## ❓ Araştırma Sorusu

**Çoklu finansal veri kaynaklarından elde edilen tarihsel fiyat verileri ve gecikmeli öznitelikler kullanılarak kripto para fiyatları ne ölçüde tahmin edilebilir?**

Bu projede aşağıdaki tahmin ufukları ele alınmaktadır:

| Tahmin Ufku | Açıklama |
|---|---|
| T+3 | 3 gün sonrası fiyat tahmini |
| T+7 | 7 gün sonrası fiyat tahmini |
| T+12 | 12 gün sonrası fiyat tahmini |
| T+15 | 15 gün sonrası fiyat tahmini |
| T+30 | 30 gün sonrası fiyat tahmini |

---

## ✅ Projenin Katkıları

Raporda sistemin temel katkıları şu şekilde özetlenmiştir:

- Yahoo Finance, Binance ve CoinGecko’yu önceliklendirme ve boşluk doldurma stratejileriyle birleştiren dayanıklı bir veri toplama mekanizması, fileciteturn1file0L17-L23
- Gecikmeli özniteliklerle genişletilmiş veri setinde SelectKBest / r_regression ile öznitelik indirgeme, fileciteturn1file0L19-L23
- Sembol ve periyot bazında dinamik yüklenebilen pickle modelleri, fileciteturn1file0L22-L23
- Eşik tabanlı al–sat sinyali üreten ve Buy & Hold ile karşılaştırma yapan Streamlit arayüzü. fileciteturn1file0L22-L23

---

## 🏗️ Sistem Mimarisi

Sistem üç ana bileşenden oluşur:

### 1) Veri Toplama Katmanı
- Yahoo Finance, Binance ve CoinGecko kaynaklarını önceliklendirir.
- Kripto sembollerde kaynaklar sırayla denenir.
- Kripto olmayan sembollerde Yahoo Finance önceliklidir. fileciteturn1file0L19-L23

### 2) Model Geliştirme Katmanı
- Veri temizleme,
- eksik veri doldurma,
- lag özellik üretimi,
- SelectKBest / r_regression ile öznitelik seçimi,
- Random Forest ve Gradient Boosting eğitimi. fileciteturn1file0L19-L23

### 3) Uygulama ve Simülasyon Katmanı
- Eğitilmiş modelleri dinamik yükler,
- eşik tabanlı al–sat sinyalleri üretir,
- portföy değerini gün gün takip eder,
- Buy & Hold stratejisi ile karşılaştırır. fileciteturn1file0L22-L23

---

## 🌐 Veri Toplama ve Ön İşleme

### Veri Kaynakları
| Kaynak | Kullanım |
|---|---|
| Yahoo Finance | Birincil kaynak |
| Binance API | İkincil kaynak |
| CoinGecko API | Yedek kaynak |

### Toplama Mantığı
Raporun metodoloji kısmına göre veri toplama katmanı `OptimizedFinancialScraper` sınıfı üzerinden çalışmakta ve kaynaklar önceliklendirilmektedir. Kripto semboller için Yahoo Finance → Binance → CoinGecko sırası izlenirken, kripto olmayan sembollerde yalnızca Yahoo Finance kullanılmaktadır. Yeterli veri elde edilirse erken sonlandırma yapılmaktadır. fileciteturn1file0L19-L23

### Temizleme Adımları
Ham veri üzerinde şu işlemler uygulanır:

- yinelenen tarihler kaldırılır,
- sayısal sütunlar dönüştürülür,
- sıfır/negatif kapanış fiyatları elenir,
- %300’ü aşan günlük değişimler filtrelenir,
- OHLC sütunları tutarlı hâle getirilir. fileciteturn1file0L19-L23

### Eksik Veri Doldurma
Eksik günler için üç adımlı strateji uygulanır:

1. başlangıç eksikleri için 0 ile doldurma,
2. enterpolasyon,
3. ileri doldurma (forward-fill). fileciteturn1file0L21-L23

---

## 🧠 Öznitelik Mühendisliği ve Seçimi

Veri seti, Date dışındaki her sütun için 1’den 7’ye kadar gecikmeli sürümler üretilerek genişletilmektedir. Bu yaklaşım, zaman serisinin kısa vadeli dinamiklerini yakalamayı hedefler. İlk yedi satırdaki NaN değerler temizlenir. fileciteturn1file0L21-L23

### Seçim Yöntemi
Raporun final sürümüne göre öznitelik seçimi için **Yapay Arı Kolonisi (ABC)** değil, uygulamada **SelectKBest + r_regression** kullanılmıştır. En yüksek doğrusal korelasyona sahip **k = 300** öznitelik seçilmektedir. fileciteturn1file0L21-L23

Bu seçim, yüksek boyutlu veri uzayında hesaplama maliyetini düşürürken gereksiz değişkenleri elemek amacı taşır.

---

## 🤖 Model Eğitimi ve Değerlendirme

### Kullanılan Modeller
- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Raporun metodoloji bölümüne göre Random Forest modelinde `n_estimators=3000`, `random_state=42`, `n_jobs=-1` kullanılmıştır. Gradient Boosting tarafında ise `n_estimators=7000`, `learning_rate=0.005`, `max_depth=7`, `subsample=0.9`, `min_samples_split=3`, `min_samples_leaf=3`, `max_features='sqrt'`, `n_iter_no_change=10`, `validation_fraction=0.2` parametreleri kullanılmıştır. fileciteturn1file0L22-L23

### Veri Bölme
Veri seti `train_test_split` ile `%83 eğitim / %17 test` şeklinde bölünmüştür ve `random_state=35` sabitlenmiştir. fileciteturn1file0L22-L23

### Değerlendirme Metrikleri
- R²
- MSE
- MAE

Rapor ayrıca eğitim ve test kümeleri üzerinde performans kıyaslamasının yapıldığını belirtmektedir. fileciteturn1file0L23-L25

### Model Kaydetme
Eğitilen modeller pickle formatında saklanır ve sembol / tahmin periyodu bazında dinamik yüklenebilir yapıdadır. Dosya adı mantığı da raporda açıklanmıştır. fileciteturn1file0L23-L25

---

## 🖥️ Streamlit Uygulaması ve Simülasyon

Streamlit arayüzü şu girdileri almaktadır:

- sembol,
- tahmin periyodu (3 / 7 / 12 / 15 / 30 gün),
- işlem periyodu (1–7 gün),
- eşik yüzdesi,
- başlangıç bakiyesi,
- test modu. fileciteturn1file0L24-L25

### Simülasyon Mantığı
Her gün için modelden alınan tahmin ile güncel fiyat karşılaştırılır:

- Tahmin, güncel fiyatın belirlenen eşik yüzdesi kadar üzerindeyse ve pozisyon açık değilse **alım** yapılır.
- Tahmin, güncel fiyatın belirlenen eşik yüzdesi kadar altındaysa ve pozisyon açıksa **satım** yapılır.
- Diğer durumlarda pozisyon korunur. fileciteturn1file0L24-L25

### Üretilen Çıktılar
- gerçek fiyat vs tahmin grafiği,
- alım/satım işaretleri,
- portföy değerinin zaman serisi,
- net kâr/zarar,
- kazanç oranı,
- maksimum drawdown,
- Buy & Hold karşılaştırması. fileciteturn1file0L24-L27

---

## 🚀 Kurulum

```bash
git clone https://github.com/ihmus/data-mining-project.git
cd data-mining-project
pip install -r requirements.txt
```

### Gereksinimler
- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- requests
- joblib
- matplotlib
- seaborn
- jupyter

---

## 💻 Kullanım

### 1) Veri Toplama
```bash
python src/datamining.py
```

Bu işlem birleşik veri dosyasını üretir.

### 2) Model Eğitimi
```bash
jupyter notebook src/notebooks/rf_api3_.ipynb
```

Notebook içinde:
- lag özellikleri üretilir,
- SelectKBest ile öznitelik seçimi yapılır,
- modeller eğitilir,
- pickle çıktıları kaydedilir.

### 3) Arayüzü Çalıştırma
```bash
streamlit run src/app.py
```

Arayüz üzerinden:
- sembol seçilir,
- tahmin periyodu belirlenir,
- işlem periyodu ve eşik girilir,
- simülasyon başlatılır.

---

## 📦 Desteklenen Varlıklar

### Endeksler
- ^GSPC
- ^DJI
- ^IXIC
- ^NDX
- ^FTSE
- ^GDAXI
- ^N225
- ^STOXX50E
- 000001.SS
- ^HSI
- XU100.IS

### Hisseler
- AAPL
- MSFT
- GOOGL
- AMZN
- NVDA
- TSLA
- META
- SPGI
- JPM
- JNJ

### Döviz ve Emtialar
- EURUSD=X
- GBPUSD=X
- USDJPY=X
- GC=F
- SI=F
- CL=F
- NG=F
- HG=F
- ZC=F
- ZS=F
- ZW=F
- DX-Y.NYB

### Kripto Paralar
- BTC-USD
- ETH-USD
- XRP-USD
- SOL-USD
- BNB-USD
- DOGE-USD
- AVAX-USD
- LINK-USD
- DOT-USD
- UNI-USD
- NEAR-USD
- APT-USD
- ARB-USD
- OP-USD
- BCH-USD
- XLM-USD
- TRX-USD
- ETC-USD
- FIL-USD
- SHIB-USD
- PEPE-USD
- SUI-USD
- STX-USD
- WLD-USD
- ZETA-USD

ve proje özelindeki ek semboller.

---

## 📁 Proje Yapısı

```text
.
├── images
│   └── ldo_usd.png
├── LICENSE
├── README.md
├── requirements.txt
├── results
│   ├── docs
│   │   ├── Ara Rapor.pdf
│   │   ├── Final Raporu.pdf
│   │   └── KriptoPara_Fiyat_Tahmini_Sunum.pptx
│   ├── reports
│   └── visuals
└── src
    ├── app.py
    ├── datamining.py
    ├── datas
    │   └── comprehensive_market_data_200_plus_features.csv
    ├── graphs.py
    ├── model_karsilastirma.py
    ├── models
    │   ├── ...
    ├── multi_model_training.py
    ├── notebooks
    │   └── rf_api3_.ipynb
    └── predict_with_models.py
```

---

## 📊 Sonuçlar ve Gözlemler

Raporun final bölümüne göre sistem, çok kaynaklı veri toplama, korelasyon tabanlı öznitelik seçimi, ağaç tabanlı topluluk regresyon modelleri ve etkileşimli bir simülasyon arayüzünü tek bir boru hattında birleştirmektedir. OptimizedFinancialScraper, kaynak önceliklendirmesi ve hafta sonu boşluklarını doldurma yaklaşımı ile dayanıklı bir veri katmanı sağlamaktadır. fileciteturn1file0L25-L27

### Örnek Görselleştirme

Aşağıda LDO-USD için model tahmin çıktılarına ait örnek bir ekran görüntüsü yer almaktadır. Mavi çizgi gerçek fiyatı, turuncu çizgi model tahminini, yeşil ve kırmızı işaretler ise alım ve satım sinyallerini göstermektedir.

<p align="center">
  <img src="images/ldo_usd.png" alt="LDO-USD Tahmin Sonuçları" width="900">
</p>

**Şekil 1.** LDO-USD için gerçek fiyat ve model tahminlerinin karşılaştırılması. Mavi çizgi gerçek fiyatı, turuncu çizgi model tahminini, yeşil (▲) ve kırmızı (▼) işaretler ise simülasyon sırasında oluşan alım ve satım sinyallerini göstermektedir.

### Simülasyon Çıktıları
Simülasyon sonunda şu metrikler sunulur:
- başlangıç ve bitiş bakiyesi,
- net kâr/zarar,
- toplam işlem sayısı,
- kazançlı işlem sayısı,
- kazanç oranı,
- maksimum geri çekilme,
- Buy & Hold karşılaştırması. fileciteturn1file0L25-L27

---

## 🔮 Gelecek Çalışmalar

Raporun tartışma ve sonuç kısmında gelecekte şu geliştirmeler önerilmektedir:

- işlem maliyeti ve slippage modelleme,
- kademeli pozisyonlama,
- stop-loss / take-profit eklenmesi,
- çoklu varlık portföyü desteği,
- gerçek zamanlı veri akışı,
- canlı ticaret arayüzü,
- LSTM ve GRU entegrasyonu,
- sezgisel öznitelik seçimi yöntemlerinin karşılaştırılması. fileciteturn1file0L27-L27

---

## 📚 Referanslar

1. Breiman, L. (2001). *Random Forests*. Machine Learning.
2. Drucker, H. (1997). *Improving Regressors Using Boosting Techniques*. ICML.
3. Karaboga, D. (2005). *An Idea Based on Honey Bee Swarm for Numerical Optimization*.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
5. Pedregosa, F. et al. (2011). *Scikit-Learn: Machine Learning in Python*.
6. Yahoo Finance API Documentation.
7. Binance API Documentation.
8. CoinGecko API Documentation.
9. Streamlit Documentation.
10. Backtrader Documentation.

Raporun kaynaklar bölümünde bu referanslar yer almaktadır. fileciteturn1file0L28-L29

---

## 🙏 Teşekkür

Bu çalışma, Veri Madenciliği dersi kapsamında hazırlanmıştır. Proje sürecindeki geri bildirimleri için ders sorumlusu ve danışmanlık desteği sağlayan Hakan Gündüz’e teşekkür ederiz. fileciteturn1file0L27-L27

---

## ⚠️ Sorumluluk Reddi

Bu proje akademik amaçlıdır. Herhangi bir finansal tavsiye niteliği taşımaz. Kripto para piyasaları yüksek risk içerir ve gerçek yatırım kararları için kullanılmamalıdır.

---

<div align="center">
  <sub>Veri Madenciliği Dersi Projesi · 2026</sub>
</div>
