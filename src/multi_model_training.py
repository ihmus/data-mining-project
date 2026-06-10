import pandas as pd , numpy as np , matplotlib.pyplot as plt ,os
from sklearn.model_selection import train_test_split,cross_val_score,TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import pickle,re


datafile_name="comprehensive_market_data_200_plus_features"
model_periyodu=12

Data_Path=os.getcwd().replace("\\","/")+f"/optimized_financial_data/{datafile_name}.csv"
save_path=os.getcwd().replace("\\","/")+f"/models"
if not os.path.exists(Data_Path):
    print(f"Dosya bulunamadı : {Data_Path}")
df=pd.read_csv(Data_Path)
for column in df.columns:
    if column!="Date":
        # 1. İlk değer gelene kadar olan NaN'ları 0 yap
        first_valid_index = df[column].first_valid_index()  # İlk NaN olmayan değerin indeksi
        if first_valid_index is not None:  # Eğer tüm değerler NaN değilse
            df.loc[:first_valid_index-1, column] = 0  # İlk değer öncesi NaN'ları 0 yap
        else:  # Eğer tüm değerler NaN ise
            df[column] = 0  # Tüm sütunu 0 yap
        
        # 2. Kalan NaN'ları önceki ve sonraki değerlerin ortalamasıyla doldur
        #df[column] = pd.to_numeric(df[column], errors='coerce').interpolate(method='linear')  # Linear interpolation  sayısal olmayan veriler için daha güvenli
        df[column] = df[column].astype(float).interpolate(method='linear')  # Linear interpolation  ben sayısal verilerle çalışıyorum

        
        # 3. Eğer son değerler NaN ise, forward fill ile doldur (isteğe bağlı)
        df[column] = df[column].ffill()
eksik_veri_sayisi = df.isnull().sum()

# Sadece eksik veri bulunan sütunları filtrele
eksik_olanlar = eksik_veri_sayisi[eksik_veri_sayisi > 0]
semboller=[i for i in df.drop(columns=["Date"],axis=1).columns.to_list()]
liste = semboller
aranan = "LDO-USD"

sonuclar = [eleman for eleman in liste if aranan in eleman]

# Sonuçları yazdır
for sutun, sayi in eksik_olanlar.items():
    print(f"Sütun '{sutun}' içinde {sayi} adet eksik veri var.")

def model_egit(sembol, periyot, save_path, df):
    try:
        # Giriş ve hedef değişkenleri oluştur
        inputs = df.drop(columns=[sembol, "Date"]).iloc[:-periyot, :]
        results = df[[sembol]].shift(-periyot).dropna(axis=0)

        # TimeSeriesSplit ile zaman serisi veri bölme
        tscv = TimeSeriesSplit(n_splits=5)

        # Son fold'u eğitim ve test olarak kullanma
        for train_idx, test_idx in tscv.split(inputs):
            x_train, x_test = inputs.iloc[train_idx], inputs.iloc[test_idx]
            y_train, y_test = results.iloc[train_idx], results.iloc[test_idx]

        # Model tanımı ve eğitimi
        model_pred = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model_pred.fit(x_train, y_train.values.ravel())

        # Güvenli dosya yolu oluşturma
        safe_sembol = re.sub(r"[-=]", "_", sembol)
        model_dir = os.path.join(save_path, safe_sembol, str(periyot))
        os.makedirs(model_dir, exist_ok=True)

        # Model kaydetme
        model_file = os.path.join(model_dir, f"{safe_sembol}_predict_{periyot}.pkl")
        with open(model_file, 'wb') as file:
            pickle.dump(model_pred, file)

        print(f"Model başarıyla kaydedildi: {model_file}")
        return model_file, x_test, y_test  # Test verisini de döndürüyoruz

    except Exception as e:
        print(f"Hata oluştu ({sembol}, {periyot}): {str(e)}")
        return None, None, None
def model_performans(df, sembol, periyot, n_splits=5):
    """
    Zaman serisi verisinde Random Forest ile cross-validation performansını ölçer.
    """
    try:
        # Giriş ve hedef verileri hazırla
        X = df.drop(columns=[sembol, "Date"]).iloc[:-periyot, :]
        y = df[[sembol]].shift(-periyot).dropna(axis=0).values.ravel()

        # TimeSeriesSplit ayarı
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Model tanımı
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )

        # Cross-validation skorları (R²)
        scores = cross_val_score(model, X, y, cv=tscv, n_jobs=-1)
        print(f"{n_splits}-fold R² skorları: {scores}")
        print(f"Ortalama R²: {np.mean(scores):.4f}")

        return scores, np.mean(scores)

    except Exception as e:
        print(f"Performans ölçümünde hata oluştu: {str(e)}")
        return None, None
def test_model_performans(model_file, x_test, y_test):
    try:
        # Kaydedilmiş modeli yükle
        with open(model_file, 'rb') as f:
            model = pickle.load(f)

        # Tahmin
        y_pred = model.predict(x_test)

        # Metrikler
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"R² Skoru: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")

        return r2, mse, rmse

    except Exception as e:
        print(f"Test performansında hata oluştu: {str(e)}")
        return None, None, None

if aranan in semboller:
    print("var")
for key in sonuclar:
    model_file, x_test, y_test = model_egit(key, model_periyodu, save_path, df)
    if model_file:
        model_performans(df, key, model_periyodu)
        test_model_performans(model_file,x_test,y_test)
