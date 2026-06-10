import pandas as pd , numpy as np , matplotlib.pyplot as plt ,os
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score,mean_squared_error,f1_score
import pickle,matplotlib.pyplot as plt

import re

tahmin_periyodu=100
symbls=[   "BERA-USD","MOVE32452-USD","RUNE-USD", "WLD-USD","AXL17799-USD","YGG-USD","BTC-USD","GC=F","LDO-USD",
             "RARI-USD","SLND-USD","STX-USD",
             "SYN-USD","ZETA-USD",
             "CTX-USD","MASK-USD","TEL-USD","AXL-USD","KTA-USD","OCEAN-USD","TRC-USD"
        ]
model_periyodu=12
# Dosya yolunu ekleme
data_1="optimized_financial_data"
data_2="tahmin/veriler"
Data_Path=os.getcwd().replace("\\","/")+f"/{data_1}"


def load_model(model_name, models_path='Models', format='pkl'):
    """
    Farklı formatlardaki modelleri yükler
    
    Args:
        model_name (str): Model adı
        models_path (str): Modeller dizini
        format (str): Model formatı ('pkl', 'h5', 'joblib', 'onnx' vb.)
    """
    file_path = models_path+ f"/{model_name}.{format}"
    
    if format == 'pkl':
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    elif format == 'joblib':
        import joblib
        return joblib.load(file_path)
    # elif format == 'h5':
    #     from tensorflow.keras.models import load_model as load_keras_model
    #     return load_keras_model(file_path)
    else:
        raise ValueError(f"Desteklenmeyen format: {format}")
    """
    Farklı formatlardaki dosyaları yükler
    
    Args:
        file_name (str): Dosya adı
        file_path (str): Dosya dizini
        format (str): Dosya formatı ('pkl', 'h5', 'joblib', 'onnx' vb.)
    """
    file_path = os.path.join(file_path, f"{file_name}.{format}")
    if not os.path.exists(file_path):
        print("Dosya bulunamadı")
    try:
        if format=="csv":
            return pd.read_csv(file_path)
        else:
            with open(file_path,"r") as file:
                return file
    except Exception as e:
        print(f"Hata -> {e}")
try:
    Datas=pd.read_csv(f"{Data_Path}/birlesik_veri.csv")
    df=Datas
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
except Exception as e:
    print(f"Hata -> {e}")
def drawing(symbl):
    Models_Path=os.getcwd().replace("\\","/")+f'/models/{re.sub(r"[-=]", "_", symbl)}/{model_periyodu}'
    model_261=load_model(model_name=f'{re.sub(r"[-=]", "_", symbl)}_predict_{model_periyodu}',models_path=Models_Path)
    ftrs=model_261.feature_names_in_.tolist()
    inputs_for_model_261 =df[model_261.feature_names_in_.tolist()]
    results=df[[symbl]]
    print(results.iloc[-3:])

    x1=[str(i).split(" ")[0] for i in df["Date"].values.flatten().tolist()][-tahmin_periyodu:]
    x2=[i for i in range(model_periyodu,tahmin_periyodu+model_periyodu)]

    y3=model_261.predict(df[ftrs].iloc[-tahmin_periyodu:,])

    r1=results.iloc[-tahmin_periyodu:,0].to_numpy()
    model_261_r2=r2_score(results.iloc[-tahmin_periyodu+7:],model_261.predict(inputs_for_model_261.iloc[-tahmin_periyodu:-7]))
    plt.figure(figsize=(13, 7), dpi=100)

    # Tahmin ve Gerçek Değerlerin Çizimi
    line3,=plt.plot(x2, y3, label="Tahmin 2", color="grey")
    plt.plot(x1, r1,linewidth=2, label="Gerçek Değerler", color="blue")

    for i in range(len(y3)):
        plt.text(x2[i], y3[i], f'{i} = {y3[i]:.2f} {((y3[i]-y3[i-1])/y3[i-1])*100:+.2f}', color='grey', fontsize=8, ha='center')

    for i in range(min(len(x1), len(r1))):
        plt.text(x1[i], r1[i], f'{r1[i]:.2f}', color='blue', fontsize=10, ha='center')


    # Grafiği Özelleştir
    plt.title(f"{symbl} {model_periyodu} günlük Tahmini")
    plt.xlabel("X Ekseni")
    plt.ylabel("Y Ekseni")
    plt.xticks(np.arange(0, len(x1), 1), rotation=45)
    plt.grid(True)
    plt.legend(handles=[line3],
        labels=[f'model_261 r2 score : {model_261_r2:.3f}'])
    plt.tight_layout()
    plt.show()
for i in symbls:
    drawing(i)
