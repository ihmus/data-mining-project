import pandas as pd , numpy as np , matplotlib.pyplot as plt ,os
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error,f1_score
import pickle,matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re,time
tahmin_periyodu=83
model_periyodu=12
# Dosya yolunu ekleme
data_1="optimized_financial_data"
data_2="tahmin/veriler"

Models_Path=os.getcwd().replace("\\","/")+f'/models'
Data_Path=os.getcwd().replace("\\","/")+f"/{data_1}"
Saved_Path=os.getcwd().replace("\\","/")+f"/graphs/{model_periyodu}"
os.makedirs(Saved_Path, exist_ok=True)  # info klasörünü oluşturur, varsa hata vermez

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
            df[column] = df[column].astype(float).interpolate(method='linear')  # Linear interpolation  ben sayısal verilerle çalışıyorum

            
            # 3. Eğer son değerler NaN ise, forward fill ile doldur (isteğe bağlı)
            df[column] = df[column].ffill()
except Exception as e:
    print(f"Hata -> {e}")
#symbls=["LDO-USD","YGG-USD","BERA-USD","MOVE32452-USD","RUNE-USD","RARI-USD","SLND-USD","STX-USD","SYN-USD","ZETA-USD","AXL17799-USD","CTX-USD","MASK-USD","TEL-USD"]
symbls=df.drop(columns=["Date"],axis=1).columns.to_list()
index=0
#symbls=[i for i in symbls if "-usd" in i or "-USD" in i]
for symbl in symbls:
    index+=1
    # model_160=load_model(model_name="model_bnb_predict_160_plus_features_3_days",models_path=Models_Path)
    # model_3_days_new=load_model(model_name="model_bnb_predict_263_features_3days",models_path=Models_Path)
    name=re.sub(r"[-=]", "_", symbl)
    model_261=load_model(model_name=f'{name}_predict_{model_periyodu}',models_path=f'{Models_Path}/{re.sub(r"[-=]", "_", symbl)}/{model_periyodu}')
    print(f'{Models_Path}/{re.sub(r"[-=]", "_", symbl)}/{model_periyodu}/{name}_predict_{model_periyodu}')
    # inputs_for_model_160 =df[model_160.feature_names_in_.tolist()]
    inputs_for_model_261 =df[model_261.feature_names_in_.tolist()]
    # inputs_for_model_3_days_new = df[model_3_days_new.feature_names_in_.tolist()]
    results=df[[symbl]]

    x1=[str(i).split(" ")[0] for i in df["Date"].values.flatten().tolist()][-tahmin_periyodu:]
    x2=[i for i in range(model_periyodu,tahmin_periyodu+model_periyodu)]
    # y1=model_160.predict(inputs_for_model_160.iloc[-tahmin_periyodu:,])
    # y2=model_3_days_new.predict(inputs_for_model_3_days_new.iloc[-tahmin_periyodu:,])
    y3=model_261.predict(inputs_for_model_261.iloc[-tahmin_periyodu:,])

    r1=results.iloc[-tahmin_periyodu:,0].to_numpy()
    # model_160_r2=r2_score(r1,y1)
    # model_3_days_new_r2=r2_score(r1,y2)
    model_261_r2=r2_score(results.iloc[model_periyodu:],model_261.predict(inputs_for_model_261)[:-model_periyodu])
    plt.figure(figsize=(13, 7), dpi=100)
    # Tahmin ve Gerçek Değerlerin Çizimi
    # line1,=plt.plot(x2, y1, label="Tahmin 1", color="orange")
    # line2,=plt.plot(x2, y2, label="Tahmin 2", color="green")
    line3,=plt.plot(x2, y3, label="Tahmin 2", color="grey")

    plt.plot(x1, r1,linewidth=2, label="Gerçek Değerler", color="blue")



    # for i in range(len(y1)):
    #     plt.text(x2[i], y1[i] + 0.3, f'{y1[i]:.2f}', color='orange', fontsize=8, ha='center')
    # for i in range(len(y2)):
    #     plt.text(x2[i], y2[i] + 0.3, f'{y2[i]:.2f}', color='green', fontsize=8, ha='center')
    for i in range(len(y3)):
        plt.text(x2[i], y3[i], f'{y3[i]:.2f}', color='grey', fontsize=8, ha='center')
    for i in range(min(len(x1), len(r1))):
        plt.text(x1[i], r1[i], f'{r1[i]:.2f}', color='blue', fontsize=10, ha='center')


    # Grafiği Özelleştir
    plt.title(f"{symbl} {model_periyodu} günlük Tahmini", fontsize=12)
    plt.xlabel("X Ekseni", fontsize=7)
    plt.ylabel("Y Ekseni", fontsize=7)
    plt.xticks(np.arange(0, len(x1), 1), rotation=45)
    plt.grid(True)
    plt.legend(handles=[ line3],
        labels=[
                f'model_261 r2 score : {model_261_r2:.3f}'])
    plt.savefig(f"{Saved_Path}/{name}grafik.png", bbox_inches='tight')  
    plt.close()  
    time.sleep(0.1)
    print(f"{index} => {name}grafik.png")
