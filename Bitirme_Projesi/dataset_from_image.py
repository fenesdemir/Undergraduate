import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import MinMaxScaler
from glob import glob

def getName(path): #isim alma fonksiyonu, path'i parametre olarak alır
    string = str(path) # path2i string'e çevirir
    return string.split("/")[2].split("\\")[0] # string'i bölerek kişi adını alır yani ./images/Authorized/authorized_1'den bize authorized'ı çeker

def getFlat(path): #flatten etme fonksiyonu, path'i paramtre olarak alır
    img = cv2.imread(path) #adresteki görseli okur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #görseli grayscale'e çevirir
    return gray.flatten() #flatten halini göndür, misal 150x150'lik görüntü 1x22500


authorized_path = glob("./images/authorized/*.jpg") #yetkili kullanıcı yolu
others_path = glob("./images/others/*.jpg") #diğer kullanıcıların yolu

path = authorized_path + others_path # bu iki path'in toplamı

print(path[::-1])

df = pd.DataFrame(data = path, columns= ["path"]) #df isimli data sütununda görüntünün yolunu tutan bir dataframe
df["person"] = df["path"].apply(getName) #person isimli sütun kişi isimleri path'dan çekerek tutuyor
df["data"] = df["path"].apply(getFlat) #data sütunu görüntü matrisinin düzleştirilmiş halini tutuyor
print(df.head())
print(df.tail())

df_toSave = df["data"].apply(pd.Series) #df_to save isimli yeni dataframe'im, data sütunundaki flat haldeki matris bilgisini pandas series haline çeviriyor yani 22500 tane sütunumuz var
df_toSave = pd.concat((df["person"], df_toSave), axis = 1) # bu sütunların başına etiketi getiriyoruz
print(df_toSave.head())

indep = df_toSave.iloc[:,1:].values #görüntü değerleri
dep = df_toSave.iloc[:,0].values #etiketler
print(indep)
print(dep)

scaler = MinMaxScaler() #min max scaler en düşük değer 0'a en yüksek değer 1'e eşit olacak şekilde normalizasyon

print(indep.shape)
print(dep.shape)

scaled_indep = scaler.fit_transform(indep) #scaling işlemini uygula
dep_norm = np.where(dep == "authorized", 1,0) # etiket değerleri "authorized" ise 1, "others" ise 0 uygula

print(dep_norm)
print(scaled_indep)

np.savez("df_normS_150x150", scaled_indep, dep_norm) #normalize edilmiş veriler ve etiketleri numpyz olarak sakla

