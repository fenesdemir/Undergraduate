import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load("df_normS_150x150.npz") #Verimizi yüklüyoruz

x = data["arr_0"] #Görüntü verilerimiz
y = data["arr_1"] #Etiket verilerimiz

print(x.shape) #Görüntü verilerinin boyut bilgilerine bakmak için
print(y.shape) #Etiket verilerinin boyut bilgilerine bakmak için

x_mean_extract = x - x.mean(axis = 0) #Mean'i yani ortalamayı çıkarıyoruz. Direkt olarak yüz verilerinden ziyade bir yüzün ortalama yüzden ne kadar farklılaştığı ile çalışacağız
pca = PCA(n_components = None, whiten = True, svd_solver = "auto") #Parametresiz boş bir PCA modeli, kaç component kullanacağımıza karar vermek için kullanacağız
x_pca = pca.fit_transform(x_mean_extract) #Ortalamayı çıkardığımız değerleri modelemize sokuyoruz, modeli x_mean_extract'e uydurur ve boyut azaltmayı uygular
print(x_pca.shape) #x_pca değerinin boyutsallığı, modele girmeden önce 22500x1200'dü, girdikten sonra karesel (1200x1200) hale geldi, PCA bunu kendi otomatik yapıyor, bu bizim için bir sorun değil, 1200 component zaten hedeflenenin çok çok üstünde

ratio = pca.explained_variance_ratio_ #açıklanan varyans değeri; misal, 1. PC bize varyansın ne kadarını açıklıyor
ratio_cumulative = np.cumsum(ratio) #toplan açıklanan varyans değeri, misal 3 tane PC kullanmayı seçersek açıklanacak toplam (PC1'in, PCA2'nin ve PC3'ün açıkladığı toplam oran)varyans oranı, kümülatif toplam

plt.figure(figsize=(10,5)) #matplotlib ile göstereceğimiz figürün boyutu
plt.subplot(1,2,1) #figürün ilk kısmı
plt.plot(ratio, "b>--") #açıklanan varyansı mavi üçgen şeklinde çizeceğiz
plt.xlabel("Bilesen Sayisi") #grafik etiketleri, harita lejantı gibi
plt.ylabel("Aciklanan Varyans Orani") #grafik etiketleri, harita lejantı gibi
plt.subplot(1,2,2) #figürün ikinci kısmı
plt.plot(ratio[:200], "b>--") #açıklanan varyansı mavi üçgen şeklinde çizeceğiz, fakat bu sefer sadece ilk 20 PC için, malum 1200 component için çizdirince okumak daha zor, burada parametrelerle oynayarak kaç component için çizdirmek istediğimizi belirtebiliriz
plt.xlabel("Bilesen Sayisi") #grafik etiketleri, harita lejantı gibi
plt.ylabel("Aciklanan Varyans Orani") #grafik etiketleri, harita lejantı gibi
plt.show() #figürü göster

plt.figure(figsize=(10,5)) #matplotlib ile göstereceğimiz figürün boyutu, bu sefer toplam varyans için
plt.subplot(1,2,1) #figürün ilk kısmı
plt.plot(ratio_cumulative, "g>--") #açıklanan toplam varyansı yeşil üçgen şeklinde çizeceğiz
plt.xlabel("Bilesen Sayisi") #grafik etiketleri, harita lejantı gibi
plt.ylabel("Toplam Aciklanan Varyans Orani") #grafik etiketleri, harita lejantı gibi
plt.subplot(1,2,2) #figürün ikinci kısmı
plt.plot(ratio_cumulative[:200], "g>--") #açıklanan toplam varyansı yeşil üçgen şeklinde çizeceğiz, fakat bu sefer sadece ilk 20 PC için, malum 1200 component için çizdirince okumak daha zor, burada parametrelerle oynayarak kaç component için çizdirmek istediğimizi belirtebiliriz
plt.xlabel("Bilesen Sayisi") #grafik etiketleri, harita lejantı gibi
plt.ylabel("Toplam Aciklanan Varyans Orani") #grafik etiketleri, harita lejantı gibi
plt.show() #figürü göster

pca_150 = PCA(n_components = 150, svd_solver = "auto") #PCA modelimizi oluşturuyoruz 100-150 component genellikle yüz tanıma için yeterlidir, burada n_components parametresini değiştirerek farklı modeller oluşturabiliriz

x_pca_150 = pca_150.fit_transform(x_mean_extract) #Ortalama çıkarılmış verilerimize PCA uygulanarak oluşturulmuş yeni verilerimiz, modelin x_mean_extract'e uydurup boyut azaltma uygulanarak oluşturulan 100 PC
pickle.dump(pca_150, open("pca_150.pickle", "wb")) #Oluşturduğumuz PCA modelini pickle kütüphanesi ile kaydediyoruz

x_pca_inverse = pca_150.inverse_transform(x_pca_150) #inverse transform, verileri orijinal alanına geri dönüştürür
print(x_pca_inverse.shape) #inverse transform ile oluşturduğumuz verini boyutunu kontrol ediyoruz, 1200x22500

ei_img = x_pca_inverse[1,:] #inverse transform ile oluşturduğumuz öyzüzlerden ilki
ei_img = ei_img.reshape((150, 150)) #özyüzü 150x150 olacak şekilde yeniden boyutlandırıyoruz
plt.imshow(ei_img, cmap = "gray") #matplotlib ile gri tonlamalı olacak şekilde gösterme
plt.show() #özyüzü ekrana yansıtma

np.savez('data_pca_150.pickle', x_pca_150, y) #Önceki satırda oluşturduğumuz yeni veriyi pickle kütüphanesi ile kaydediyoruz
pickle.dump(x.mean(axis=0), open("mean.pickle", "wb")) #Görüntü değerlerinin ortalamasını kaydediyoruz. İleriki aşamalarda ihtiyacımız olacak
