Projeyi Hazırlayan: Furkan Enes DEMİR, 170201117

Projenin amacı makine öğrenmesi kullanarak araba içinde yüz tespiti yapmaktır. Yöntem olarak PCA vs SVM kullanılmıştır.

1- İlk olarak "face_from_video.py" dosyasındaki kodlar çalıştırılarak video görüntülerinde oluşmuş verisetinde yüz görüntüleri 150x150 ve grayscale olacak şekilde alınır. bu dosyada 5, 6 ve 7. satırdaki kodlar oluşturulacak yolu (path),
13. satır kullanılacak videoyu, 32. satır kaydedilen görüntülerin isim formatını belirtir. Bu satırlar ve 37. satırdaki sayaç değeri ile oynayarak farklı sayıda, farklı görüntüler alınabilir.

2- İkinci sırada "dataset_from_image.py" dosyasındaki kodlar çalıştırılır. Bu kodlar önceki adımda oluşturulan klasörlerdeki görünütlerin piksel verilerini ve etiketlerini her satır bir görüntüyü ifade edecek şekilde bir dataframe
şekline getirir.

3- Üçüncü olarak "PCA.py" dosyasındaki kodlar çalıştırılır. Burada oluşturulmuş dataframe yüklenir ve açıklanan varyans oranı ve toplan açıklanan varyans oranı grafikleri çizdirilerek PCA modeli oluşturulur ve veriler bu modele 
sokularak çıkış verileri kaydedilir. Burada 28. ve 39. satırlarda değişiklik yaparak istediğiniz aralıklar için açıklanan varyans oranı ve toplan açıklanan varyans oranını görebilirsiniz. Ek olarak 44. satırdaki parametreleri 
değiştirerek farklı PCA modelleri oluşturabilirsiniz.

4- Dördüncü olarak "SVM.py" dosyasındaki kodlar çalıştırılır. Bu kısımda öncedi adımdaki PCA uygulanmış veriler ile çalışılarak SVM modeli oluşturulur. 17-30. satırlar arasında bir GridSearchCV ile model için optimum parametreler aranır.
Burada "parameters" dictionary'sindeki değerleri değiştirerek farklı değerler için test yapabilrisiniz. Burada 4'ten az principal component için bu işlemin sonsuza kadar sürdüğünü not düşmek isterim. Ek olarak 34. satırda parametreleri
değiştirerek farklı SVM modelleri oluşturabilirsiniz.

5- Beşinci adımda "test_realtime.py" dosyasındaki kodlar çalıştırılır. Bu kısımda artık oluşturulmuş model gerçek zamanlı test edilir. 5. satırda SVM modeli, 6. satırda PCA modeli ve 7. satırda kaydedilmiş ortalama veriler yüklenir. Bu
satırda değişklik yaparak istediğiniz modelleri yükleyebilirsiniz.

Gerçek zamanlı test için kameranızın olması gerekmektedir.

