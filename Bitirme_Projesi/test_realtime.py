import pickle
import cv2

cascade = cv2.CascadeClassifier(r"./cascades/data/haarcascade_frontalface_default.xml") # cascade classfier yüklenir
svm_model = pickle.load(open("./SVM_150c.pickle", "rb")) # svm modeli yüklenir, model ismi değiştirilerek farklı modeller yüklenebilir
pca_model  = pickle.load(open('./pca_150.pickle','rb')) # pca modeli yüklenir, model ismi değiştirilerek farklı modeller yüklenebilir
mean  = pickle.load(open('./mean.pickle','rb')) # verilerin ortalaması yüklenir

persons = ['others','authorized'] #kişi etiketleri
font = cv2.FONT_HERSHEY_SIMPLEX # opencv'de kullanılacak font belirlenir

print("Models have been loaded succesfully.")

def face_rec(img,color='rgb'): #yüz tespit edecek özel fonksiyonumuz
    if color == 'bgr': # renk kanalı bgr ise gray'e, eğer değilse yani rgb ise yine gri tonlamalıya (grayscale) çevirilir
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray,1.5,3) # gri tonlamalı yüz tespiti yapılır
    for x,y,w,h in faces: # döndürülen yüz koordinatları için
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # yüzün etrafına kare çizilir
        roi = gray[y:y+h,x:x+w] # yüzün olduğu alanı gri tonalamalı olarak alınır, bu bizim region of interestimizdir (roi)

        roi = roi / 255.0 # roi normalleştirilir.
        roi_resize = cv2.resize(roi,(150,150)) # roi yeniden boyutlandırılır

        roi_reshape = roi_resize.reshape(1,22500) # roi'deki veriler tek satıra indirgenir

        roi_mean = roi_reshape - mean # roi'deki verilerden mean yani ortalama çıkarılır böylece yüzün ortalama yüzden farkını alınmışl olunur

        eigen_image = pca_model.transform(roi_mean) # ortalama çıkarılmış veri PCA modeline sokulur ve boyutsal küçülme yapılarak özyüz alınır

        results = svm_model.predict_proba(eigen_image)[0] # oluşturulan özyüz SVM modeline sokularak tahmin alınır

        predict = results.argmax() # tahmin edilen en büyük olasılığa sahip sınıf bulunur
        score = results[predict] # tahminin skoru alınır

        text = "%s : %0.2f"%(persons[predict],score) # opencv'ye yazılacak mesaj
        cv2.putText(img,text,(x,y),font,1,(0,255,0),2) # mesaj opencv'ye yazılır
    return img # görüntü döndürülür

cap = cv2.VideoCapture(0) # video başlatılır

while True:
    ret, frame = cap.read() #video kare kare okunur

    if ret == False:
        break

    frame = face_rec(frame, color='bgr') # kare özel fonksiyona sokulur
    cv2.imshow("Facial Recognition", frame) # kare gösterilir
    if cv2.waitKey(20) == ord("q"): # "q" tuşuna basılırsa çıkılır
        break

cv2.destroyAllWindows() # bütün pencereler yokedilir
cap.release() # kaynaklar salınır