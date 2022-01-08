import cv2
import os
import imutils

personName = 'Authorized' #Kullanıcının alt klasörü
dataPath = "images" # Kullanıcı görüntülerinin klasörü
personPath = dataPath + '/' + personName # oluşturulacak yol (dosyalar bütünü) images/Authorized

if not os.path.exists(personPath): # eğer bu path yoksa;
	print('Folder creadted to hold data: ',personPath) # Klaösrlerin yaratıldığını belirt
	os.makedirs(personPath) # Klasörleri yarat

cap = cv2.VideoCapture(r".\vids\Dash\Female\5-FemaleNoGlasses.avi") #Video capture'yi başlat

faceClassif = cv2.CascadeClassifier(r"cascades\data\haarcascade_frontalface_default.xml") #Classifier'ini (sınıfını) oluştur.
count = 0 #Sayacı başlat

while True: #while döngüsü

	ret, frame = cap.read() #frame frame oku
	if ret == False: # okuyamazsan döngüyü kır
		break
	frame =  imutils.resize(frame, width = 640, height = 480) #frame'i resize et
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale (gri tonlamalıya çevir)

	faces = faceClassif.detectMultiScale(gray,1.3,5) # grayscale haline classifier'e sok

	for (x,y,w,h) in faces: #bulunan her yüzün koordinatları için;
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #yüzün etrafına rectangle'ı çiz
		face = gray[y:y+h,x:x+w] # rectangle'ın içindeki yüz görüntüsünü al
		face = cv2.resize(face, (150,150), interpolation = cv2.INTER_CUBIC) # görüntüyü 150x150 oalcak şekilde resize et
		cv2.imwrite(personPath + '/authorized_{}.jpg'.format(count),face) # görüntüyü klasöre "isim_sayac.jpg" olarak (örn: authorized_1.jpg) kaydet
		count = count + 1 #sayacı arttır
	cv2.imshow('frame',frame) #frame'i kullanıcıya göster

	k =  cv2.waitKey(1)
	if k == 27 or count >= 600: # sayac 600'e geldiyse veya kullanıcı tuşa basarsa döngüyü kır
		break

cap.release() # VideoCapture objesini sal
cv2.destroyAllWindows() # Bütün pencereleri yok et