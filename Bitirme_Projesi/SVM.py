import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

data = np.load("data_pca_150.pickle.npz") #Verimizi yüklüyoruz

X = data['arr_0'] #Görüntü verilerimiz
y = data['arr_1'] #Etiket verilerimiz

print(X.shape)
print(y)

model_tuner = SVC() # model oluşturmadan önce en iyi parametreleri bulmak için boş classifier

parameters = {"C": [1, 10, 20, 30, 40, 50, 100],
              "kernel": ["rbf", "poly", "linear", "sigmoid"],
              "gamma": ["scale", "auto"],
             } # test edilecek parametreler, C, kernel metodları ve kernel metodları için gamma

model_grid = GridSearchCV(model_tuner, param_grid = parameters, scoring = "accuracy", cv = 5, verbose = 1) #test parametreleri modele uydurularak uygulanır
model_grid.fit(X, y) # veriler test modeline sokulur

print("Best parameters: ") # en iyi parametreler yazdırılır
print(model_grid.best_params_)
print("Best score: ") # en iyi parametrenin puanı yazdırılır
print(model_grid.best_score_)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y) # veri train ve test olarak ayrılır

model = SVC(C=1, kernel='linear', gamma="scale", random_state = 42, probability = True) # model oluşturulur buradaki parametreler yukarıdaki işlemin çıktısına göre değiştirilebilir
model.fit(x_train,y_train) # veriler modele sokulur
print('Model has been trained with succes.')

print("Training score:" ) # eğitim skoru yazdırılır
print(model.score(x_train,y_train))
print("Test score: ") # test skoru yazdırılır
print(model.score(x_test,y_test))

y_pred = model.predict(x_test) # x yani görüntü verileri kullanılarak etiket tahmini yapılır

conf_mat = metrics.confusion_matrix(y_test, y_pred) # karmaşıklık matrisi yazdırılır
print("Confusion matrix: ")
print(conf_mat)

report = metrics.classification_report(y_test, y_pred, target_names= ["authorized", "other"], output_dict = True) # classification report yazdırlır
print("Classification report: ")
print(pd.DataFrame(report).T)

kappa_score = metrics.cohen_kappa_score(y_test, y_pred) #kappa skoru yazdırılır
print("Kappa score:")
print(kappa_score)

pickle.dump(model, open("SVM_150c.pickle", "wb")) #model kaydedilir





