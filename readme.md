# Problem definition
**Purpose**

The aim of the project is to perform facial detection in the cars using machine learning. And as constraint deep-learning methods cannot be used.


# Requirements

- As stated above deep-learing methods cannot be used.
- Application should recognize the authorized user of a vehicle.
- Application should work on a real-time environment.

# Method

In this project PCA was used to reduce the workload and determine the number of components needed to classfiy. And then and SVM model is used to perform the actual classification.

# Ease of Use 

- First, the codes in the "face_from_video.py" file are run and the face images are taken as 150x150 and grayscale in the dataset is formed from the video images. In this file, the path (path) where the codes on the 5th, 6th and 7th lines will be created, Line 13 indicates the video to be used, line 32 indicates the name format of the recorded images. By playing with these lines and the counter value on the 37th line, different numbers of different images can be obtained.

- In the second step, the codes in the "dataset_from_image.py" file are run. These codes transform the pixel data and labels of the images in the folders created in the previous step into a dataframe, with each row representing one image.

- After, the codes in the "PCA.py" file are run. The dataframe created here is loaded and the PCA model is created by plotting the explained variance rate and the total explained variance rate graphs, and the output data is recorded by inserting the data into this model. Here, you can change the 28th and 39th lines to see the explained variance rate for the ranges you want and the total explained variance rate. In addition, you can create different PCA models by changing the parameters in line 44.

- Fourthly, the codes in the "SVM.py" file are run. In this part, the SVM model is created by working with the PCA applied data in the previous step. 17-30. A GridSearchCV searches between rows for optimum parameters for the model. Here you can test for different values ​​by changing the values ​​in the "parameters" dictionary. I would like to note here that this process takes forever for less than 4 principal components. In addition, you can create different SVM models by changing the parameters in line 34.

- In the fifth step, the codes in the "test_realtime.py" file are run. In this section, the created model is tested in real time. Line 5 loads the SVM model, line 6 the PCA model, and line 7 the recorded average data. You can load the models you want by changing this line.

- And of course, you must have a camera for real-time testing.
