import numpy as np
import cv2
import os


def labels_for_training(directory):
    faces=[]
    faceID=[]
    trainX=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith('.'):
                print("skipping the file: " +filename)
                continue
            
            ids=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("img_path: ", img_path)
            print("ids: ",ids)
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image is not loaded properly")
                continue
            face_rect,gray_img=faceDetection(test_img)
            if len(face_rect)!=1:
                continue
            (x,y,w,h)=face_rect[0]
            gray_cap=gray_img[y:y+w,x:x+h]
            roi_gray=cv2.resize(gray_cap,(500,500))
            temp_ndarray=np.reshape(roi_gray,(1,(roi_gray.shape[0]*roi_gray.shape[0])))
            faces.append(roi_gray)
            trainX.append(temp_ndarray)
            faceID.append(int(ids))
            print("labels_for_training")
    temp_numpy=np.array(trainX)
    temparray=temp_numpy.reshape(temp_numpy.shape[0],temp_numpy.shape[2])
    print(temparray.shape)

    return faces,faceID,temparray
            
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def faceDetection(test_img):
    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)
    
    return faces,gray_img
"""
name={0:"Ajith",1:"Suriya"}
# Step1:
directory='images_training'
faces,faceID,x_train=labels_for_training(directory)
print(faceID)
# training the data for the first time and save it so that it will usefull for next iterations
face_recognizer_model=train_classifier(faces,faceID)
face_recognizer_model.save('trainingData.yml')"""


