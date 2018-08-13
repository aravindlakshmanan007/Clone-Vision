import cv2
import numpy as np
import os
from PIL import Image

cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id=input('enter your id')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize= (30,30))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("imagesdb/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()

#dbgenerator
def getImagesAndLabels(path):
    #get the path of all files in the folder
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    #create empty face list
    faceSamples=[]

    #create empty ID list
    Ids=[]

    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        
        #Convert PIL images to numpy arrays
        imageNp = np.array(pilImage,'uint8')

        #Get the ID from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        #Extract the face from the training samples
        faces = detector.detectMultiScale(imageNp, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize= (30,30))

        for(x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('imagesdb')
recognizer.train(faces, np.array(Ids))
recognizer.save('dbgenerator/dbgenerator.yml')

cv2.destroyAllWindows()

        
