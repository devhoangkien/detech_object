#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def run(frame):
       try:
              parser=argparse.ArgumentParser()
              parser.add_argument('--image')

              args=parser.parse_args()

              faceProto="./gender_age/opencv_face_detector.pbtxt"
              faceModel="./gender_age/opencv_face_detector_uint8.pb"
              ageProto="./gender_age/age_deploy.prototxt"
              ageModel="./gender_age/age_net.caffemodel"
              genderProto="./gender_age/gender_deploy.prototxt"
              genderModel="./gender_age/gender_net.caffemodel"

              MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
              ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
              genderList=['Male','Female']

              faceNet=cv2.dnn.readNet(faceModel,faceProto)
              ageNet=cv2.dnn.readNet(ageModel,ageProto)
              genderNet=cv2.dnn.readNet(genderModel,genderProto)

              # video=cv2.VideoCapture(args.image if args.image else 0)
              padding=20
              resultImg,faceBoxes=highlightFace(faceNet,frame)
              if not faceBoxes:
                            print("No face detected")
              data = {"gender": {},"age":{}}
              for faceBox in faceBoxes:
                            face=frame[max(0,faceBox[1]-padding):
                                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                                   :min(faceBox[2]+padding, frame.shape[1]-1)]

                            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                            genderNet.setInput(blob)
                            genderPreds=genderNet.forward()
                            gender=genderList[genderPreds[0].argmax()].replace('(','').replace(')','').replace('-','_').lower()
                            ageNet.setInput(blob)
                            agePreds=ageNet.forward()
                            age=ageList[agePreds[0].argmax()].replace('(','').replace(')','').replace('-','_')
                            if(gender not in data['gender']):
                                   data['gender'][gender] = 1
                            else:
                                   data['gender'][gender] += 1
                            if(age not in data['age']):
                                   data['age'][age] = 1
                            else:
                                   data['age'][age] += 1
              return data
       except Exception as e:
              print(e)