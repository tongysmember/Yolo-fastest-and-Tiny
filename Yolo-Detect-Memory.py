#encoding:utf-8
import cv2
import numpy as np 
from cv2 import dnn
import time
import os
import psutil
from csv import writer

confThreshold = 0.5

#Model_Name, Path_Cfg, Path_Weight, img_size ='yolov4-tiny', 'my_yolov4-tiny-custom.cfg', 'my_yolov4-tiny-custom_best.weights', 416
Model_Name, Path_Cfg, Path_Weight, img_size ='yolo-fastest', 'yolo-fastest.cfg', 'yolo-fastest_best.weights',320

net = dnn.readNetFromDarknet(Path_Cfg, Path_Weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(DNN_TARGET_CPU)

label_Name = ['car','bus','truck','taxi','motocycle','people']
list_FPS, list_Mem, list_CPU = [], [], []

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs

    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight,frameWidth = frame.shape[0], frame.shape[1]
 
    classIds, confidences, confidences, boxes, classIds, confidences, boxes = [], [], [], [], [], [], []
    classCNT = [0,0,0,0,0,0]
    for out in outs:

        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                #print('Name:'+label_Name[classId])
                classCNT[classId]+=1
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.5)

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        #print(box)
        #cv2.rectangle(frame,(left,top),(left+width,top+height),(0,0,255))

    return frame

def detect(img):

    img = cv2.resize(img,(img_size,img_size))
    #print(img.shape)
    w = img.shape[1]
    h = img.shape[0]

    blob = dnn.blobFromImage(img,1/255.0)

    net.setInput(blob)

    layername = getOutputsNames(net)
    #print("layername:",layername)

    detections = net.forward(layername)

    #print("detections.shape:",len(detections))

    img = postprocess(img,detections)


    return img 

def main():

    cap = cv2.VideoCapture("Taipei_Inertsection.mp4")
    #cap = cv2.VideoCapture(0)
    index = 0

    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0
    print('Yolo-Detect-FPS Start, Waiting Until Detecte Stop!')
    while cap.isOpened():
        index +=1
        #print('index:'+str(index))
        
        #if (index%10 == 0):
        #if True:
        if index<=100:
            ret,img = cap.read()

            new_frame_time = time.time()
                        
            if img is not None:
                img = detect(img)
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                list_CPU.append(str(float(psutil.Process(os.getpid()).cpu_percent())))
                list_Mem.append(str(float(psutil.Process(os.getpid()).memory_percent())))
                #cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)                
                #cv2.imshow("Yolo-Detect",img)

        else:
            break

        # 按 ESC 键结束
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    global Model_Name
    #with open('./FPS-Collect/'+Model_Name+'.txt', 'a', newline='') as f_object:  
    #    writer_object = writer(f_object)
    #    print('list_FPS')
    #    print(list_FPS)
    #    writer_object.writerow(list_FPS)  
    #    f_object.close()
    with open('./FPS-Collect/'+Model_Name+'_CPU.txt', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(list_CPU)  
        f_object.close()
    with open('./FPS-Collect/'+Model_Name+'_Memory.txt', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(list_Mem)  
        f_object.close()
    
def memory_usage_psutil():
    process = psutil.Process(os.getpid()).memory_percent()
    mem = process.memory_percent()
    return mem    

if __name__ == "__main__":
    main()