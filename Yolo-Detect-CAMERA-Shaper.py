#encoding:utf-8
import cv2
import numpy as np 
from cv2 import dnn
import time

confThreshold = 0.5


#net, img_size = dnn.readNetFromDarknet("my_yolov4-tiny-custom.cfg","my_yolov4-tiny-custom_best.weights"), 416
net, img_size = dnn.readNetFromDarknet("yolo-fastest.cfg","yolo-fastest_best.weights"), 320
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(DNN_TARGET_CPU)

label_Name = ['car','bus','truck','taxi','motocycle','people']
list_FPS = []

def getOutputsNames(net):
    layersNames = net.getLayerNames()

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
 
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.5)

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(frame,(left,top),(left+width,top+height),(0,0,255))

    StrClassCNT = ' Car:'+str(classCNT[0]) + ' BUS:'+str(classCNT[1]) + ' TRUCK:'+str(classCNT[2]) + ' TAXI:'+str(classCNT[3]) + ' MOTOCYCLE:'+str(classCNT[4]) + ' PEOPLE:'+str(classCNT[5])
    print(StrClassCNT)
    return frame

def sharper(img):
    KSIZE = 11
    ALPHA = 2
    kernel = cv2.getGaussianKernel(KSIZE, 0)
    kernel = -ALPHA * kernel @ kernel.T
    kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered[:, :, [0, 1, 2]]

def detect(img):

    img = cv2.resize(img,(img_size,img_size))
    w = img.shape[1]
    h = img.shape[0]

    blob = dnn.blobFromImage(img,1/255.0)

    net.setInput(blob)

    layername = getOutputsNames(net)

    detections = net.forward(layername)

    img = postprocess(img,detections)


    return img 

def main():

    #cap = cv2.VideoCapture("Taipei_Inertsection.mp4")
    cap = cv2.VideoCapture(0)
    index = 0

    prev_frame_time = 0
    
    # used to record the time at which we processed current frame
    new_frame_time = 0

    while cap.isOpened():
        index +=1
        #print('index:'+str(index))
        #if (index%10 == 0):
        if True:
        #if index<=500:
            ret,img = cap.read()

            new_frame_time = time.time()            
            if img is not None:
                cv2.imshow("Original",img)
                img = cv2.medianBlur(img, 3)
                blur = sharper(img)
                cv2.imshow("Sharper",blur)

                #img = detect(img)
                img = detect(blur)
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                cv2.putText(img, str(float(round(fps,1))), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)                
                cv2.imshow("Yolo-Detect",img)
        else:
            break

        # 按 ESC 键结束
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()