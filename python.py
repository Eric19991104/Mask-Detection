import dlib
import cv2 as cv
import simpleaudio as sa
import numpy as np
def sound ( x , z ) :
 frequency = x # Our played note will be 440 Hz
 fs = 44100  # 取樣頻率
 seconds = z   # 持續時間為3秒

 # 生成具有秒乘以取樣頻率步驟的陣列，範圍在0到秒之間
 t = np. linspace ( 0 , seconds , seconds * fs , False )

 # 產生440 Hz正弦波
 note = np. sin ( frequency * t * 2 * np. pi )

 # 確保最大值在16位元範圍內
 audio = note * ( 2 ** 15 - 1 ) / np. max ( np. abs ( note ) )
 # 轉換為16位元資料
 audio = audio. astype ( np. int16 )

 # 開始播放
 play_obj = sa. play_buffer( audio , 1 , 2 , fs )

 # 等待播放完成退出
 play_obj. wait_done ()

detector = dlib.get_frontal_face_detector()                                                                         #檢測人臉變數

color = ('b','g','r')
cap = cv.VideoCapture(0)                                                                                            #抓取鏡頭 (如果你有的話)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 650)                                                                               #設定影像尺寸大小
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

near_threshold = 0.0001                                                                                               #上1/3跟下2/3相似度為0.01

while(cap.isOpened()):                                                                                              #如果cap能夠成功打開
    ret,frame = cap.read()
    face_rects, scores, idx = detector.run(frame, 0)
    
    for i, d in enumerate(face_rects):
        x1 = d.left()                                                                                               # 取得左上角的座標、右下角座標
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        
        faceLong = y2-y1
        upface = frame[y1:y1+int(1/3*faceLong),x1:x2]                                                               # 把抓到的人臉分成上 1/3 與下 2/3
        downface = frame[y1+int(1/3*faceLong):y2,x1:x2]
        
        #因為上半臉與下半臉的pixel數量不同，所以我們會先做正規化normalize。並計算兩者之間的相似度。
        hist1 = cv.calcHist([upface], [0,1,2], None, [256,256,256], [0, 256, 0, 256,0, 256])
        hist2 = cv.calcHist([downface], [0,1,2], None, [256,256,256], [0, 256, 0, 256,0, 256])
        
        # 平移縮放
        cv.normalize(hist1, hist1, 0, 1.0, cv.NORM_MINMAX)
        cv.normalize(hist2, hist2, 0, 1.0, cv.NORM_MINMAX)
        # 比較方法 HISTCMP_CORREL (相關姓) 0、HISTCMP_CHISQR (卡方) 1、HISTCMP_BHATTACHARYYA(巴式) 2
        near = cv.compareHist(hist1,hist2,0)
        
        if(near >= near_threshold):
            # 畫框框與寫字
            cv.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4,cv.LINE_AA)
            cv.putText(frame,"No Mask", (x1, y1), cv.FONT_HERSHEY_DUPLEX,0.7, (0,0,255), 1, cv.LINE_AA)
            sound ( 300 , 2 )
    cv.imshow("Face Detection", frame)
    
    if(cv.waitKey(1) & (0xFF == ord('q'))):                                                                           # 按下q離開
        break
        
cap.release()                                                                                                       #關閉視窗
cv.destroyAllWindows()