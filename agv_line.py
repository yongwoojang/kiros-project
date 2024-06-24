import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import time

agv = MyAgv("/dev/ttyAMA2", 115200)

def main():
    camera = cv2.VideoCapture(0)    #640*480

    cmt = 0
    CL_count = 0
    CR_count = 0
    state = 0
    t_state = 0
    

    while(camera.isOpened()):
        ret, frame = camera.read()
        
        crop_img_yel = frame[200:480,]
        cv2.imshow('normal_yel', crop_img_yel)

        hsv_yel = cv2.cvtColor(crop_img_yel, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        yel_mask = cv2.inRange(hsv_yel, lower_yellow, upper_yellow)

        cv2.imshow('mask_yel', yel_mask)

        contours_yel, hierarchy = cv2.findContours(yel_mask.copy(), 1, cv2.CHAIN_APPROX_NONE)

        if len(contours_yel) > 0:
            c = max(contours_yel, key=cv2.contourArea)
            M = cv2.moments(c)
            CL_count = 0
            CR_count = 0
            state = 0  
            t_state = 0
            
            if M['m00'] != 0:  
                cx = int((M['m10']/M['m00']))
                cmt = cx
                if(cx < 320):
                    cmt = 1
                else:
                    cmt = 0
                print(cx)
                
                
                if cx <= 240:           
                    print("Turn Left!")
                    if(cx <= 160):
                        agv.move_control(128, 128, 131)
                        time.sleep(0.05)
                    else:
                        agv.move_control(128, 128, 129)
                        time.sleep(0.05)
                    agv.move_control(129, 128, 128)
                    time.sleep(0.05)
                
                elif cx >= 400:
                    print("Turn Right")
                    if(cx >= 480):
                        agv.move_control(128, 128, 125)
                        time.sleep(0.05)
                    else:
                        agv.move_control(128, 128, 127)
                        time.sleep(0.05) 
                    agv.move_control(129, 128, 128)
                    time.sleep(0.05) 

                else:
                    print("go")
                    agv.move_control(129, 128, 128)
                    time.sleep(0.1)
   

            if(cmt == 1):  
                      
                if(CL_count == 40):
                    print("STOP_Turn_Left!")
                    print(CL_count)
                    print(CR_count)
                    
                    if(state == 0):
                        print("Turn!!!!!!!!!!!!!!!")
                        agv.move_control(128, 128, 125)
                        time.sleep(2) 
                        state = 1  
                    else:
                        if(CR_count == 40):
                            if(t_state == 0):
                                print("Turn!!!!!!!!!!!!!!!")
                                agv.move_control(128, 128, 131)
                                time.sleep(2) 
                                
                                print("Left_go")
                                agv.move_control(128, 131, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                agv.move_control(131, 128, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                agv.move_control(128, 125, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                t_state = 1
                            else:
                                agv.stop()
                        else:    
                            agv.move_control(128, 128, 125)
                            time.sleep(0.05) 
                            CR_count = CR_count + 1



                else:
                    agv.move_control(128, 128, 131)
                    time.sleep(0.05)
                    CL_count = CL_count+1                    

            elif(cmt == 0):
                if(CR_count == 40):
                    print("STOP_Turn_Right!")
                    print(CR_count)
                    print(CL_count)

                    if(state == 0):
                        print("Turn!!!!!!!!!!!!!!!")
                        agv.move_control(128, 128, 131)
                        time.sleep(2) 
                        state = 1
                    else:
                        if(CL_count == 40):
                            if(t_state == 0):
                                print("Turn!!!!!!!!!!!!!!!")
                                agv.move_control(128, 128, 125)
                                time.sleep(2) 
                                
                                print("Left_go")
                                agv.move_control(128, 131, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                agv.move_control(131, 128, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                agv.move_control(128, 125, 128)
                                time.sleep(10)
                                agv.stop()
                                time.sleep(1)

                                t_state = 1
                            else:
                                agv.stop()
                        else:
                            agv.move_control(128, 128, 131)
                            time.sleep(0.05) 
                            CL_count = CL_count + 1
                    

                else:
                    agv.move_control(128, 128, 125)
                    time.sleep(0.05)
                    CR_count = CR_count+1   

            else:
                agv.stop()
            

        if cv2.waitKey(1) == ord('q'):
            agv.stop()
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
