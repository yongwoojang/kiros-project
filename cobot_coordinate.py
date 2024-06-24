import cv2
import numpy as np
from typing import Tuple, Any
import sys
import os
from typing import *
from dataclasses import dataclass, astuple

from pymycobot.mycobot import MyCobot
from pymycobot.genre import Coord
import time


mc = MyCobot('COM8', 115200)
mc.set_gripper_mode(0)
mc.init_eletric_gripper()
coords = mc.get_coords()
mc.set_tool_reference([0,0,0,0,0,0])
tool_coords = mc.get_tool_reference()
print(coords)

sys.path.append(os.getcwd())

def base_move(mc): 
    mc.set_gripper_value(100, 30, 1)
    time.sleep(1)
    mc.send_coords([254, 63, 328, 167.3, 8.4, (-178.3)], 30)
    time.sleep(3)
def move_red(mc, initial_r=178):
    r = initial_r
    mc.send_coords([277.8, 35, 290.7, (-178.5), 4.8, 176], 25)
    time.sleep(4)
    mc.set_gripper_value(10, 20, 1)
    time.sleep(3)
    mc.send_coords([210.7, 17.5, 390, 167.4, 25.5, 176], 30)
    time.sleep(2)
    mc.send_coords([59.3, 199, 350, (-178), 9.5, (-90.2)], 30)
    time.sleep(4)
    mc.send_coords([-87.8, 281.45, r, (-176.3), 0, (-90)], 20)
    time.sleep(5)
    mc.set_gripper_value(100, 10, 1)
    time.sleep(3)
    mc.send_coords([63, 165.3, 412.67, 161, 33, -117], 30)
    time.sleep(3)
    return r + 29
def move_purple(mc, initial_p = 180):
    p = initial_p
    mc.send_coords([277.8, 35, 290.7, (-178.5), 4.8, 176], 25)
    time.sleep(4)
    mc.set_gripper_value(10, 20, 1)
    time.sleep(3)
    mc.send_coords([210.7, 17.5, 390, 167.4, 25.5, 176], 30)
    time.sleep(2)
    mc.send_coords([59.3, 199, 350, (-178), 9.5, (-90.2)], 30)
    time.sleep(4)
    mc.send_coords([-17.8, 281.45, p, (-176.3), 0, (-90)], 20)
    time.sleep(5)
    mc.set_gripper_value(100, 10, 1)
    time.sleep(3)
    mc.send_coords([63, 165.3, 412.67, 161, 33, -117], 30)
    time.sleep(3)
    return p + 29
def move_yellow(mc, initial_y = 181):
    y = initial_y
    mc.send_coords([277.8, 35, 290.7, (-178.5), 4.8, 176], 20)
    time.sleep(5)
    mc.set_gripper_value(10, 20, 1)
    time.sleep(3)
    mc.send_coords([210.7, 17.5, 390, 167.4, 25.5, 176], 30)
    time.sleep(2)
    mc.send_coords([59.3, 199, 350, (-178), 9.5, (-90.2)], 30)
    time.sleep(4)
    mc.send_coords([67.8, 281.45, y, (-176.3), 0, (-90)], 20)
    time.sleep(5)
    mc.set_gripper_value(100, 10, 1)
    time.sleep(3)
    mc.send_coords([63, 165.3, 412.67, 161, 33, -117], 30)
    time.sleep(3)

    return y + 28
# def move_yellow(mc, initial_y = 182):
#     y = initial_y
#     mc.send_coords([277.8, 35, 290.7, (-178.5), 4.8, 176], 20)
#     time.sleep(5)
#     mc.set_gripper_value(10, 20, 1)
#     time.sleep(3)
#     mc.send_coords([210.7, 17.5, 390, 167.4, 25.5, 176], 30)
#     time.sleep(4)
#     mc.send_coords([59.3, 199, 350, (-178), 9.5, (-90.2)], 30)
#     time.sleep(4)
#     mc.send_coords([137.8, 281.45, y, (-176.3), 0, (-90)], 20)
#     time.sleep(5)
#     mc.set_gripper_value(100, 10, 1)
#     time.sleep(3)
#     mc.send_coords([63, 165.3, 412.67, 161, 33, -117], 30)
#     time.sleep(3)
#     return y + 28
def get_angle_from_rect(corners: np.ndarray) -> int:
    center, size, angle = cv2.minAreaRect(corners)
    angle = angle - 360 if angle > 360 else angle
    return angle
def crop_and_resize_frame(frame: np.ndarray, crop_width: int, crop_height: int, scale_factor: float) -> np.ndarray:
    # 원본 프레임의 크기
    frame_height, frame_width = frame.shape[:2]

    # 원본 프레임에서 가운데 부분을 잘라냄
    start_x = (frame_width - crop_width) // 2
    start_y = (frame_height - crop_height) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    # 크기 조정
    resized_frame = cv2.resize(cropped_frame, None, fx=scale_factor, fy=scale_factor)

    return resized_frame


class ColorDetector:
    @dataclass
    class DetectResult:
        color: str
        corners: np.ndarray

        def __iter__(self):
            return iter(astuple(self))

    def __init__(self) -> None:
        self.area_low_threshold = 10000
        self.detected_name = None
        self.hsv_range = {

            "blue": ((91, 100, 100), (105, 256, 256)),
            "yellow": ((20, 200, 100), (30, 256, 256)),
            "redA": ((0, 100, 100), (6, 256, 256)),
            "redB": ((170, 100, 100), (179, 256, 256)),
            "purple": ((125, 50, 50), (150, 256, 256))

        }

    def get_radian(self, res: DetectResult):
        return get_angle_from_rect(res.corners) / 180 * np.pi

    def detect(self, frame: np.ndarray):
        """Detect certain color in HSV color space, return targets min bounding box.

        Args:
            frame (np.ndarray): Src frame
            hsv_low (Tuple[int, int, int]): HSV lower bound
            hsv_high (Tuple[int, int, int]): HSV high bound

        Returns:
            List[Tuple[int, int, int, int]] | None: list of bounding box or empty list
        """
        result = []
        for color, (hsv_low, hsv_high) in self.hsv_range.items():
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            in_range = cv2.inRange(hsv_frame, hsv_low, hsv_high)

            # 对颜色区域进行膨胀和腐蚀
            kernel = np.ones((2, 2), np.uint8)
            in_range = cv2.morphologyEx(in_range, cv2.MORPH_CLOSE, kernel)
            in_range = cv2.morphologyEx(in_range, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(
                in_range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contours = list(
                filter(lambda x: cv2.contourArea(x) > self.area_low_threshold, contours)
            )

            rects = list(map(cv2.minAreaRect, contours))
            boxes = list(map(cv2.boxPoints, rects))
            boxes = list(map(np.int32, boxes))

            if len(boxes) != 0:
                if color.startswith("red"):
                    color = "red"
                for box in boxes:
                    result.append(ColorDetector.DetectResult(color, box))
                    # self.detected_name = result
                    self.detected_name = result[0].color
        return result

    def draw_result(self, frame: np.ndarray, res: List[DetectResult]):
        for obj in res:
            cv2.drawContours(frame, [obj.corners], -1, (0, 0, 255), 3)
            cv2.putText(
                frame,
                obj.color,
                self.target_position(obj),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1,
            )

    def target_position(self, res: DetectResult) -> Tuple[int, int]:
        pos = np.mean(np.array(res.corners), axis=0).astype(np.int32)
        x, y = pos
        return x, y
    def get_rect(self, res: DetectResult):
        return res.corners


def main():
    # 웹캠 화면에서 가운데 부분을 자를 크기와 확대할 비율 설정
    crop_width = 200
    crop_height = 200
    scale_factor = 1.4

    # Create an instance of ColorDetector
    detector = ColorDetector()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    mc.set_reference_frame(1)


    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    # Flag to indicate if base_move should be called
    call_base_move = True
    r = 178
    p = 180
    y = 181
    

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame is captured successfully
        if not ret:
            print("Error: Couldn't read frame from the webcam.")
            break

        # Crop and resize the frame
        cropped_resized_frame = crop_and_resize_frame(frame, crop_width, crop_height, scale_factor)

        # Detect colors in the frame
        detections = detector.detect(cropped_resized_frame)

        # Draw results on the frame
        detector.draw_result(cropped_resized_frame, detections)

        # Display the frame
        cv2.imshow("Color Detection", cropped_resized_frame)
        
        # Check for key press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        mc.set_fresh_mode(0)
        base_move(mc)
        for detection in detections:
            if detection.color == "red":
                r = move_red(mc, r)
            elif detection.color == "purple":
                p = move_purple(mc, p)
            elif detection.color == "yellow":
                y = move_yellow(mc, y)

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
