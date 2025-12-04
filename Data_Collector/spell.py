### Relevant Lib

import cv2
import time
import queue
import random
import pyautogui
import threading
import numpy as np
from mss import mss
from PIL import Image

def detect(needle, haystack, threshold=0.98):
    res = cv2.matchTemplate(haystack, needle, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    p1 = max_loc
    p2 = (p1[0] + needle.shape[:2][1], p1[1] + needle.shape[:2][0])
    if max_val > threshold:
        return True, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    else:
        return False, 0, 0

### Constants Used 
scale = 0.5
left = {'top': 60, 'left': 450, 'width': 370, 'height': 654}
right = {'top': 60, 'left': 870, 'width': 370, 'height': 654}

exit_img = cv2.imread("assets/exit.png", cv2.IMREAD_COLOR)
exit_img = cv2.resize(exit_img, (0, 0), fx=scale, fy=scale)
itr = 625
while True:
    sct = mss()
    frame = np.array(sct.grab(left))
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    exit_detected, x, y = detect(exit_img, frame)

    ### Restarts Friendly Match in Both Screens 4
    if exit_detected:
            x = x + right['left']
            y = y + right['top']
            time.sleep(0.1)
            pyautogui.click(x, y)
            time.sleep(0.1)
            pyautogui.click(x, y)
            time.sleep(0.1)
            pyautogui.click(x-450, y)
            time.sleep(0.1)
            pyautogui.click(x-450, y)
            time.sleep(10)
            pyautogui.click(right["left"] + 155, right["top"] + 555) 
            time.sleep(0.1)
            pyautogui.click(right["left"] + 155, right["top"] + 555) 
            time.sleep(3)
            pyautogui.click(right["left"] + 175, right["top"] + 180) 
            time.sleep(3) 
            pyautogui.click(left["left"] + 300, left["top"] + 480)
            time.sleep(0.1)
            pyautogui.click(left["left"] + 300, left["top"] + 480)
    else:
        pyautogui.click(random.randint(left['left'] + 20, left['left'] + left['width'] - 20),
            random.randint(left['top'] + 0, left['top'] + 500))
        pyautogui.press(random.choice(["1", "2", "3", "4"]))
        pyautogui.click(random.randint(left['left'] + 20, left['left'] + left['width'] - 20),
            random.randint(left['top'] + 0, left['top'] + 500))
        
        pyautogui.click(random.randint(right['left'] + 20, right['left'] + right['width'] - 20),
            random.randint(right['top'] + 300, right['top'] + 500))
        pyautogui.press(random.choice(["1", "2", "3", "4"]))
        pyautogui.click(random.randint(right['left'] + 20, right['left'] + right['width'] - 20),
            random.randint(right['top'] + 300, right['top'] + 500))
        11
        itr += 1
        time.sleep(random.randint(14, 16)/10)
        arena = sct.grab(right)
        arena_img = Image.frombytes("RGB", arena.size, arena.rgb)
        arena_img.save(f"Data_Collector/spell/{itr}.png")
        arena_img = np.array(arena)
        print(f"saved {itr}")
        arena_img = np.array(arena)
        arena_img = cv2.resize(arena_img, (0, 0), fx=scale, fy=scale)
        arena_img = cv2.cvtColor(arena_img, cv2.COLOR_RGBA2RGB)
        cv2.imshow("screen", arena_img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    time.sleep(2.8*3)