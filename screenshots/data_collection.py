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

### Constants Used 
scale = 0.5
card_height, card_width = 40, 20
left = {'top': 58, 'left': 455, 'width': 370, 'height': 654}
right = {'top': 58, 'left': 870, 'width': 370, 'height': 654}


### Simple Template Matching
def detect(needle, haystack, threshold=0.98):
    res = cv2.matchTemplate(haystack, needle, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    p1 = max_loc
    p2 = (p1[0] + needle.shape[:2][1], p1[1] + needle.shape[:2][0])
    if max_val > threshold:
        return True, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    else:
        return False, 0, 0

### Used to Detect Spawn and Capture Surrounding Region for Training (Left Screen)
def vision_process(queue):
    print("[Vision] Started")
    sct = mss()
    itr = 400 # For Managing Interupted Runs 
    spawn = cv2.imread("screenshots/spawn.png", cv2.IMREAD_COLOR)
    spawn = cv2.resize(spawn, (0, 0), fx=scale, fy=scale)
    while True:
        ### Capturing Left Screen  
        frame = np.array(sct.grab(left))
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        spawn_detected, x, y = detect(spawn, frame)
        if spawn_detected:
            
            itr += 1
            x = int(x + left['left'])
            y = int(y + left['top'])

            ### Capturing Surrounding Region for Training 
            card_box = {'top': y - card_height, 'left': x - card_width, 'width': 2 * card_width, 'height': int(1.5 * card_height)}
            troop = sct.grab(card_box)
            troop_img = Image.frombytes("RGB", troop.size, troop.rgb)
            troop_img.save(f"screenshots/troop{itr}.png")
            print(f"[Vision] Saved troop{itr}.png") # Optional : For Debuging 
            queue.put(("spawn_detected", x, y))  # send detection info3

### Used to Randomly Spawn Troops in Right Screen
def click_process(queue):
    print("[Clicker] Started")
    sct = mss()
    exit_img = cv2.imread("screenshots/exit.png", cv2.IMREAD_COLOR)
    exit_img = cv2.resize(exit_img, (0, 0), fx=scale, fy=scale)

    while True:
        ### Capture Right Screen 
        frame = np.array(sct.grab(right))
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        exit_detected, x, y = detect(exit_img, frame)

        ### Restarts Friendly Match in Both Screens 
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
            ### Randomly Select and Deploy a Card 
            if not queue.empty():
                msg, det_x, det_y = queue.get()
                if msg == "spawn_detected":
                    print(f"[Clicker] Reacting to spawn at ({det_x}, {det_y})")
            pyautogui.press(random.choice(["1", "2", "3", "4"]))
            pyautogui.click(random.randint(right['left'], right['left'] + right['width'] - 20),
                            random.randint(right['top'] + 300, right['top'] + 500))
            time.sleep(2)

import threading


### Running Two Loops for Both Screen to Reduce Lag 
if __name__ == "__main__":
    queue = queue.Queue()

    t1 = threading.Thread(target=vision_process, args=(queue,))
    t2 = threading.Thread(target=click_process, args=(queue,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()


