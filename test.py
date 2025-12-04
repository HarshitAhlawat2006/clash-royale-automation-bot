
import cv2
import time 
import torch
import pyautogui
import numpy as np
from mss import mss
from PIL import Image
import torch.nn as nn
from ultralytics import YOLO
import multiprocessing as mp
import torch.nn.functional as F
from torchvision import transforms, models

def load_resnet_classifier():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 4)
    )

    model.load_state_dict(torch.load("Models/resnet50.pth", map_location=device))
    model.to(device)
    model.eval()

    # Class names
    class_names = ['Flying_Machine', 'Goblin_Cage', 'Hog_Rider', 'Valkyrie']

    # Preprocessing transform
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return model, class_names, eval_transform, device

def predict_card(img, model, class_names, eval_transform, device):
    import torch.nn.functional as F

    tensor = eval_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        idx = probs.argmax(1).item()
        conf = probs[0, idx].item()

    return class_names[idx], conf

def load_yolo_model():
    from ultralytics import YOLO
    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔋 YOLO using device: {device}")

    model = YOLO("Models/yolo.pt")

    model.to(device)
    model.conf = 0.9   # confidence threshold
    model.iou = 0.5    # NMS IOU
    model.eval()

    return [model, device]

def detect(needle, haystack, threshold=0.975):
    res = cv2.matchTemplate(haystack, needle, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    p1 = max_loc
    p2 = (p1[0] + needle.shape[:2][1], p1[1] + needle.shape[:2][0])
    if max_val > threshold:
        return True, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    else:
        return False, 0, 0


########################################
# WORKER PROCESS A : YOLO
########################################
import time
import math

def worker_A(input_queue, output_queue):
    model, device = load_yolo_model()

    conf_threshold = 0.8
    cooldown = 1.2                 # time threshold
    position_threshold = 200       # pixel distance threshold

    last_seen_time = {}            # class_name → time
    last_seen_pos = {}             # class_name → (cx, cy)

    while True:
        frame = input_queue.get()
        if frame is None:
            break

        results = model.predict(source=frame, verbose=False, device=device)
        boxes = results[0].boxes.data.cpu().numpy()

        if len(boxes) == 0:
            continue

        for det in boxes:
            x1, y1, x2, y2, conf, cls = det
            card_name = results[0].names[int(cls)]
            conf = float(conf)
            now = time.time()

            if conf < conf_threshold:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            new_pos = (cx, cy)

            if card_name in last_seen_time:
                time_diff = now - last_seen_time[card_name]
                last_pos = last_seen_pos[card_name]
                dist = math.dist(new_pos, last_pos)

                # Duplicate prevention
                if time_diff < cooldown and dist < position_threshold:
                    print(card_name, last_seen_pos[card_name], new_pos, dist)
                    last_seen_pos[card_name] = new_pos
                    continue

            # Accept new spell
            last_seen_time[card_name] = now
            last_seen_pos[card_name] = new_pos

            output_queue.put([card_name, conf])


########################################
# WORKER PROCESS B : Resnet 
########################################
import time
import math

def worker_B(input_queue, output_queue):
    model, class_names, eval_transform, device = load_resnet_classifier()

    conf_threshold = 0.9

    last_detect = None
    last_time = 0
    cooldown = 1.0  # seconds

    while True:
        img = input_queue.get()
        if img is None:
            break

        # img is a PIL Image (crop of troop)
        card_name, conf = predict_card(img, model, class_names, eval_transform, device)

        output_queue.put([card_name, conf])


########################################
# MAIN PROCESS
########################################
def main():
    # Queues for communication
    queue_A_in  = mp.Queue(maxsize=5)
    queue_A_out = mp.Queue(maxsize=5)

    queue_B_in  = mp.Queue(maxsize=5)
    queue_B_out = mp.Queue(maxsize=5)

    # Start both worker processes
    proc_A = mp.Process(target=worker_A, args=(queue_A_in, queue_A_out))
    proc_B = mp.Process(target=worker_B, args=(queue_B_in, queue_B_out))

    proc_A.start()
    proc_B.start()

    print("Main started. Sending work...")

    i = 0
    prev_card = None
    start = time.time()
    hand = ["?" for i in range(8)]
    card_height, card_width, scale = 50, 20, 0.5
    screen = {'top': 65, 'left': 880, 'width': 370, 'height': 654}
    needle = cv2.imread("spawn.png", cv2.IMREAD_COLOR)
    needle = cv2.resize(needle, (0, 0), fx=scale, fy=scale)

    last_spawn = (0,0)
    threshold_dist = 10

    sct = mss()
    print("Started")
    while pyautogui.position()[0] > 50:
        ### Grabs Game Screen
        frame = np.array(sct.grab(screen))
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if not queue_A_in.full():
            queue_A_in.put(frame)

        spawn_detected, x, y = detect(needle, frame)
        if spawn_detected:
            x = int(x + screen['left'])
            y = int(y + screen['top'])

            if math.dist(last_spawn, (x,y)) > threshold_dist:

                ### Capturing Card Near Detected Spawn Region
                card_box = {'top': y - card_height, 'left': x - card_width, 'width': 2 * card_width, 'height': int(1.5 * card_height)}
                troop = sct.grab(card_box)
                troop_img = Image.frombytes("RGB", troop.size, troop.rgb).convert("RGB")
                queue_B_in.put(troop_img)
            last_spawn = (x, y)
        resultA = [None, None]
        resultB = [None, None]
        if not queue_A_out.empty():
            resultA = queue_A_out.get()
        if not queue_B_out.empty():
            resultB = queue_B_out.get()

        if resultA[0] == None:
            if resultB[0] != None:
                print(f"Troop Deployed = {resultB[0]}, Conf = {resultB[1]}")
        else:
            print(f"Spell Deployed = {resultA[0]}, Conf. = {resultA[1]}")

        cv2.imshow("Live Window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    # ---- SEND DATA TO WORKERS ----
    #for i in range(10):
    #    data = f"task-{i}"
#
    #    if not queue_A_in.full():
    #        queue_A_in.put(data)
#
    #    if not queue_B_in.full():
    #        queue_B_in.put(data)
#
    #    time.sleep(0.1)
#
#
    ## ---- READ RESULTS FROM WORKERS ----
    #for _ in range(10):
    #    if not queue_A_out.empty():
    #        resultA = queue_A_out.get()
    #        print("From A:", resultA)
#
    #    if not queue_B_out.empty():
    #        resultB = queue_B_out.get()
    #        print("From B:", resultB)
#
    #    time.sleep(0.1)

    # ---- STOP WORKERS ----
    queue_A_in.put(None)
    queue_B_in.put(None)

    proc_A.join()
    proc_B.join()

    print("All done.")


if __name__ == "__main__":
    mp.set_start_method("spawn")  # required on macOS
    main()