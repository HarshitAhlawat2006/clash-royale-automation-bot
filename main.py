### Relevant Lib

import cv2
import torch
import pyautogui
import numpy as np
from mss import mss
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

### Using Mac GPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

### Load Model 
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 8)
)
model.load_state_dict(torch.load("model/resnet50_card_weights.pth", map_location=device))
model = model.to(device)
model.eval()

class_names = ['Valk', 'battle_ram', 'cage', 'dark_prince', 'healer', 'hog_rider', 'mini_pekka', 'musk']

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

### Pushes Cropped Image To Model For Classification
def predict_card(img):
    img_tensor = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        idx = probs.argmax(1).item()
        conf = probs[0, idx].item()
    return class_names[idx], conf

### Simple Tempalte Matching
def detect(needle, haystack, threshold=0.98):
    res = cv2.matchTemplate(haystack, needle, 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    p1 = max_loc
    p2 = (p1[0] + needle.shape[:2][1], p1[1] + needle.shape[:2][0])
    if max_val > threshold:
        return True, (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    else:
        return False, 0, 0

### Constants 
i = 0
hand = ["?" for i in range(8)]
card_height, card_width, scale = 40, 20, 0.5
left = {'top': 58, 'left': 455, 'width': 370, 'height': 654}
needle = cv2.imread("screenshots/spawn.png", cv2.IMREAD_COLOR)
needle = cv2.resize(needle, (0, 0), fx=scale, fy=scale)

### Screen Capture 
sct = mss()


while pyautogui.position()[1] >= 50:
    ### Grabs Game Screen
    frame = np.array(sct.grab(left))
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    spawn_detected, x, y = detect(needle, frame)
    if spawn_detected:

        ### Converting Image to Screen Coordinate 
        x = int(x + left['left'])
        y = int(y + left['top'])

        ### Capturing Card Near Detected Spawn Region
        card_box = {'top': y - card_height, 'left': x - card_width, 'width': 2 * card_width, 'height': int(1.5 * card_height)}
        troop = sct.grab(card_box)
        troop_img = Image.frombytes("RGB", troop.size, troop.rgb).convert("RGB")
        card_name, cnf = predict_card(troop_img)
        # troop_img.save("debug_latest.png") (Optional For Debuging)

        ### Updating Opponents Hand Based on Model's Prediction
        s = card_name
        if s not in hand:
            hand[i%4] = s
            i += 1
        x = hand.index(s)
        hand[7], hand[6], hand[5], hand[4], hand[x] = hand[x], hand[7], hand[6], hand[5], hand[4]
        # print(f"Spawned card: {card_name} (conf={cnf:.2f})") (Optional For Debuging)
        print(f"Current hand: {hand[:4]}")

    cv2.imshow("screen", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break