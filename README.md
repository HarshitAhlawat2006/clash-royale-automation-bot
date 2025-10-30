# 🤖 Clash Royale Automation Bot

An automation bot built using **computer vision + PyTorch** to detect troop cards and automate gameplay decisions in **Clash Royale**.

---

## 🚀 Features

- 🧠 Custom-trained **ResNet50** model for card detection  
- 🎮 Automates troop deployment using **mouse control + OpenCV**  
- 📊 Built custom dataset of 1000+ troop screenshots  
- ⚙️ Modular and extendable architecture  

---

## 🧩 Tech Stack

| Component | Tech |
|------------|------|
| Model | PyTorch (ResNet50) |
| Vision | OpenCV |
| Automation | PyAutoGUI |
| Training | Custom dataset pipeline |
| Language | Python 3.10 |

---

## 📂 Project Structure
Card Cycle V2/
│
├── model/                     # Trained weights + model code
├── screenshots/               # Dataset screenshots
├── scripts/                   # Training + testing scripts
├── utils/                     # Helper functions
├── main.py                    # Main entry point
└── README.md

---

## 🧠 How It Works

1. The model identifies **current card cycle** using live screenshots  
2. Cards are mapped to troop names + elixir cost  
3. The bot predicts optimal deployment strategy  
4. Mouse automation executes precise troop placement  

---

## 🧩 Demo

🎥 **Demo video:** *(add YouTube link once recorded)*  
📷 Example output: *(add GIF or sample image)*  

---

## 🛠️ Setup

```bash
git clone https://github.com/HarshitAhlawat2006/clash-royale-automation-bot.git
cd clash-royale-automation-bot
pip install -r requirements.txt
python main.py
