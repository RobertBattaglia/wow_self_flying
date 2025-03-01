import torch
import numpy as np
from utils import capture_minimap
from train import MinimapCNN
from pynput.keyboard import Controller, Key
import time
import random

keyboard = Controller()
pressed_keys = set()

def send_keystrokes(keys):
    for k in pressed_keys:
        keyboard.release(k)
    pressed_keys.clear()
    for k in keys:
        pressed_keys.add(k)
        keyboard.press(k)

def preprocess_image(image):
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return image

def predict(model, image, device):
    model.eval()
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(image)
        print(output)
    return output.cpu().numpy()[0]

def main():
    model_path = "models/minimap_cnn_latest.pth"
    minimap_region = {"top": 100, "left": 2310, "width": 200, "height": 200}
    
    # Load model
    model = MinimapCNN()
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print('starting in 3 seconds')
    time.sleep(3)
    while True:
        time.sleep(0.1)
        #randomly skip some frames
        if random.random() < 0.05:
            continue

        # Capture current frame
        edges = capture_minimap(minimap_region)
        
        # Predict actions
        actions = predict(model, edges, device)
        print("Predicted actions:", actions)
        
        print([f"{val:.2f}" for val in actions])
        keys_to_press = []
        if actions[0] > 0.5:
            keys_to_press.append(Key.left)
        if actions[1] > 0.5:
            keys_to_press.append(Key.right)
        if actions[2] > 0.5:
            keys_to_press.append(Key.space)
        if actions[3] > 0.5:
            keys_to_press.append('w')
        send_keystrokes(keys_to_press)
        

if __name__ == "__main__":
    main()
