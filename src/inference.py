import torch
import numpy as np
from utils import capture_minimap
from train import MinimapCNN
from pynput.keyboard import Controller, Key
import time
import random

keyboard = Controller()
last_pressed = 0b000

def send_keystrokes(keys):
    """
    Manage key presses and releases based on bit changes.
    
    Args:
        keys: Integer with bits representing key states
              bit 0 (0b001): left arrow
              bit 1 (0b010): right arrow
              bit 2 (0b100): 'w' key
    """
    global last_pressed
    
    # Define key mapping (bit position to key)
    key_mapping = [
        Key.left,   # bit 0
        Key.right,  # bit 1
        'w'         # bit 2
    ]
    
    # Find which bits changed (using XOR)
    changed_bits = last_pressed ^ keys
    
    for i in range(3):
        bit = 1 << i
        
        # If this bit changed
        if changed_bits & bit:
            key = key_mapping[i]
            
            # New state is 1 (key should be pressed)
            if keys & bit:
                keyboard.press(key)
            # New state is 0 (key should be released)
            else:
                keyboard.release(key)
    
    # Save current key state for next comparison
    last_pressed = keys

def preprocess_image(image):
    image = image.astype(np.float32) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return image

def predict(model, image, device):
    model.eval()
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        logits = model(image)
        predictions = torch.sigmoid(logits)
    return predictions.cpu().numpy()[0]

def main():
    model_path = "models/minimap_cnn_latest.pth"
    minimap_region = {"top": 100, "left": 2310, "width": 200, "height": 200}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Architecture hyperparameters
    kernel_sizes = [8, 5, 3]
    strides = [2, 1, 1]
    pool_sizes = [3, 3, 2]
    channels = [32, 64, 128]
    hidden_dims = [1024, 256, 128]
    
    # Initialize model with hyperparameters
    dropout1 = 0.7
    dropout2 = 0.7
    dropout3 = 0.7
    
    model = MinimapCNN(
        num_actions=3,
        dropout1=dropout1,
        dropout2=dropout2,
        dropout3=dropout3,
        kernel_sizes=kernel_sizes,
        strides=strides,
        pool_sizes=pool_sizes,
        channels=channels,
        hidden_dims=hidden_dims
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print('starting in 3 seconds')
    time.sleep(3)
    while True:
        time.sleep(random.random() * 0.1)
        #randomly skip some frames
        if random.random() < 0.05:
            continue

        # Capture current frame
        edges = capture_minimap(minimap_region)
        
        # Predict actions
        actions = predict(model, edges, device)
        print("Predicted actions:", actions)
        
        print([f"{val:.2f}" for val in actions])
        keys_to_press = 0b000
        if actions[0] > 0.4:
            keys_to_press |= 0b001
        if actions[1] > 0.4:
            keys_to_press |= 0b010
        if actions[2] > 0.9:
            keys_to_press |= 0b100
        send_keystrokes(keys_to_press)
        

if __name__ == "__main__":
    main()
