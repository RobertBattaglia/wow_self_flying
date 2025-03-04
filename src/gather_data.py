import numpy as np
import cv2
from pynput import keyboard
import time
import os
from utils import capture_minimap 

# Track keys that are currently pressed
pressed_keys = set()

def on_press(key):
    # Get key name
    if hasattr(key, 'char'):
        key_name = key.char
    else:
        key_name = str(key).replace('Key.', '')
    
    # Add to pressed keys set
    pressed_keys.add(key_name)

def on_release(key):
    # Get key name
    if hasattr(key, 'char'):
        key_name = key.char
    else:
        key_name = str(key).replace('Key.', '')
    
    # Remove from pressed keys set
    if key_name in pressed_keys:
        pressed_keys.remove(key_name)

def keys_to_multi_hot(pressed_keys):
    KEYS = ["left", "right", "w"]
    one_hot = np.zeros(len(KEYS))
    for i, key in enumerate(KEYS):
        if key in pressed_keys:
            one_hot[i] = 1

    return one_hot

def main():
    # Initialize keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # Create data directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/batches", exist_ok=True)
    
    minimap_region = {"top": 100, "left": 2310, "width": 200, "height": 200}
    frame_counter = int(time.time())

    frames_buffer = []
    keys_buffer = []

    print('starting in 3 seconds')
    time.sleep(3)

    while True:
        if "x" in pressed_keys:
            break

        # Capture current frame
        edges = capture_minimap(minimap_region)
        
        # Save both the processed edges and the original masked image
        multi_hot_encoded_keys = keys_to_multi_hot(pressed_keys)
        print(multi_hot_encoded_keys)
        
        frames_buffer.append(edges)
        keys_buffer.append(multi_hot_encoded_keys)

        # Save every 100 frames:
        if len(frames_buffer) >= 100:
            print(f"Saving batch of {len(frames_buffer)} frames")
            batch_filename = f"data/batches/batch_{frame_counter}.npz"
            np.savez(batch_filename, 
                     input_data=np.array(frames_buffer),
                     output_data=np.array(keys_buffer))

            # Reset buffers
            frames_buffer = []
            keys_buffer = []
        
        frame_counter += 1
        time.sleep(.1)

    cv2.destroyAllWindows()
    listener.stop()


if __name__ == "__main__":
    main()