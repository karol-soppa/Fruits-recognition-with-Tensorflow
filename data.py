import os
import numpy as np
from PIL import Image

def img_dataset_from_directory(path):
    data = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path).convert('RGB')
            data.append(np.array(img))
    return np.array(data)

def label_dataset_from_directory(path):
    data = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.txt'):
            label_path = os.path.join(path, filename)
            with open(label_path, 'r') as file:
                first_line = file.readline().strip()
                if first_line:
                    first_value = first_line.split()[0]
                    # Rzutowanie na int. Jeśli potrzebujesz ułamków, zmień na float(first_value)
                    data.append(int(first_value))
    return np.array(data)


