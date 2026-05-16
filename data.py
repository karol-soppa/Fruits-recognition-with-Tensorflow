import os
import numpy as np
from PIL import Image


def load_dataset(images_path, labels_path):
    # 1. Pobieramy surowe listy plików
    # Używamy sorted(), żeby mieć pewność, że Python przeczyta je w identycznej kolejności systemowej
    raw_image_files = sorted(os.listdir(images_path))
    raw_label_files = sorted(os.listdir(labels_path))

    # 2. Filtrujemy tylko poprawne rozszerzenia
    image_files = [f for f in raw_image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    label_files = [f for f in raw_label_files if f.lower().endswith('.txt')]

    # ZABEZPIECZENIE: Sprawdzamy, czy fizyczna liczba plików w folderach się zgadza
    if len(image_files) != len(label_files):
        print(
            f"❌ BŁĄD KRYTYCZNY: Liczba obrazów ({len(image_files)}) nie zgadza się z liczbą etykiet ({len(label_files)})!")
        print("Upewnij się, że w folderach jest dokładnie tyle samo plików przed uruchomieniem.")
        return np.array([]), np.array([])

    images_data = []
    labels_data = []

    # 3. Przechodzimy przez oba pliki jednocześnie za pomocą zip()
    # img_name (np. "photo1.jpg") oraz label_name (np. "class1.txt")
    for img_name, label_name in zip(image_files, label_files):
        full_label_path = os.path.join(labels_path, label_name)

        with open(full_label_path, 'r') as file:
            first_line = file.readline().strip()

            # Warunek: Plik etykiety NIE jest pusty
            if first_line:
                try:
                    first_value = first_line.split()[0]
                    label_val = int(first_value)

                    # DOPIERO TUTAJ, gdy etykieta jest w 100% poprawna,
                    # wczytujemy odpowiadający jej obrazek
                    full_img_path = os.path.join(images_path, img_name)
                    img = Image.open(full_img_path).convert('RGB')

                    # Dodajemy obie rzeczy jednocześnie do naszych list
                    images_data.append(np.array(img))
                    labels_data.append(label_val)

                except ValueError:
                    print(
                        f"❌ POMINIĘTO PARĘ: Plik {label_name} nie zaczyna się od liczby. Pomijam ten plik oraz obraz {img_name}.")
            else:
                print(f"❌ POMINIĘTO PARĘ: Plik {label_name} jest pusty. Pomijam ten plik oraz obraz {img_name}.")

    return np.array(images_data), np.array(labels_data)


# --- Jak tego teraz użyć w kodzie? ---

path_train_img = '.\\Dataset\\train\\images'
path_train_labels = '.\\Dataset\\train\\labels'

# Jedno wywołanie zwraca od razu obie gotowe, zsynchronizowane tablice
images_train, labels_train = load_dataset(path_train_img, path_train_labels)

print("\n--- Podsumowanie treningowe ---")
print("Format train_images:", images_train.shape)
print("Format train_labels:", labels_train.shape)