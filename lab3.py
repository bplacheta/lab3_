import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def crop_and_save_textures(input_dir, output_dir, size=(128, 128)):
    for category in os.listdir(input_dir):
        category_dir = os.path.join(input_dir, category)
        if os.path.isdir(category_dir):
            output_category_dir = os.path.join(output_dir, category)
            os.makedirs(output_category_dir, exist_ok=True)
            for image_name in os.listdir(category_dir):
                if image_name.lower().endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(category_dir, image_name)
                    image = cv2.imread(image_path)
                    height, width, _ = image.shape
                    for y in range(0, height, size[1]):
                        for x in range(0, width, size[0]):
                            cropped = image[y:y+size[1], x:x+size[0]]
                            if cropped.shape[:2] == size:
                                output_path = os.path.join(output_category_dir, f"{image_name}_{y}_{x}.jpg")
                                cv2.imwrite(output_path, cropped)

def calculate_texture_features(image, distances, angles):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bit_depth = 6
    max_gray_value = 2 ** bit_depth - 1
    normalized_image = np.uint8(gray_image / 256 * max_gray_value)
    features = []
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(normalized_image, distances=[distance], angles=[angle], levels=max_gray_value+1, symmetric=True, normed=True)
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            asm = graycoprops(glcm, 'ASM')[0, 0]
            features.append([dissimilarity, correlation, contrast, energy, homogeneity, asm])
    return features

def save_features_to_csv(features, categories, output_file):
    output_file = "C:\\Users\\blaze\\PycharmProjects\\pythonProject1\\jpg\\texture_features\\texture_features.csv"
    df = pd.DataFrame(features, columns=['Dissimilarność', 'Korelacja', 'Kontrast', 'Energia', 'Homogeniczność', 'ASM'])
    df['Kategoria'] = categories
    df.to_csv(output_file, index=False)

def classify_features(features, labels):
    if len(features) == 0:
        print("Brak cech do klasyfikacji.")
        return
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    print("Liczba unikalnych etykiet:", len(np.unique(labels)))
    print("Unikalne etykiety:", np.unique(labels))
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dokladnosc = accuracy_score(y_test, y_pred)
    print("Dokładność klasyfikacji:", dokladnosc)

def main():
    input_dir = "C:\\Users\\blaze\\PycharmProjects\\pythonProject1\\jpg\\input_images"
    output_dir = "C:\\Users\\blaze\\PycharmProjects\\pythonProject1\\jpg\\conv"
    crop_and_save_textures(input_dir, output_dir)

    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    features = []
    categories = []
    for category in os.listdir(output_dir):
        category_dir = os.path.join(output_dir, category)
        if os.path.isdir(category_dir):
            for image_name in os.listdir(category_dir):
                image_path = os.path.join(category_dir, image_name)
                category_name = os.path.basename(category_dir)
                image = cv2.imread(image_path)
                if image is not None:
                    image_features = calculate_texture_features(image, distances, angles)
                    features.extend(image_features)
                    categories.extend([category_name] * len(image_features))
                else:
                    print("Błąd podczas wczytywania obrazu:", image_path)

    save_features_to_csv(features, categories, 'texture_features.csv')

    if features:
        classify_features(features, categories)
    else:
        print("Nie wykryto cech. Sprawdź swoje obrazy wejściowe i obliczenia cech tekstury.")

if __name__ == "__main__":
    main()
