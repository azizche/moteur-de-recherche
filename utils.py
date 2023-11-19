import cv2
import numpy as np

import json
def calculate_histogram(image_path, N):

    image =cv2.imread(image_path)
    # Diviserchaque axe de l'espace de couleurs en N intervalles
    hist_bins = [N, N, N]

    try:
        histogram = cv2.calcHist([image], [0, 1, 2], None, hist_bins,  [0, 256, 0, 256, 0, 256])
    except cv2.error as e:
        print(f"Error calculating histogram: {e}")
        return None

    # Normaliser l'histogramme
    histogram = cv2.normalize(histogram, histogram).flatten().tolist()

    return histogram

def euclidean_distance(hist1,hist2):
  distance = np.sqrt(np.sum(np.square(np.array(hist1) - np.array(hist2))))
  return distance
def calcul_similarité_histogramme(img_requete_path,json_file):
  histogram_requete = calculate_histogram(img_requete_path,8)
  with open(json_file, 'r') as f:
        data = json.load(f)
  distances = []
  for entry in data:
        image_name = entry["image_path"]
        histogram = entry["histogram"]
        # Calculer la distance eucliedienne entre les histogrammes
        distance = euclidean_distance(histogram_requete, histogram)
        distances.append((distance,image_name))
  distances.sort()
  top3 =distances[:3]
  return top3

def texture_descriptor(image_path):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculer la Transformée de Fourier 2D
    fft_image = np.fft.fft2(image)

    # Obtenir le spectre en amplitude
    amplitude_spectrum = np.abs(fft_image)

    # Découper la moitié supérieure en 6x3 blocs
    blocks = np.array_split(amplitude_spectrum[:amplitude_spectrum.shape[0]//2, :], 6, axis=0)
    blocks = [np.array_split(block, 3, axis=1) for block in blocks]

    # Calculer le logarithme de l'énergie moyenne sur chaque bloc
    descriptors = [np.log(np.mean(np.square(block))) for row in blocks for block in row]

    return descriptors
def manhattan_distance(descriptor1, descriptor2):
    # Calcul de la distance de Manhattan entre deux descripteurs
    return np.sum(np.abs(np.array(descriptor1) - np.array(descriptor2)))
def calcul_similarité_texture(img_requete_path,json_file):
  descriptors=texture_descriptor(img_requete_path)
  with open(json_file, 'r') as f:
        data = json.load(f)
  distances = []
  for entry in data:
        image_name = entry["filename"]
        descriptor = entry["descriptors"]
        # Calculer la distance de Manhattan entre les descripteurs
        distance = manhattan_distance(descriptors, descriptor)

        # Ajouter la distance et le nom de l'image à la liste
        distances.append((distance, image_name))

  distances.sort()

    # Récupérer les trois images les plus similaires
  top3_similar_images = distances[:3]

  return top3_similar_images