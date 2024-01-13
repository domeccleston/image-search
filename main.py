from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

data_path = 'flickr8k/Images/'


def save_encodings(encodings, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(encodings, file)

def load_encodings(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    return None

def plot_images(images, query, n_row=2, n_col=2):
  _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
  axs = axs.flatten()
  for img, ax in zip(images, axs):
    ax.set_title(query)
    ax.imshow(img)
  plt.show()

img_model = SentenceTransformer('clip-ViT-B-32')
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

img_names = list(glob.glob(f'{data_path}*.jpg'))

# print("Images:", len(img_names))

# Assuming img_emb.pkl is the file where you save/load your encodings
encodings_path = 'img_emb.pkl'

# Load previously saved encodings if they exist
img_emb = load_encodings(encodings_path)

# If there are no saved encodings, encode the images and save the encodings
if img_emb is None:
    img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], 
                               batch_size=128, 
                               convert_to_tensor=True, 
                               show_progress_bar=True)
    save_encodings(img_emb, encodings_path)

print(img_emb.shape, type(img_emb))

def search(query, k=4):
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, img_emb, top_k=k)[0]

    matched_images = []
    for hit in hits:
      matched_images.append(Image.open(img_names[hit['corpus_id']]))
    
    plot_images(matched_images, query)

search("Soccer")
