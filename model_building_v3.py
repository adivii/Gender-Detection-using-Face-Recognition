import os

import pickle
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

X = []
y = []

for class_ in os.listdir("/content/drive/MyDrive/Kuliah/Face Detection/cropped_dataset"):
  filenames = os.listdir(f"/content/drive/MyDrive/Kuliah/Face Detection/cropped_dataset/{class_}")
  for i in tqdm(range(len(filenames[:7014]))):
    filename = filenames[i]
    if not filename.endswith(".jpg"):
        continue # not a face picture
    with open(f"/content/drive/MyDrive/Kuliah/Face Detection/cropped_dataset/{class_}/{filename}", "rb") as image:
        # If we extracted files from zip, we can use cv2.imread(filename) instead
        img = cv2.imdecode(np.asarray(bytearray(image.read())), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200), interpolation= cv2.INTER_LINEAR)
        X.append(img)
        y.append(class_)

fig, axes = plt.subplots(4,4,sharex=True,sharey=True,figsize=(8,10))
faceimages = X[-16:] # take last 16 images
for i in range(16):
    axes[i%4][i//4].imshow(faceimages[i], cmap="gray")
plt.show()

faceshape = X[0].shape
print("Face image shape:", faceshape)

facematrix = []
facelabel = []
for img, class_ in zip(X, y):
    facematrix.append(img.flatten())
    facelabel.append(class_)

# Create a NxM matrix with N images and M pixels per image
facematrix = np.array(facematrix)

pca = PCA(n_components=50).fit(facematrix)

n_components = 50
eigenfaces = pca.components_[:n_components]

# with open("/content/drive/MyDrive/Kuliah/Face Detection/data.bin", "wb") as f:
#   pickle.dump(facematrix, f)

# with open("/content/drive/MyDrive/Kuliah/Face Detection/pca_fitted.bin", "rb") as f:
#   pca = pickle.load(f)

# with open("/content/drive/MyDrive/Kuliah/Face Detection/target.bin", "wb") as f:
#   pickle.dump(y, f)

face_pca = pca.transform(facematrix)

face_pca.shape

f = open("/content/drive/MyDrive/Kuliah/Face Detection/pca_fitted.bin", "wb")
pickle.dump(pca, f)

# df = pd.DataFrame(facematrix)
# df['class'] = pd.Series(facelabel)
# # df.head(5)
# df.to_csv('/content/drive/MyDrive/Kuliah/Face Detection/image_data.csv', index=False)

# with open('/content/drive/MyDrive/Kuliah/Face Detection/model_svm.bin', 'rb') as f:
#   model = pickle.load(f)

# test_image = cv2.imread('/content/drive/MyDrive/Kuliah/Face Detection/cropped_dataset/woman/woman_19.jpg')
# test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# face = cv2.resize(test_image, (200, 200), interpolation= cv2.INTER_LINEAR)
# face_pca = pca.transform(face.flatten().reshape(1, -1))
# output = model.predict(face_pca)

# print(output)