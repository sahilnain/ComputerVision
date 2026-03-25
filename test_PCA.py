import numpy as np
import imageio
import matplotlib.pyplot as plt
import pandas as pd
from glob import iglob
from sklearn.decomposition import PCA

def example_read_img():
    img = imageio.imread('./att_faces/s1/1.pgm')
    img = img.astype(np.uint8)
    img = img / 255
    plt.imshow(img,cmap='gray')

def read_all_faces():
    faces = pd.DataFrame([])
    for path in iglob('./att_faces/*/*.pgm'):
        img=imageio.imread(path)
        face = pd.Series(img.flatten(),name=path)
        faces = pd.concat([faces, pd.DataFrame([face])], ignore_index=True)
    
    # fig, axes = plt.subplots(10,10,figsize=(9,9),
    # subplot_kw={'xticks':[], 'yticks':[]},
    # gridspec_kw=dict(hspace=0.01, wspace=0.01))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(faces.iloc[i].values.reshape(112,92),cmap='gray')
    return faces

def main():

    faces = read_all_faces()

    print("min:", faces.min())
    print("max:", faces.max())
    # faces_pca = PCA(n_components=0.8)
    # faces_pca.fit(faces)

    # fig, axes = plt.subplots(2,10,figsize=(9,3),
    # subplot_kw={'xticks':[], 'yticks':[]},
    # gridspec_kw=dict(hspace=0.01, wspace=0.01))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(faces_pca.components_[i].reshape(112,92),cmap="gray")
    # plt.show()

if __name__ == "__main__":
    main()
