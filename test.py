import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from metrics import dice_loss, dice_coef, iou
from train import load_dataset
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.metrics import binary_crossentropy, Precision, Recall, accuracy
from train import *

""" Global parameters """
H = 256
W = 256

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_segmented_slices(images, output = "results"):
    # Create the output folder if it doesn't exist
    os.makedirs(output, exist_ok=True)
    for i, image in enumerate(images):
        filename = os.path.join(output, f"testing_{i+1}.jpg")
    cv2.imwrite(filename, images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Load the model """
    with custom_object_scope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = load_model (os.path.join("files", "model.h5" ))

    """ Dataset """
    dataset_path = "C:/Users/HI/PycharmProjects/pythonProject2/New data"
    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_path)

    """ Prediction and Evaluation """

    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = x.split("/")[-1]
        print(name)

        """ Reading the image """
        images = cv2.imread(x, cv2.IMREAD_COLOR) ## [H, w, 3]
        images = cv2.resize(images, (W, H))       ## [H, w, 3]
        x = images/255.0
        x = x.astype(np.float32)           ## [H, w, 3]
        x = np.expand_dims(x, axis=0)           ## [1, H, w, 3]

        """ Reading the mask """
        masks = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        masks = cv2.resize(masks, (W, H))
        masks = np.expand_dims(masks, axis=-1)
        masks = np.concatenate([masks, masks, masks], axis=-1)


        """ Prediction """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = y_pred * 255
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)


        """ Saving the prediction """
        line = np.ones((H, 10, 3)) * 255


        comb_images = np.concatenate([images, line, masks, line, y_pred], axis=1)
        save_image_path = os.path.join("results", name)
        save_segmented_slices(comb_images, save_image_path)
        #plt.figure(figsize=(20,3))
        #plt.imshow(y_pred)
        #plt.show()

        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)
        """ Flatten the array """
        masks = masks/255.0
        masks = (masks > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        f1_value = f1_score(masks, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(masks, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(masks, y_pred, labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(masks, y_pred, labels=[0, 1], average="binary", zero_division=0)
        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"F1: {score[0]:0.5f}")
    print(f"Jaccard: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")

    save_image_path = os.path.join('results', name )

