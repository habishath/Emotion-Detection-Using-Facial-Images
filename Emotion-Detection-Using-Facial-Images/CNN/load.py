import os
import cv2
from PIL import Image
import numpy as np

train_data=[]
train_labels=[]

test_data=[]
test_labels=[]

# LABELS
# Angry = 0
# Disgust = 1
# Fear = 2
# Happy = 3
# Neutral = 4
# Sad = 5
# Surprise = 6

# angry 0

train_angry = os.listdir(os.getcwd() +"/CNN/data/train/angry/")
for x in train_angry:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/angry/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(0)

test_angry = os.listdir(os.getcwd() + "/CNN/data/test/angry/")
for x in test_angry:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/angry/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(0)

# disgust 1

train_disgust = os.listdir(os.getcwd() +"/CNN/data/train/disgust/")
for x in train_disgust:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/disgust/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(1)

test_disgust = os.listdir(os.getcwd() + "/CNN/data/test/disgust/")
for x in test_disgust:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/disgust/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(1)

# fear
train_fear = os.listdir(os.getcwd() +"/CNN/data/train/fear/")
for x in train_fear:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/fear/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(2)

test_fear = os.listdir(os.getcwd() + "/CNN/data/test/fear/")
for x in test_fear:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/fear/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(2)

# happy 3
train_happy = os.listdir(os.getcwd() +"/CNN/data/train/happy/")
for x in train_happy:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/happy/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(3)

test_happy = os.listdir(os.getcwd() + "/CNN/data/test/happy/")
for x in test_happy:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/happy/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(3)

# neutral 4
train_neutral = os.listdir(os.getcwd() +"/CNN/data/train/neutral/")
for x in train_neutral:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/neutral/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(4)

test_neutral = os.listdir(os.getcwd() + "/CNN/data/test/neutral/")
for x in test_neutral:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/neutral/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(4)

# sad 5
train_sad = os.listdir(os.getcwd() +"/CNN/data/train/sad/")
for x in train_sad:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/sad/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(5)

test_sad = os.listdir(os.getcwd() + "/CNN/data/test/sad/")
for x in test_sad:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/sad/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(5)

# surprise 6
train_surprise = os.listdir(os.getcwd() +"/CNN/data/train/surprise/")
for x in train_surprise:
    imag = cv2.imread(os.getcwd() + "/CNN/data/train/surprise/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    train_data.append(np.array(resized_image))
    train_labels.append(6)

test_surprise = os.listdir(os.getcwd() + "/CNN/data/test/surprise/")
for x in test_surprise:
    imag = cv2.imread(os.getcwd() + "/CNN/data/test/surprise/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((48, 48))
    test_data.append(np.array(resized_image))
    test_labels.append(6)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

np.save("train_data", train_data)
np.save("train_labels", train_labels)
np.save("test_data", test_data)
np.save("test_labels", test_labels)