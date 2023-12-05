import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import os


def kernelInitializer(shape, dtype=None):
    sobel_x = tf.constant(
        [
            [-5, -4, 0, 4, 5],
            [-8, -10, 0, 10, 8],
            [-10, -20, 0, 20, 10],
            [-8, -10, 0, 10, 8],
            [-5, -4, 0, 4, 5]
        ], dtype=dtype )
    #create the missing dims.
    sobel_x = tf.reshape(sobel_x, (5, 5, 1, 1))

    print(tf.shape(sobel_x))
    #tile the last 2 axis to get the expected dims.
    sobel_x = tf.tile(sobel_x, (1, 1, shape[-2],shape[-1]))

    print(sobel_x)
    return sobel_x
# loaded_model = tf.keras.saving.load_model("save_good.keras")
loaded_model = tf.keras.saving.load_model("save_good_me.keras", custom_objects={"kernelInitializer": kernelInitializer},)
# loaded_model = tf.keras.saving.load_model("save_at_50.keras", custom_objects={"kernelInitializer": kernelInitializer},)



ClassificationTypes = [
    'Helmet',
    'Gauntlets',
    'Chest Armor',
    'Leg Armor',
    'Hand Cannon',
    'Scout Rifle',
    'Combat Bow',
    'Submachine Gun',
    'Auto Rifle',
    'Pulse Rifle',
    'Sidearm',
    'Shotgun',
    'Sniper Rifle',
    'Fusion Rifle',
    'Trace Rifle',
    'Machine Gun',
    'Rocket Launcher',
    'Linear Fusion Rifle',
    'Sword',
    'Grenade Launcher'
]

ClassificationTypes.sort()



classCorrectnessLst = []

totalCorrect = 0.0
total = 0.0

for index in range(20):
    numCorrect = 0
    folder_path = os.path.join("Images", ClassificationTypes[index])
    numberOfImages = len(os.listdir(folder_path))
    for itemNumber in range (numberOfImages):
        total += 1
        img = keras.utils.load_img(
            f"Images/{ClassificationTypes[index]}/{ClassificationTypes[index]}{itemNumber}.jpg", target_size=(96, 96)
        )
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = loaded_model.predict(img_array, verbose=0)
        predictionClass = predictions[0].argmax()

        if (predictionClass == index):
            numCorrect = numCorrect + 1
            totalCorrect += 1

    correctness = float(numCorrect)/float(numberOfImages)
    classCorrectnessLst.append(correctness)
    print(f"{index+1}/20")



print(classCorrectnessLst)
for index in range(20):
    print(f"{ClassificationTypes[index]}: {classCorrectnessLst[index]}%")

print(sum(classCorrectnessLst)/(len(classCorrectnessLst)))
print(f"{totalCorrect / total}%")
