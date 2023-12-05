# import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib


ClassificationTypes = {
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
}
ClassificationTypesCount = {
    'Helmet': 1104,
    'Gauntlets': 1132,
    'Chest Armor': 1100,
    'Leg Armor': 1089,
    'Hand Cannon': 207,
    'Scout Rifle': 163,
    'Combat Bow': 60,
    'Submachine Gun': 141,
    'Auto Rifle': 161,
    'Pulse Rifle': 172,
    'Sidearm': 115,
    'Shotgun': 174,
    'Sniper Rifle': 151,
    'Fusion Rifle': 124,
    'Trace Rifle': 22,
    'Machine Gun': 79,
    'Rocket Launcher': 91,
    'Linear Fusion Rifle': 44,
    'Sword': 85,
    'Grenade Launcher': 141
}


def loadImages() :
    dir = "./Images"
    print("this is the start")
    tf.keras.preprocessing.image_dataset_from_directory(dir)

    print("This is the endpyt")
