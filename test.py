import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

num_skipped = 0

ClassificationTypes = (
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
)
for folder_name in ClassificationTypes:
    folder_path = os.path.join("Images", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)


print("Deleted %d images" % num_skipped)