import os
import shutil


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



smallestDataSet = 10000
smallestClass = -1

for index in range(20):
    imageFolderPath = os.path.join("./Images", ClassificationTypes[index])
    trainFolderPath = os.path.join("./trainData", ClassificationTypes[index])
    numberOfImages = len(os.listdir(imageFolderPath))
    for itemNumber in range(23):
        srcFileName = f"{ClassificationTypes[index]}{itemNumber}.jpg"
        dstFileName = f"{ClassificationTypes[index]}-{itemNumber}.jpg"
        src = os.path.join(imageFolderPath, srcFileName)
        dst = os.path.join(trainFolderPath, dstFileName)
        shutil.copyfile(src, dst)

