import sqlite3
import json
import asyncio
import urllib.request
# Connect to the database

conn = sqlite3.connect('Manifest.db')

cur = conn.cursor()
cur.execute('SELECT json FROM DestinyInventoryItemDefinition')

jsoninfo = cur.fetchall()

BASE_URL = "http://www.bungie.net/"
ClassificationTypes = {
    # 'Helmet',
    # 'Gauntlets',
    # 'Chest Armor',
    # 'Leg Armor',
    # 'Hand Cannon',
    # 'Scout Rifle',
    # 'Combat Bow',
    # 'Submachine Gun',
    # 'Auto Rifle',
    # 'Pulse Rifle',
    # 'Sidearm',
    # 'Shotgun',
    'Sniper Rifle'
    # 'Fusion Rifle'
    # 'Trace Rifle',
    # 'Machine Gun',
    # 'Rocket Launcher',
    # 'Linear Fusion Rifle',
    # 'Sword',
    # 'Grenade Launcher'
}
ClassificationTypesCount = {
    'Helmet': 0,
    'Gauntlets': 0,
    'Chest Armor': 0,
    'Leg Armor': 0,
    'Hand Cannon': 0,
    'Scout Rifle': 0,
    'Combat Bow': 0,
    'Submachine Gun': 0,
    'Auto Rifle': 0,
    'Pulse Rifle': 0,
    'Sidearm': 0,
    'Shotgun': 0,
    'Sniper Rifle': 0,
    'Fusion Rifle': 0,
    'Trace Rifle': 0,
    'Machine Gun': 0,
    'Rocket Launcher': 0,
    'Linear Fusion Rifle': 0,
    'Sword': 0,
    'Grenade Launcher': 0
}


async def loadImage(item):
    itemJSON = json.loads(item[0])
    try:
        itemType = itemJSON['itemTypeDisplayName']
        if itemType in ClassificationTypes:
            print(itemJSON['displayProperties']['name'])
            iconPath = itemJSON['displayProperties']['icon']
            urllib.request.urlretrieve(BASE_URL + iconPath,
                                       f"Images/{itemType}/{itemType}{ClassificationTypesCount[itemType]}.jpg")
            ClassificationTypesCount[itemType] += 1
    except:
         pass

async def main():
    for item in jsoninfo:
        asyncio.create_task(loadImage(item))


asyncio.run(main())
    # itemJSON = json.loads(item[0])
    # try:
    #     itemType = itemJSON['itemTypeDisplayName']
    #     if itemType in ClassificationTypes:
    #         print(itemJSON['displayProperties']['name'])
    #         iconPath = itemJSON['displayProperties']['icon']
    #         urllib.request.urlretrieve(BASE_URL + iconPath,
    #           f"Images/{itemType}/{itemType}{ClassificationTypesCount[itemType]}.jpg")
    #         ClassificationTypesCount[itemType]+= 1
    # except:
    #     pass




# Important Types are
#   Head (Name: 'Helmet')
#   Arms (Name: 'Gauntlets')
#   Chest (Name: 'Chest Armor')
#   Boots (Name: 'Leg Armor')

#   Hand Cannon (Name: 'Hand Cannon')
#   Scout Rifle (Name: 'Scout Rifle')
#   Bow (Name: 'Combat Bow')
#   SMG (Name: 'Submachine Gun')
#   AR (Name: 'Auto Rifle')
#   Pulse (Name: 'Pulse Rifle')
#   Sidearms (Name: 'Sidearm')

#   Shot Gun (Name: 'Shotgun')
#   Sniper (Name: 'Sniper Rifle')
#   Fusion Rifle (Name: 'Fusion Rifle')
#   Trace (Name: 'Trace Rifle')

#   Machine Gun (Name: 'Machine Gun')
#   Rocket (Name: 'Rocket Launcher')
#   Linear (Name: 'Linear Fusion Rifle')
#   Sword (Name: 'Sword')
#   GL (Name: 'Grenade Launcher')








# Close the connection
conn.close()