import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers



# ref (https://keras.io/examples/vision/image_classification_from_scratch/)

image_size = (96, 96)
batch_size = 64

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    "Images",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=(-0.2, 0.2)),
    ]
)


augmented_train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y))


train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)








# This is the pre-made one with little config
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add   residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)





# https://stackoverflow.com/questions/61983606/how-to-construct-a-sobel-filter-for-kernel-initialization-in-input-layer-for-ima
@keras.saving.register_keras_serializable(package="my_package", name="kernelInitializer")
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

def make_my_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)

    x = keras.layers.Conv2D(1, kernel_size=(5, 5), kernel_initializer=kernelInitializer, strides=(2, 2),
                              activation='relu')(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

# model = make_model(input_shape=image_size + (3,), num_classes=20)
model = make_my_model(input_shape=image_size + (3,), num_classes=20)

# keras.utils.plot_model(model, show_shapes=True)


epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.legacy.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)








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

img = keras.utils.load_img(
    "Images/Sniper Rifle/sniper Rifle10.jpg", target_size=image_size
)
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)

print(predictions)
print(ClassificationTypes)



for i in range(20):
    print(f"Class = {ClassificationTypes[i]}, prob = {predictions[0][i]}")



print(f"The classification would be {ClassificationTypes[predictions[0].argmax()]}")
