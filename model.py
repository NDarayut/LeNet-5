from tensorflow.keras import layers, models, Input

def create_LeNet5(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(6, (5, 5), activation='tanh', padding='valid', name="C1")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="S2")(x)
    x = layers.Conv2D(16, (5, 5), activation='tanh', padding='valid', name="C3")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, name="S4")(x)
    x = layers.Conv2D(120, (4, 4), activation='tanh', padding='valid', name="C5")(x)
    x = layers.Flatten(name="Flatten")(x)
    x = layers.Dense(84, activation='tanh', name="F6")(x)
    outputs = layers.Dense(10, activation='softmax', name="Output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="modifiedLeNet5")
    return model