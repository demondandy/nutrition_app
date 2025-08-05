import tensorflow as tf
import numpy as np
import os

# Classes for food categories
classes = ["Oatmeal", "Jollof Rice", "Egusi Soup"]

# Dummy training data
x_train = np.random.rand(50, 224, 224, 3).astype(np.float32)
y_train = np.random.randint(0, len(classes), 50)

# Convert labels to one-hot
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train dummy model
model.fit(x_train, y_train, epochs=3)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/cnn_model.keras")

print("Model saved successfully!")
