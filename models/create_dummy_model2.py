from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Define class names
classes = ['Akamu', 'Rice', 'Beans', 'Plantain', 'Yam']
num_classes = len(classes)

# Build dummy CNN model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Dummy training with random data just to generate weights
X_dummy = np.random.rand(20, 150, 150, 3)
y_dummy = np.zeros((20, num_classes))
for i in range(20):
    y_dummy[i, np.random.randint(0, num_classes)] = 1

model.fit(X_dummy, y_dummy, epochs=1, verbose=1)

# Save the dummy model
model.save('models/food_classifier_model.h5')

print("Dummy model saved successfully!")
