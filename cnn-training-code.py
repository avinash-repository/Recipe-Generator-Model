import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Alright, let's set up where our image data is and some model settings.
data_dir = "./images"  
img_height, img_width = 224, 224 
batch_size = 40  # How many images we process at once
epochs = 100  
num_classes = len(os.listdir(data_dir)) 

# Let's get our image data ready and add some variations to help the model learn better.
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    validation_split=0.2, 
    rotation_range=40,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    shear_range=0.2,  
    zoom_range=0.2,  
    horizontal_flip=True,  
    vertical_flip=True,  
    brightness_range=[0.8, 1.2],  
    channel_shift_range=0.2,  
    fill_mode='nearest'  
)

# Load the training images from our directory.
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # We're classifying into multiple categories
    subset='training'
)

# And load the validation images too.
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

#prevent pretrained model from training
for layer in base_model.layers:
    layer.trainable = False

# Now, we'll add some custom layers on top for our specific task.
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Turn the feature maps into a single vector
x = Dense(512, activation='relu')(x)  # Add a fully connected layer
x = Dropout(0.5)(x)  # Add dropout to prevent overfitting
predictions = Dense(num_classes, activation='softmax')(x)  # Output layer for our ingredient categories

# Put it all together into our final model.
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with the Adam optimizer.
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']
)

# Let's see what our model looks like.
model.summary()

# We'll set up some callbacks to help with training.
checkpoint = ModelCheckpoint(
    'ingredient_classifier_checkpoint.h5',  # Save the best model
    monitor='val_accuracy',  # Based on validation accuracy
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',  
    patience=5,  # 5 epochs 
    restore_best_weights=True,  
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Reduce learning rate if validation loss plateaus
    factor=0.2,  # Reduce by 20%
    patience=5,  # Wait 5 epochs
    min_lr=1e-6,  # Don't go below this learning rate
    verbose=1
)

# Let's train the model!
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Let's see how we did.
    print("\nFinal Training Results:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

    # Now, let's try fine-tuning by unfreezing some layers.
    print("\nFine-tuning the model...")
    for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
        layer.trainable = True

    # Recompile with a lower learning rate.
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train again with fine-tuning.
    history_fine_tune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=20,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Save our trained model.
    model.save('ingredient_classifier.h5')
    print("Model saved successfully as 'ingredient_classifier.h5'")

    # Save the class indices for later use.
    np.save('class_indices.npy', train_generator.class_indices)
    print("Class indices saved as 'class_indices.npy'")

except Exception as e:
    print(f"An error occurred during training: {e}")