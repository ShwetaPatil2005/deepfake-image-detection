from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_Meso4 import Meso4
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Image Data Generator
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./127.5 - 1.0,  # normalize to [-1,1]
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=(256, 256),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)

# -----------------------------
# Load model
# -----------------------------
model = Meso4()
model.load_weights("Meso4_DF.h5")
print("✅ Loaded pretrained Meso4_DF!")

# -----------------------------
# Compile and fine-tune
# -----------------------------
model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5  # you can increase epochs if needed
)

# -----------------------------
# Save fine-tuned weights
# -----------------------------
model.save_weights("Meso4_DF_finetuned.weights.h5")
print("✅ Fine-tuning complete and weights saved!")

