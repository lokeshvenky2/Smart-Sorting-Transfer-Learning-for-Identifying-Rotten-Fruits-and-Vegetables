import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import CosineDecay

# ---- Paths ----
train_data_path = 'A:/Smart_Sorting/dataset/fruits-360_100x100/fruits-360/Training'
model_output_path = 'models/fruit_classifier_effnet_light.keras'
class_indices_path = 'models/class_indices.json'

# ---- Image Settings (Reduced) ----
IMG_HEIGHT = 100
IMG_WIDTH = 100
BATCH_SIZE = 16  # Reduced from 32
SEED = 123
EPOCHS = 10      # Reduced from 20

print("📂 Loading dataset...")
# ---- Dataset ----
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    validation_split=0.1,
    subset='training',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_data_path,
    validation_split=0.1,
    subset='validation',
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)
print("✅ Dataset loaded.")

# ---- Save class indices ----
class_names = train_ds.class_names
class_indices = {name: idx for idx, name in enumerate(class_names)}
os.makedirs(os.path.dirname(class_indices_path), exist_ok=True)
with open(class_indices_path, 'w') as f:
    json.dump(class_indices, f, indent=4)
print("✅ Class indices saved.")

# ---- Data Pipeline (Optimized) ----
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(512).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
print("⚙️  Data pipeline ready.")

# ---- Data Augmentation ----
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# ---- Build Model ----
print("🔨 Building model...")
base_model = EfficientNetB0(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
print("✅ Model built.")

# ---- Compile ----
lr_schedule = CosineDecay(initial_learning_rate=1e-3, decay_steps=EPOCHS * len(train_ds), alpha=1e-5)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
print("⚙️  Model compiled.")

# ---- Callbacks ----
callbacks = [
    ModelCheckpoint(model_output_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
]

# ---- Train Top Layers ----
print("🚀 Phase 1: Training top layers...")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks, verbose=1)

# ---- Fine-tune EfficientNet ----
print("🔧 Unfreezing and fine-tuning EfficientNet...")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=['accuracy']
)

print("🚀 Phase 2: Fine-tuning model...")
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks, verbose=1)

# ---- Save Model ----
print("💾 Saving model...")
model.save(model_output_path)
print("✅ Model saved to:", model_output_path)
print("✅ Class indices saved to:", class_indices_path)
