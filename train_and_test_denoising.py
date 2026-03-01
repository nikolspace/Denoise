import os
import glob
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. Metrics and Loss Functions
# -----------------------------------------------------------------------------
def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return 1 - dice

def dice_bce_loss(y_true, y_pred):
    return 0.5 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

# -----------------------------------------------------------------------------
# 2. U-Net Architecture (4-Layer)
# -----------------------------------------------------------------------------
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes=1):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    b1 = conv_block(p3, 512)
    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d3)
    return Model(inputs, outputs, name="U-Net-Denoising")

# -----------------------------------------------------------------------------
# 3. Data Loading Utility
# -----------------------------------------------------------------------------
def load_paired_dataset(root_dir, num_images=1000, size=512):
    img_dir = os.path.join(root_dir, "noisy_input")
    mask_dir = os.path.join(root_dir, "ground_truth_template")
    
    img_paths = sorted(glob.glob(img_dir + '/*.png'))[:num_images]
    mask_paths = sorted(glob.glob(mask_dir + '/*.png'))[:num_images]
    
    X, Y = [], []
    print(f"Loading {len(img_paths)} image pairs...")
    for i_path, m_path in zip(img_paths, mask_paths):
        img = cv2.imread(i_path, 0)
        mask = cv2.imread(m_path, 0)
        X.append(cv2.resize(img, (size, size)))
        Y.append(cv2.resize(mask, (size, size)))
    
    X = np.expand_dims(np.array(X), axis=3) / 255.0
    Y = np.expand_dims(np.array(Y), axis=3) / 255.0
    return X, Y

# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_ROOT = "./denoising_dataset"
    SAVE_DIR = "./noise_removal_output"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. Load Data
    X, Y = load_paired_dataset(DATA_ROOT, num_images=5000)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 2. Build and Compile
    model = build_unet((512, 512, 1))
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_bce_loss, metrics=[iou])

    # 3. Resume Check
    WEIGHTS_PATH = os.path.join(SAVE_DIR, "checkpoint.weights.h5")
    HISTORY_PATH = os.path.join(SAVE_DIR, "history.csv")
    initial_epoch = 0
    if os.path.exists(WEIGHTS_PATH):
        print("Resuming from checkpoint...")
        model.load_weights(WEIGHTS_PATH)
        if os.path.exists(HISTORY_PATH):
            initial_epoch = len(pd.read_csv(HISTORY_PATH))

    # 4. Train
    callbacks = [
        EarlyStopping(patience=15, monitor='val_iou', mode='max', verbose=1),
        ModelCheckpoint(WEIGHTS_PATH, monitor='val_iou', mode='max', save_best_only=False, save_weights_only=True),
        CSVLogger(HISTORY_PATH, append=True)
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=4,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        shuffle=True
    )

    # 5. Evaluate and Plot
    full_history = pd.read_csv(HISTORY_PATH)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.plot(full_history['iou'], label='Train IoU'); plt.plot(full_history['val_iou'], label='Val IoU'); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(full_history['loss'], label='Train Loss'); plt.plot(full_history['val_loss'], label='Val Loss'); plt.legend()
    plt.show()

    # Visual Test
    idx = random.randint(0, len(X_test)-1)
    pred = model.predict(np.expand_dims(X_test[idx], 0))[0]
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Input (Otsu Noisy)"); plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.subplot(1, 3, 2); plt.title("Ground Truth"); plt.imshow(y_test[idx].squeeze(), cmap='gray')
    plt.subplot(1, 3, 3); plt.title("Prediction"); plt.imshow(pred.squeeze() > 0.5, cmap='gray')
    plt.show()