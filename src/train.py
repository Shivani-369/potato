from src.data_loader import create_data_generators
from src.model import build_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint

DATA_DIR = 'data/tubers'
IMG_SIZE = (128,128)
BATCH_SIZE = 16
EPOCHS = 10

train_gen, val_gen = create_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
model = build_cnn_model(input_shape=(128,128,3), num_classes=len(train_gen.class_indices))

checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint])
