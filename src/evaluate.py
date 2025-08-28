from tensorflow.keras.models import load_model
from src.data_loader import create_data_generators

DATA_DIR = 'data/tubers'
IMG_SIZE = (128,128)
BATCH_SIZE = 16

# Load model
model = load_model('best_model.h5')

# Load data (here we can use validation split as test for simplicity)
_, test_gen = create_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")
