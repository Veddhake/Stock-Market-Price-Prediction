from tensorflow.python.keras.models import load_model

model_path = r"C:\Users\Ved Dhake\stock-market-prediction\predict.keras"

try:
    model = load_model(model_path)  # Correct way to load a .keras file
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
