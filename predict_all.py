from model_Meso4 import Meso4
from utils_preprocess import preprocess_face
import numpy as np

# -----------------------------
# Load model
# -----------------------------
model = Meso4()
model.load_weights("Meso4_DF_finetuned.weights.h5")
print("✅ Loaded fine-tuned Meso4_DF!")

# -----------------------------
# Prediction function
# -----------------------------
def predict(image_path, threshold_low=0.2, threshold_high=0.8):
    face = preprocess_face(image_path)
    face = np.expand_dims(face, axis=0)

    score = model.predict(face)[0][0]
    label = "Real Face" if score >= 0.4 else "DeepFake Face"

    # Confidence warning
    if score > threshold_high and label == "Real Face":
        warning = "⚠️ High confidence but may not generalize"
    elif score < threshold_low and label == "DeepFake Face":
        warning = "⚠️ High confidence but may not generalize"
    else:
        warning = ""

    # Print result
    print("\n--- Prediction ---")
    print(f"Image: {image_path}")
    print(f"Score={score:.4f} → {label} {warning}")

# -----------------------------
# Example usage
# -----------------------------
test_image_path = r"D:/mesonet-project/dataset/fake/00S2D5CDKQ.jpg"
predict(test_image_path)
