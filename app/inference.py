import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ===============================
# CONFIGURACIÓN
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model/resnet50_hf_final.pt"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ===============================
# TRANSFORMACIONES (IGUAL A VALIDACIÓN)
# ===============================
infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# CARGAR MODELO
# ===============================
def load_model():
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASS_NAMES))
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model

# ===============================
# FUNCIÓN DE PREDICCIÓN
# ===============================
def predict_image(image_path, show_image=True):
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    x = infer_transforms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    prediction = CLASS_NAMES[pred_idx]

    # -------- PRINT BONITO --------
    print("\n🧠 Predicción:", prediction)
    print("\nProbabilidades:")
    for cls, p in zip(CLASS_NAMES, probs):
        print(f"{cls:10s}: {p.item()*100:.2f}%")

    if show_image:
        plt.imshow(img)
        plt.title(prediction)
        plt.axis("off")
        plt.show()

    return prediction, probs.cpu().numpy()
