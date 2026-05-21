"""
dream-mirror / live.py

La boucle vivante. A chaque tour :
  1. on capture la frame courante de la webcam
  2. le modele predit a quoi ressemblera la frame SUIVANTE
  3. on attend la vraie frame suivante
  4. on calcule l'erreur et on fait UN pas de gradient (= live training)
  5. on affiche  [ REEL | PREDICTION | ERREUR ]

Lance avec :   python live.py
Quitte avec :  touche Q  (dans la fenetre d'affichage)

Tourne sur CPU. Garde la fenetre au premier plan pour que les touches
soient capturees.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from model import DreamMirror

# ----------------------- reglages -----------------------
SIZE = 64            # resolution de travail (carre, niveaux de gris)
LR = 1e-3            # learning rate du pas live
DISPLAY_SCALE = 4    # facteur d'agrandissement de l'affichage
WEBCAM_INDEX = 0     # change en 1, 2... si ta webcam n'est pas la 0
# --------------------------------------------------------


def to_tensor(gray_frame: np.ndarray) -> torch.Tensor:
    """image grise uint8 [H,W] -> tenseur float [1,1,SIZE,SIZE] dans [0,1]"""
    small = cv2.resize(gray_frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    arr = small.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def to_image(tensor: torch.Tensor) -> np.ndarray:
    """tenseur [1,1,SIZE,SIZE] dans [0,1] -> image grise uint8 [SIZE,SIZE]"""
    arr = tensor.detach().squeeze().clamp(0, 1).numpy()
    return (arr * 255).astype(np.uint8)


def main():
    device = torch.device("cpu")
    model = DreamMirror().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la webcam (index {WEBCAM_INDEX}). "
            "Essaie un autre WEBCAM_INDEX, ou verifie qu'aucune autre "
            "application n'utilise la camera."
        )

    print("dream-mirror en cours. Fenetre active + touche Q pour quitter.")

    # on a besoin de la frame precedente pour predire la courante
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Lecture de la premiere frame impossible.")
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_tensor = to_tensor(prev_gray)

    step = 0
    ema_loss = None  # moyenne glissante de la loss, pour l'affichage

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        target_tensor = to_tensor(gray)  # la vraie frame_{t+1}

        # --- prediction a partir de la frame precedente ---
        prediction = model(prev_tensor)

        # --- live training : un pas de gradient ---
        loss = criterion(prediction, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # moyenne glissante pour un affichage stable
        val = loss.item()
        ema_loss = val if ema_loss is None else 0.98 * ema_loss + 0.02 * val
        step += 1

        # --- affichage cote a cote : reel | prediction | erreur ---
        real_img = to_image(target_tensor)
        pred_img = to_image(prediction)
        err_img = cv2.absdiff(real_img, pred_img)
        err_img = cv2.applyColorMap(err_img, cv2.COLORMAP_INFERNO)

        real_bgr = cv2.cvtColor(real_img, cv2.COLOR_GRAY2BGR)
        pred_bgr = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)

        panel = np.hstack([real_bgr, pred_bgr, err_img])
        big = cv2.resize(
            panel,
            (panel.shape[1] * DISPLAY_SCALE, panel.shape[0] * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )

        # labels + compteur de loss
        cv2.putText(big, "REEL", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(big, "PREDICTION", (SIZE * DISPLAY_SCALE + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(big, "ERREUR", (2 * SIZE * DISPLAY_SCALE + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(big, f"step {step}  loss {ema_loss:.4f}",
                    (10, big.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        cv2.imshow("dream-mirror", big)

        # la frame courante devient la precedente pour le prochain tour
        prev_tensor = target_tensor

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Termine apres {step} pas. Loss finale (lissee) : {ema_loss:.4f}")


if __name__ == "__main__":
    main()