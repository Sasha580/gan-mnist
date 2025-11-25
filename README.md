# Convolutional GAN (Fashion‑MNIST) — TensorFlow 2.x

A compact DCGAN‑style implementation for **image synthesis** on Fashion‑MNIST.  
This repo/notebook trains a convolutional generator and discriminator end‑to‑end and logs fixed‑seed sample grids. A short **training animation** is included below.

![Training animation](gan_anim.gif)

---

## Contents
- `gan-mnist.ipynb` — main training notebook (TensorFlow/Keras)
- `requirements.txt` — minimal dependencies
- `gan_anim.gif` — generated samples over training (preview above)

---

## Quickstart

### 1) Create an environment & install deps
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the notebook
```bash
jupyter notebook gan-mnist.ipynb
```

> The notebook downloads **Fashion‑MNIST**, scales to [-1, 1], and trains on 32×32 inputs (upsampled from 28×28).

---

## Model overview

- **Generator**: transposed‑conv blocks with BatchNorm + (Leaky)ReLU, final **Tanh**.
- **Discriminator**: conv blocks with BatchNorm + ReLU + Dropout, final sigmoid score.
- **Optimizers**: Adam (`lr=2e-4`, `beta1=0.5`).
- **Loss**: binary cross‑entropy with **label smoothing/noise** for stability.
- **Logging**: fixed‑noise seed for periodic sample grids; checkpointing.

> Architecture follows the classic **DCGAN** recipe on small images.

---

## Results

- Progressive improvement in sample quality (see animation).  
- Stable training with Adam β1=0.5 and light label smoothing/noise.

If you run for more epochs or tweak capacity, you can usually get sharper textures and better class shapes.

---

## Tips & Tweaks

- **Resolution**: keep 32×32 for speed; increase depth/channels for quality.
- **Regularization**: try spectral norm or gradient penalty (WGAN‑GP) for tougher datasets.
- **Conditioning**: extend to **cGAN** by injecting class embeddings in G and D.

---

## Citation (background)
- Radford, Metz, & Chintala. *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN).* ICLR 2016.

---

## License
MIT — feel free to use/adapt with attribution.
