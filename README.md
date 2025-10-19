# Projet_StatApp

## Installation

### Prérequis sur SSP Cloud

Pour que le projet fonctionne correctement sur SSP Cloud, vous devez installer manuellement les dépendances suivantes :

#### 1. Installation de FFmpeg
```bash
sudo apt install ffmpeg -y
```

#### 2. Installation de PyTorch avec support CUDA
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> **Note** : Cette version de PyTorch est compatible avec CUDA 12.4.
