# Projet_StatApp - API de Transcription Audio

API de transcription audio utilisant Whisper d'OpenAI, déployée sur SSP Cloud avec Kubernetes.

## Table des matières

- [Prérequis](#prérequis)
- [Installation locale](#installation-locale)
- [Déploiement sur SSP Cloud](#déploiement-sur-ssp-cloud)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Commandes utiles](#commandes-utiles)

## Prérequis

### Logiciels requis

1. **Docker Desktop**
   - Télécharger : https://docs.docker.com/desktop/setup/install/windows-install/
   - Vérifier l'installation : `docker --version`

2. **kubectl** (Kubernetes CLI)
   - Installation : https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/
   - Vérifier l'installation : `kubectl version --client`

3. **Helm** (gestionnaire de packages Kubernetes)
   ```bash
   # Installer Chocolatey d'abord (si pas déjà installé)
   # Suivre: https://chocolatey.org/install

   # Puis installer Helm
   choco install kubernetes-helm
   ```
   - Vérifier l'installation : `helm version`

4. **Compte Docker Hub**
   - Créer un compte sur https://hub.docker.com/

5. **Accès SSP Cloud**
   - Compte sur https://datalab.sspcloud.fr/

## Installation locale

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd Projet_StatApp
```

### 2. Créer un environnement virtuel

```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Installer les dépendances

```bash
pip install -r requirements-dev.txt
```

### 4. Tester localement

```bash
# Lancer l'API
fastapi dev app/main.py

# Dans un autre terminal, tester avec le script
python interact_with_app/interact.py
```

L'API sera accessible à l'adresse : `http://localhost:8000`

## Déploiement sur SSP Cloud

### Étape 1 : Configuration de kubectl

1. Aller sur https://datalab.sspcloud.fr/account/k8sCodeSnippets
2. Copier le script shell fourni
3. **Important** : Le script est mal indenté, remplacer tous les `\` par des retours à la ligne
4. Coller et exécuter le script dans PowerShell

```bash
# Vérifier la connexion au cluster Kubernetes
kubectl get nodes
```

### Étape 2 : Personnaliser le chart Helm

Éditer `./helm-chart/values.yaml` et remplacer les valeurs suivantes :

```yaml
image:
  repository: votre-pseudo-dockerhub/audio-to-text  # Remplacer par votre pseudo Docker Hub
  pullPolicy: Always
  tag: "latest"

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: audio-to-text-user-votre-username.lab.sspcloud.fr  # Remplacer par votre username SSP Cloud
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: audio-to-text-tls
      hosts:
        - audio-to-text-user-votre-username.lab.sspcloud.fr  # Même username que ci-dessus
```

### Étape 3 : Builder et pusher l'image Docker

```bash
# Se connecter à Docker Hub (Docker Desktop doit être lancé)
docker login

# Builder l'image (peut prendre 5-10 minutes)
docker build -t votre-pseudo-dockerhub/audio-to-text:latest .

# Pusher l'image sur Docker Hub (peut prendre 5-10 minutes selon la connexion)
docker push votre-pseudo-dockerhub/audio-to-text:latest
```

**Note importante** : Le modèle Whisper `tiny` (~75 MB) est préchargé dans l'image Docker lors du build pour éviter de le télécharger à chaque démarrage du conteneur. Cela augmente la taille de l'image mais améliore significativement les performances au runtime.

### Étape 4 : Déployer avec Helm

```bash
# Vérifier que le chart Helm est valide (pas d'erreurs de syntaxe)
helm lint helm-chart

# Déployer l'application sur le cluster Kubernetes
helm install audio-to-text ./helm-chart
```

**En cas de mise à jour du code** :

```bash
# Rebuild et push l'image avec les changements
docker build -t votre-pseudo-dockerhub/audio-to-text:latest .
docker push votre-pseudo-dockerhub/audio-to-text:latest

# Mettre à jour le déploiement
helm upgrade audio-to-text ./helm-chart
```

### Étape 5 : Vérifier le déploiement

```bash
# Voir les pods (attendre qu'ils passent à l'état "Running")
kubectl get pods -w
# Appuyer sur Ctrl+C pour sortir du mode watch

# Voir les services déployés
kubectl get services

# Récupérer l'URL publique de l'application
kubectl get ingress
```

L'API sera accessible à l'adresse : `https://audio-to-text-user-votre-username.lab.sspcloud.fr`

### Étape 6 : Tester l'API déployée

Modifier le fichier `interact_with_app/interact.py` pour utiliser votre URL personnalisée :

```python
API_URL = "https://audio-to-text-user-votre-username.lab.sspcloud.fr/transcribe"
```

Puis exécuter le script de test :

```bash
python interact_with_app/interact.py
```

Le script ouvrira un explorateur de fichiers pour sélectionner un fichier audio et affichera la transcription.

## Utilisation

### Endpoints disponibles

- **GET** `/` : Informations sur le serveur (version du modèle, device utilisé)
- **POST** `/transcribe` : Transcription d'un fichier audio

### Formats audio supportés

- WAV (`.wav`)
- MP3 (`.mp3`)
