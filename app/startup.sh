#!/bin/bash
# startup.sh

# Se placer dans le dossier app
cd /home/site/wwwroot/app

# Upgrade pip et installer les dépendances
echo "Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Lancer l'application FastAPI
echo "Démarrage de l'application..."
exec uvicorn api:app --host 0.0.0.0 --port 8000
