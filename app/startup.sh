#!/bin/bash
cd /home/site/wwwroot/app

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier si les données existent, sinon les télécharger
if [ ! -f "data/training.1600000.processed.noemoticon.csv" ]; then
    echo "Téléchargement des données..."
    python scripts/download_data.py
fi

# Lancer l'API
exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info

#!/bin/bash
# startup.sh

# Se placer dans le dossier app
# cd /home/site/wwwroot/app

# # Upgrade pip et installer les dépendances
# echo "Installation des dépendances..."
# pip install --upgrade pip
# pip install -r requirements.txt

# # Vérifier que l'installation s'est bien passée
# echo "Vérification de l'installation..."
# pip list

# # Lancer l'application FastAPI
# echo "Démarrage de l'application..."
# exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}



# #!/bin/bash
# # startup.sh

# # Se placer dans le dossier app
# cd /home/site/wwwroot/app

# # Upgrade pip et installer les dépendances
# echo "Installation des dépendances..."
# pip install --upgrade pip
# pip install -r requirements.txt

# # Lancer l'application FastAPI
# echo "Démarrage de l'application..."
# exec uvicorn api:app --host 0.0.0.0 --port 8000
