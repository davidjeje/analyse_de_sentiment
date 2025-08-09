import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Téléchargements nécessaires une seule fois
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialiser stopwords et lemmatiseur
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Fonction de nettoyage complète
def preprocess_text(text):
    # Mise en minuscules
    text = text.lower()

    # Suppression des URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Suppression mentions et hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Suppression ponctuation et chiffres
    text = re.sub(r"[^a-z\s]", "", text)

    # Suppression des mots composés d’une seule lettre répétée (ex: "aaaa", "hahaha")
    text = re.sub(r'\b([a-z])\1{2,}\b', '', text)

    # Tokenisation
    tokens = nltk.word_tokenize(text)

    # Lemmatisation + filtrage
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalpha() and len(word) >= 2 and word not in stop_words
    ]

    return " ".join(tokens)
    
    # Fonction pour illustrer les étapes de nettoyage
def illustrate_preprocessing(original_text, step_name):
    print(f"\n--- Étape: {step_name} ---")
    print(f"Original: '{original_text}'")

    current_text = original_text

    # Mise en minuscules
    if step_name == "Mise en minuscules":
        current_text = original_text.lower()
        print(f"Nettoyé:  '{current_text}'")
        return current_text

    # Suppression des URLs
    elif step_name == "Suppression des URLs":
        current_text = re.sub(r"http\S+|www\S+|https\S+", "", original_text)
        print(f"Nettoyé:  '{current_text}'")
        return current_text

    # Suppression mentions et hashtags
    elif step_name == "Suppression mentions et hashtags":
        current_text = re.sub(r"@\w+|#\w+", "", original_text)
        print(f"Nettoyé:  '{current_text}'")
        return current_text

    # Suppression ponctuation et chiffres
    elif step_name == "Suppression ponctuation et chiffres":
        current_text = re.sub(r"[^a-z\s]", "", original_text)
        print(f"Nettoyé:  '{current_text}'")
        return current_text

    # Suppression des mots composés d’une seule lettre répétée
    elif step_name == "Suppression des mots répétés":
        current_text = re.sub(r'\b([a-z])\1{2,}\b', '', original_text)
        print(f"Nettoyé:  '{current_text}'")
        return current_text

    # Tokenisation + Lemmatisation + filtrage (cette étape doit être appliquée sur un texte déjà nettoyé)
    elif step_name == "Tokenisation, Lemmatisation et Stopwords":
        # Pour cette étape, nous partons du principe que les étapes précédentes ont été appliquées.
        # Nous allons donc appliquer une version de votre fonction qui va jusqu'à cette étape.
        temp_text = original_text.lower()
        temp_text = re.sub(r"http\S+|www\S+|https\S+", "", temp_text)
        temp_text = re.sub(r"@\w+|#\w+", "", temp_text)
        temp_text = re.sub(r"[^a-z\s]", "", temp_text)
        temp_text = re.sub(r'\b([a-z])\1{2,}\b', '', temp_text)

        tokens = nltk.word_tokenize(temp_text)
        current_text = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and len(word) >= 2 and word not in stop_words
        ]
        current_text = " ".join(current_text)
        print(f"Nettoyé:  '{current_text}'")
        return current_text
    else:
        return original_text