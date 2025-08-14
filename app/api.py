from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import mlflow
import pandas as pd
import os
from typing import List

# Azure Application Insights
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Kaggle API pour télécharger les données
from kaggle.api.kaggle_api_extended import KaggleApi

# Config logger pour Azure Application Insights
logger = logging.getLogger(__name__)
INSTRUMENTATION_KEY = os.getenv(
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34"
)
logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
logger.setLevel(logging.INFO)

app = FastAPI()

# L’URI que tu as donné pointe vers un run stocké sur ton tracking server
model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# Exporter le modèle en local (si pas déjà téléchargé)
if not os.path.exists("./mlflow_model"):
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path="./mlflow_model"
    )

# 📌 Définition du chemin vers le fichier CSV
DATA_PATH = os.path.join("app", "data", "training.1600000.processed.noemoticon.csv")

# 📥 Télécharger depuis Kaggle si le fichier n'existe pas
if not os.path.exists(DATA_PATH):
    logger.info(f"Fichier de données non trouvé à {DATA_PATH}, téléchargement depuis Kaggle...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        "kazanova/sentiment140",
        file_name="training.1600000.processed.noemoticon.csv",
        path=os.path.dirname(DATA_PATH)
    )
    logger.info("Téléchargement terminé ✅")

class TweetIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    sentiment: str
    confidence: float

class TweetsOut(BaseModel):
    tweets: List[str]

label_mapping = {
    0: "negative",
    2: "neutral",
    4: "positive"
}

@app.get("/")
def read_root():
    return {"message": "API FastAPI OK"}

@app.get("/health")
def health_check():
    logger.info("Health check OK")
    return {"status": "ok"}

@app.get("/tweets", response_model=TweetsOut)
def get_tweets(sample_frac: float = 0.15):
    if not os.path.exists(DATA_PATH):
        logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
        raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

    col_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
    df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    logger.info(f"Retourne {len(df_text)} tweets en échantillon")
    return TweetsOut(tweets=df_text.tolist())

@app.post("/predict", response_model=PredictionOut)
def predict_sentiment(tweet: TweetIn):
    text = [tweet.text]

    pred = model.predict(text)
    pred_label = int(pred[0])

    sentiment_label = label_mapping.get(pred_label, "unknown")
    confidence = 1.0

    logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
    return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# Tu peux ensuite le recharger depuis le chemin local
local_model_uri = "./mlflow_model"
model = mlflow.pyfunc.load_model(local_model_uri)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import mlflow
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# # Remplace "TON_INSTRUMENTATION_KEY" par ta vraie clé dans les variables d'env ou secrets
# INSTRUMENTATION_KEY = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34")
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # L’URI que tu as donné pointe vers un run stocké sur ton tracking server
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"

# # Exporter le modèle en local (si pas déjà téléchargé)
# if not os.path.exists("./mlflow_model"):
#     mlflow.artifacts.download_artifacts(
#         artifact_uri=model_uri,
#         dst_path="./mlflow_model"
#     )

# DATA_PATH = "data/training.1600000.processed.noemoticon.csv"

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)

# # Tu peux ensuite le recharger depuis le chemin local
# local_model_uri = "./mlflow_model"
# model = mlflow.pyfunc.load_model(local_model_uri)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.pyfunc
# import pandas as pd
# import os
# from typing import List

# # Azure Application Insights
# from opencensus.ext.azure.log_exporter import AzureLogHandler
# import logging

# # Config logger pour Azure Application Insights
# logger = logging.getLogger(__name__)
# # Remplace "TON_INSTRUMENTATION_KEY" par ta vraie clé dans les variables d'env ou secrets
# INSTRUMENTATION_KEY = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "dc668c66-e558-48aa-aedd-845404a47a18;IngestionEndpoint=https://canadacentral-1.in.applicationinsights.azure.com/;LiveEndpoint=https://canadacentral.livediagnostics.monitor.azure.com/;ApplicationId=7f4f2e23-49d7-4d86-82e9-f80804203f34")
# logger.addHandler(AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"))
# logger.setLevel(logging.INFO)

# app = FastAPI()

# # Charge le modèle MLflow enregistré (au démarrage)
# model_uri = "runs:/62c9722eb896400dabe73d9302cddea7/model"
# model = mlflow.pyfunc.load_model(model_uri)

# DATA_PATH = "data/training.1600000.processed.noemoticon.csv"

# class TweetIn(BaseModel):
#     text: str

# class PredictionOut(BaseModel):
#     sentiment: str
#     confidence: float

# class TweetsOut(BaseModel):
#     tweets: List[str]

# label_mapping = {
#     0: "negative",
#     2: "neutral",
#     4: "positive"
# }

# @app.get("/")
# def read_root():
#     return {"message": "API FastAPI OK"}

# @app.get("/health")
# def health_check():
#     logger.info("Health check OK")
#     return {"status": "ok"}

# @app.get("/tweets", response_model=TweetsOut)
# def get_tweets(sample_frac: float = 0.15):
#     if not os.path.exists(DATA_PATH):
#         logger.error(f"Fichier de données non trouvé à {DATA_PATH}")
#         raise HTTPException(status_code=404, detail=f"Fichier de données non trouvé à {DATA_PATH}")

#     col_names = ["target", "ids", "date", "flag", "user", "text"]
#     df = pd.read_csv(DATA_PATH, encoding="latin-1", names=col_names, header=None)
#     df_text = df["text"].sample(frac=sample_frac, random_state=42).reset_index(drop=True)

#     logger.info(f"Retourne {len(df_text)} tweets en échantillon")
#     return TweetsOut(tweets=df_text.tolist())

# @app.post("/predict", response_model=PredictionOut)
# def predict_sentiment(tweet: TweetIn):
#     text = [tweet.text]

#     pred = model.predict(text)
#     pred_label = int(pred[0])

#     sentiment_label = label_mapping.get(pred_label, "unknown")
#     confidence = 1.0

#     logger.info(f"Prediction réalisée: texte='{tweet.text[:50]}...', sentiment={sentiment_label}")
#     return PredictionOut(sentiment=sentiment_label, confidence=confidence)
