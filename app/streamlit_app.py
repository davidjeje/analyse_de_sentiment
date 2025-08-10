import streamlit as st
import requests

API_URL_PREDICT = "http://127.0.0.1:8000/predict"
API_URL_TWEETS = "http://127.0.0.1:8000/tweets"

st.title("Analyse de sentiment avec API FastAPI + MLflow")

def load_tweets_once():
    if "tweets_list" not in st.session_state:
        try:
            resp = requests.get(API_URL_TWEETS)
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("tweets", [])
            # Nettoyage : enlever tweets vides ou None
            tweets = [t for t in tweets if t and t.strip()]
            st.session_state["tweets_list"] = tweets
        except Exception as e:
            st.error(f"Erreur lors du chargement des tweets : {e}")
            st.session_state["tweets_list"] = []

load_tweets_once()

tweets_list = st.session_state.get("tweets_list", [])

if not tweets_list:
    st.warning("Aucun tweet chargé.")
else:
    # Initialise la sélection s'il n'y a pas encore de sélection
    if "selected_tweet" not in st.session_state:
        st.session_state["selected_tweet"] = tweets_list[0]

    selected_tweet = st.selectbox(
        "Choisissez un tweet à analyser :",
        tweets_list,
        index=tweets_list.index(st.session_state["selected_tweet"]),
        key="tweet_selectbox"
    )

    # Met à jour la session_state quand on change la sélection
    if selected_tweet != st.session_state["selected_tweet"]:
        st.session_state["selected_tweet"] = selected_tweet

    if st.button("Prédire le sentiment"):
        if not selected_tweet.strip():
            st.warning("Veuillez sélectionner un tweet valide.")
        else:
            try:
                response = requests.post(API_URL_PREDICT, json={"text": selected_tweet})
                response.raise_for_status()
                prediction = response.json()

                sentiment = prediction.get("sentiment", "N/A")
                confidence = prediction.get("confidence", 0)

                st.subheader("Résultat de la prédiction")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", sentiment)
                with col2:
                    st.metric("Confiance", f"{confidence:.4f}")

                if sentiment == "positive":
                    st.balloons()
                elif sentiment == "negative":
                    st.error("Attention, tweet à surveiller !")
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la requête à l'API : {e}")



# import streamlit as st
# import requests

# API_URL_PREDICT = "http://127.0.0.1:8000/predict"
# API_URL_TWEETS = "http://127.0.0.1:8000/tweets"

# st.title("Analyse de sentiment avec API FastAPI + MLflow")

# @st.cache_data(ttl=3600, show_spinner=True)
# def load_tweets():
#     try:
#         resp = requests.get(API_URL_TWEETS)
#         resp.raise_for_status()
#         data = resp.json()
#         tweets = data.get("tweets", [])
#         # Nettoyage : retirer les tweets vides ou nuls
#         tweets = [t for t in tweets if t and t.strip()]
#         return tweets
#     except Exception as e:
#         st.error(f"Erreur lors du chargement des tweets : {e}")
#         return []

# tweets_list = load_tweets()

# if tweets_list:
#     selected_tweet = st.selectbox(
#         "Choisissez un tweet à analyser :", 
#         tweets_list, 
#         key="tweet_selectbox"
#     )
# else:
#     st.warning("Aucun tweet chargé.")

# if st.button("Prédire le sentiment") and tweets_list:
#     if not selected_tweet.strip():
#         st.warning("Veuillez sélectionner un tweet valide.")
#     else:
#         try:
#             response = requests.post(API_URL_PREDICT, json={"text": selected_tweet})
#             response.raise_for_status()
#             prediction = response.json()

#             sentiment = prediction.get("sentiment", "N/A")
#             confidence = prediction.get("confidence", 0)

#             st.subheader("Résultat de la prédiction")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Sentiment", sentiment)
#             with col2:
#                 st.metric("Confiance", f"{confidence:.4f}")

#             if sentiment == "positive":
#                 st.balloons()
#             elif sentiment == "negative":
#                 st.error("Attention, tweet à surveiller !")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Erreur lors de la requête à l'API : {e}")




# app.py
# import streamlit as st
# import requests

# API_URL_PREDICT = "http://127.0.0.1:8000/predict"
# API_URL_TWEETS = "http://127.0.0.1:8000/tweets"

# st.title("Analyse de sentiment avec API FastAPI + MLflow")

# @st.cache_data(show_spinner=True)
# def load_tweets():
#     try:
#         resp = requests.get(API_URL_TWEETS)
#         resp.raise_for_status()
#         data = resp.json()
#         return data.get("tweets", [])
#     except Exception as e:
#         st.error(f"Erreur lors du chargement des tweets : {e}")
#         return []

# tweets_list = load_tweets()

# if tweets_list:
#     selected_tweet = st.selectbox("Choisissez un tweet à analyser :", tweets_list)
# else:
#     st.warning("Aucun tweet chargé.")

# if st.button("Prédire le sentiment") and tweets_list:
#     if not selected_tweet.strip():
#         st.warning("Veuillez sélectionner un tweet valide.")
#     else:
#         try:
#             response = requests.post(API_URL_PREDICT, json={"text": selected_tweet})
#             response.raise_for_status()
#             prediction = response.json()

#             sentiment = prediction.get("sentiment", "N/A")
#             confidence = prediction.get("confidence", 0)

#             st.subheader("Résultat de la prédiction")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Sentiment", sentiment)
#             with col2:
#                 st.metric("Confiance", f"{confidence:.4f}")

#             if sentiment == "positive":
#                 st.balloons()
#             elif sentiment == "negative":
#                 st.error("Attention, tweet à surveiller !")
#         except requests.exceptions.RequestException as e:
#             st.error(f"Erreur lors de la requête à l'API : {e}")



# # app.py
# import streamlit as st
# import requests
# import pandas as pd

# # URL de ton API FastAPI
# API_URL = "http://127.0.0.1:8000/predict"

# st.title("Analyse de sentiment avec API FastAPI + MLflow")
# st.write("Entrez un texte et obtenez la prédiction du sentiment.")

# # Saisie utilisateur
# user_input = st.text_area("Votre texte :", "I love this product, it is amazing!")

# if st.button("Prédire le sentiment"):
#     if user_input.strip():
#         # Envoi à l'API
#         response = requests.post(API_URL, json={"text": user_input})
        
#         if response.status_code == 200:
#             result = response.json()
#             st.subheader("Résultat")
#             st.write(f"**Sentiment :** {result['sentiment']}")
#             st.write(f"**Confiance :** {result['confidence']:.2f}")
#         else:
#             st.error(f"Erreur {response.status_code} : {response.text}")