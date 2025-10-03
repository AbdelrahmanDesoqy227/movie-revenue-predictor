import streamlit as st
import ast
import joblib
import numpy as np

# Load models and vectorizers
xgb_reg = joblib.load("xgb_regressor.pkl")
xgb_cls = joblib.load("xgb_classifier.pkl")
# Load vectorizers & dictionaries
vectorizer_genres = joblib.load("vectorizer_genres.pkl")
vectorizer_keywords = joblib.load("vectorizer_keywords.pkl")
vectorizer_companies = joblib.load("vectorizer_companies.pkl")

freq_dicts = joblib.load("freq_dicts.pkl")
target_dicts = joblib.load("target_dicts.pkl")

# Global mean revenue as fallback
global_mean_revenue = 82275112

def preprocess_input(genres, keywords, companies, actor_1, actor_2, actor_3, director, budget):
    genres = [item.lower() for item in genres]
    keywords = [item.lower() for item in keywords]
    companies = [item.lower() for item in companies]
    actor_1 = actor_1.lower()
    actor_2 = actor_2.lower()
    actor_3 = actor_3.lower()
    director = director.lower()
    # 1️⃣ Encode text features
    genres_encoded = vectorizer_genres.transform([" ".join(genres)]).toarray()
    keywords_encoded = vectorizer_keywords.transform([" ".join(keywords)]).toarray()
    companies_encoded = vectorizer_companies.transform([" ".join(companies)]).toarray()

    # 2️⃣ Safe lookup
    def get_feature_value(mapping, key, default=0):
        return mapping.get(key, default)

    # Frequencies
    actor_1_freq = get_feature_value(freq_dicts['actor_1'], actor_1)
    actor_2_freq = get_feature_value(freq_dicts['actor_2'], actor_2)
    actor_3_freq = get_feature_value(freq_dicts['actor_3'], actor_3)
    director_freq_val = get_feature_value(freq_dicts['director'], director)

    # Target encodings
    actor_1_target = get_feature_value(target_dicts['actor_1'], actor_1, global_mean_revenue)
    actor_2_target = get_feature_value(target_dicts['actor_2'], actor_2, global_mean_revenue)
    actor_3_target = get_feature_value(target_dicts['actor_3'], actor_3, global_mean_revenue)
    director_target_val = get_feature_value(target_dicts['director'], director, global_mean_revenue)

    # 3️⃣ Numeric features
    numeric_features = np.array([
        actor_1_freq, actor_2_freq, actor_3_freq, director_freq_val,
        actor_1_target, actor_2_target, actor_3_target, director_target_val,
        budget
    ]).reshape(1, -1)

    # 4️⃣ Combine
    X_input = np.hstack([
        genres_encoded,
        keywords_encoded,
        companies_encoded,
        numeric_features
    ])

    return X_input


# ------------------- Streamlit UI -------------------
st.title("🎬 Movie Success & Revenue Predictor")
st.write("Enter movie details and get predictions 👇")

# User inputs
genres = st.text_input("Genres (example: \"Drama\",\"Comedy\",\"Sci-Fi\")", "\"Action\",\"Adventure\"")
keywords = st.text_input("Keywords (example: \"future\",\"hero\")", "\"future\",\"hero\"")
companies = st.text_input("Production Companies (example: \"WarnerBros\",\"Disney\")", "\"WarnerBros\",\"Disney\"")

actor_1 = st.text_input("Actor 1", "Tom Cruise")
actor_2 = st.text_input("Actor 2", "Zoe Saldana")
actor_3 = st.text_input("Actor 3", "Johnny Depp")
director = st.text_input("Director", "Christopher Nolan")
budget = st.number_input("Budget ($)", min_value=1000000, step=1000000)

# Prediction button
if st.button("🔮 Predict"):
    try:
        # Convert strings to lists using ast.literal_eval
        genres_list = ast.literal_eval(f"[{genres}]")
        keywords_list = ast.literal_eval(f"[{keywords}]")
        companies_list = ast.literal_eval(f"[{companies}]")

        # Preprocess input
        new_movie = preprocess_input(
            genres_list,
            keywords_list,
            companies_list,
            actor_1,
            actor_2,
            actor_3,
            director,
            budget
        )

        st.success("✅ Preprocessing done! Ready for prediction.")


    except Exception as e:
        st.error(f"⚠️ Error in input format: {e}")

    # Predict revenue
    reg_pred = xgb_reg.predict(new_movie)
    predicted_revenue = int(round(np.expm1(reg_pred[0])))

    # Predict success + probability
    cls_pred = xgb_cls.predict(new_movie)
    cls_prob = xgb_cls.predict_proba(new_movie)[0][1]

    # Show results
    st.subheader("📊 Results")
    st.write(f"🎬 **Predicted Revenue:** ${predicted_revenue:,}")
    st.write(f"✅ **Success Prediction:** {'Yes' if cls_pred[0] == 1 else 'No'} "
             f"(Confidence: {cls_prob*100:.2f}%)")
