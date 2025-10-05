import streamlit as st
import ast
import joblib
import numpy as np

# ------------------- CINEMATIC PAGE STYLE -------------------
page_bg = """
<style>
/* Background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1524985069026-dd778a71c7b4?auto=format&fit=crop&w=1950&q=80");
    background-size: cover;
    background-position: center;
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}

/* Add dark overlay for better text contrast */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: rgba(0, 0, 0, 0.6);
    z-index: 0;
}

/* Container styling */
.block-container {
    background-color: rgba(0, 0, 0, 0.6);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.6);
    position: relative;
    z-index: 1;
}

/* Input boxes */
input, textarea {
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: black !important;
    border-radius: 10px;
    border: 1px solid #E50914 !important; /* Netflix red accent */
}

/* Number input box */
div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.1);
    color: white;
}

/* Buttons */
div.stButton > button:first-child {
    background: linear-gradient(to right, #E50914, #B81D24);
    color: white;
    border-radius: 12px;
    border: none;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
}
div.stButton > button:hover {
    background: linear-gradient(to right, #B81D24, #E50914);
    transform: scale(1.05);
    box-shadow: 0 6px 25px rgba(229, 9, 20, 0.6);
}

/* Titles */
h1 {
    color: #E50914;
    text-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
    font-weight: 900;
    font-size: 2.5rem;
}
h2, h3, h4, h5 {
    color: #FFD700; /* gold accent */
    font-weight: 700;
}

/* Success box */
.stSuccess {
    background-color: rgba(34,139,34,0.3);
    border-left: 4px solid #00FF7F;
}

/* Result text */
.result-box {
    background-color: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #FFD700;
    box-shadow: 0 4px 10px rgba(255, 215, 0, 0.2);
    color: #fff;
    font-size: 1.1rem;
}
/* Label color for input fields */
label, .stTextInput label, .stNumberInput label {
    color: #FFD700 !important;  /* Gold color */
    font-weight: 700;
    font-size: 1.05rem;
}
/*
🔴 Netflix red: color: #E50914 !important;

🔵 Cool cyan: color: #00FFFF !important;

⚪ Bright white: color: #FFFFFF !important;
*/
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
# ---------------------------------------------------


# Load models and vectorizers
xgb_reg = joblib.load("xgb_regressor.pkl")
xgb_cls = joblib.load("xgb_classifier.pkl")
vectorizer_genres = joblib.load("vectorizer_genres.pkl")
vectorizer_keywords = joblib.load("vectorizer_keywords.pkl")
vectorizer_companies = joblib.load("vectorizer_companies.pkl")
freq_dicts = joblib.load("freq_dicts.pkl")
target_dicts = joblib.load("target_dicts.pkl")

global_mean_revenue = 82275112


def preprocess_input(genres, keywords, companies, actor_1, actor_2, actor_3, director, budget):
    genres = [item.lower() for item in genres]
    keywords = [item.lower() for item in keywords]
    companies = [item.lower() for item in companies]
    actor_1, actor_2, actor_3, director = map(str.lower, [actor_1, actor_2, actor_3, director])

    genres_encoded = vectorizer_genres.transform([" ".join(genres)]).toarray()
    keywords_encoded = vectorizer_keywords.transform([" ".join(keywords)]).toarray()
    companies_encoded = vectorizer_companies.transform([" ".join(companies)]).toarray()

    def get_feature_value(mapping, key, default=0):
        return mapping.get(key, default)

    actor_1_freq = get_feature_value(freq_dicts['actor_1'], actor_1)
    actor_2_freq = get_feature_value(freq_dicts['actor_2'], actor_2)
    actor_3_freq = get_feature_value(freq_dicts['actor_3'], actor_3)
    director_freq_val = get_feature_value(freq_dicts['director'], director)

    actor_1_target = get_feature_value(target_dicts['actor_1'], actor_1, global_mean_revenue)
    actor_2_target = get_feature_value(target_dicts['actor_2'], actor_2, global_mean_revenue)
    actor_3_target = get_feature_value(target_dicts['actor_3'], actor_3, global_mean_revenue)
    director_target_val = get_feature_value(target_dicts['director'], director, global_mean_revenue)

    numeric_features = np.array([
        actor_1_freq, actor_2_freq, actor_3_freq, director_freq_val,
        actor_1_target, actor_2_target, actor_3_target, director_target_val,
        budget
    ]).reshape(1, -1)

    X_input = np.hstack([
        genres_encoded,
        keywords_encoded,
        companies_encoded,
        numeric_features
    ])
    return X_input


# ------------------- Streamlit UI -------------------
st.title("🍿 Movie Success & Revenue Predictor")
st.write("Get your blockbuster predictions below! 🎥")

genres = st.text_input("🎭 Genres", "\"Action\",\"Adventure\"")
keywords = st.text_input("🔑 Keywords", "\"future\",\"hero\"")
companies = st.text_input("🏢 Production Companies", "\"WarnerBros\",\"Disney\"")

actor_1 = st.text_input("⭐ Actor 1", "Tom Cruise")
actor_2 = st.text_input("⭐ Actor 2", "Zoe Saldana")
actor_3 = st.text_input("⭐ Actor 3", "Johnny Depp")
director = st.text_input("🎬 Director", "Christopher Nolan")
budget = st.number_input("💰 Budget ($)", min_value=1000000, step=1000000)

if st.button("🎯 Predict Now"):
    try:
        genres_list = ast.literal_eval(f"[{genres}]")
        keywords_list = ast.literal_eval(f"[{keywords}]")
        companies_list = ast.literal_eval(f"[{companies}]")

        new_movie = preprocess_input(
            genres_list,
            keywords_list,
            companies_list,
            actor_1, actor_2, actor_3, director, budget
        )

        st.success("✅ Preprocessing done! Predicting...")

        reg_pred = xgb_reg.predict(new_movie)
        predicted_revenue = int(round(np.expm1(reg_pred[0])))

        cls_pred = xgb_cls.predict(new_movie)
        cls_prob = xgb_cls.predict_proba(new_movie)[0][1]

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"### 💵 **Predicted Revenue:** ${predicted_revenue:,}")
        st.markdown(f"### 🏆 **Will it be a Hit?** {'🎉 Yes!' if cls_pred[0] == 1 else '❌ No'} "
                    f"(Confidence: **{cls_prob*100:.2f}%**)")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Error in input format: {e}")
