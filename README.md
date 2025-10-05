# 🎬 **Movie Revenue and Success Predictor**

An advanced **Machine Learning web app** that predicts a movie’s **box office revenue** 💰 and **success likelihood** 🌟 based on metadata such as genres, keywords, production companies, cast, director, and budget.

---

## 🧠 **Project Overview**

The goal of this project is to analyze and model the **TMDB 5000 Movies dataset** to:
- 📈 Predict **movie revenue (regression)** using **XGBoost Regressor**
- ✅ Classify **movie success (classification)** using **XGBoost Classifier**

Both models were trained after deep **EDA**, advanced **feature engineering**, and **data preprocessing** to extract meaningful patterns from metadata.

---

## 📊 **Dataset**

- **Source:** [TMDB 5000 Movies Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Total Movies:** ~5000  
- **Features Used:**  
  Genres, Keywords, Production Companies, Actors, Director, and Budget.

---

## 🔍 **Exploratory Data Analysis (EDA)**

Performed detailed visual exploration to understand:
- 🎭 Genre distributions  
- 🎥 Top-grossing actors, directors, and production companies  
- 💸 Correlations between budget, popularity, and revenue  
<img width="643" height="385" alt="Screenshot (75)" src="https://github.com/user-attachments/assets/d10f8894-3f84-4d90-8711-2c0ba8bf8f25" />
<img width="585" height="378" alt="Screenshot (76)" src="https://github.com/user-attachments/assets/57a2533c-8e33-47e7-8679-a655ccecb8ff" />
<img width="588" height="450" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/ba37618d-f872-497b-a32a-ccc002127b37" />
<img width="458" height="326" alt="Screenshot (78)" src="https://github.com/user-attachments/assets/be2c7f9e-20b1-437a-8844-9b7f498f7123" />
<img width="672" height="451" alt="Screenshot (79)" src="https://github.com/user-attachments/assets/03c88e5a-4fee-4912-9410-bd716719a61a" />
<img width="665" height="433" alt="Screenshot (80)" src="https://github.com/user-attachments/assets/a8bf8a16-7f83-4da5-b3a9-d8da16e0e665" />
<img width="644" height="431" alt="Screenshot (81)" src="https://github.com/user-attachments/assets/0182a1e2-c57f-412c-a87b-759e2e0ebd4a" />
<img width="434" height="463" alt="Screenshot (82)" src="https://github.com/user-attachments/assets/2f3b9d18-02ec-4e2a-b29d-05668dc8bec0" />
<img width="463" height="446" alt="Screenshot (83)" src="https://github.com/user-attachments/assets/01694f93-4dfa-46aa-86ba-a93e66c9665a" />
<img width="816" height="360" alt="Screenshot (84)" src="https://github.com/user-attachments/assets/b91a0cdc-5ca4-416b-bd6e-ee6220f90cf2" />
<img width="399" height="182" alt="Screenshot (85)" src="https://github.com/user-attachments/assets/548e7863-7793-482f-95ef-d498271509c9" />
<img width="606" height="407" alt="Screenshot (86)" src="https://github.com/user-attachments/assets/b4888c05-8a4b-4f55-914e-0ad4f1d7aa6b" />
<img width="412" height="316" alt="Screenshot (87)" src="https://github.com/user-attachments/assets/58349254-66ff-43db-b78a-43d58d80d0f2" />


EDA was done using:
```python
Python, Pandas, NumPy, Matplotlib, Seaborn
````

---

## ⚙️ **Feature Engineering & Preprocessing**

* Built **global actor frequency** and **target revenue encodings**
* Encoded genres, keywords, and companies into numerical representations
* Created **actor-level statistics** combining actor_1, actor_2, and actor_3
* Filled missing values intelligently to maintain data consistency
* Split dataset into **train/test sets** for fair evaluation

---

## 🤖 **Modeling**

Trained and evaluated two models:

| Task                      | Model         | R² / Accuracy      | Description                                        |
| ------------------------- | ------------- | ------------------ | -------------------------------------------------- |
| 💰 Revenue Prediction     | XGBRegressor  | R² = **0.87**      | Predicts expected box office revenue               |
| 🌟 Success Classification | XGBClassifier | Accuracy = **86%** | Predicts whether a movie will be successful or not |

---

## 🧪 **Evaluation Metrics**

### 🔹 Regression:

* **RMSE:** 2.84
* **R² Score:** 0.87

### 🔹 Classification:

|   Metric  | Class 0 | Class 1 |
| :-------: | :-----: | :-----: |
| Precision |   0.88  |   0.84  |
|   Recall  |   0.85  |   0.87  |
|  F1-score |   0.87  |   0.85  |

✅ **Overall Accuracy:** 86%

---

## 🖥️ **Streamlit Web Application**

Developed an interactive **Streamlit web app** that allows users to:

* Enter movie details (genres, keywords, companies, actors, director, budget)
* Run real-time predictions for:

  * 🎯 **Revenue (numeric prediction)**
  * ✅ **Success likelihood (classification)**

### ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧩 **Tech Stack**

```
Python | Pandas | NumPy | Seaborn | Scikit-learn | XGBoost | Matplotlib | Streamlit
```

---

## 🧠 **Project Insights**

This project demonstrates:

* End-to-end **ML pipeline development**
* Integration of **regression and classification models**
* Advanced **feature engineering** from unstructured text data
* **Streamlit deployment** for user interaction

---

## 🖼️ **Sample Preview**
<img width="1366" height="590" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/74232e15-7962-496b-a578-a51b9d0231aa" />
<img width="1366" height="586" alt="Screenshot (74)" src="https://github.com/user-attachments/assets/9580fa42-fd74-4569-a6ef-4e5a0b896e17" />



---

## 📂 **Repository Structure**

```
movie_app/
│
├── app.py                   # Streamlit main file
├── requirements.txt          # Dependencies
├── model_regressor.pkl       # Trained XGBRegressor model
├── model_classifier.pkl      # Trained XGBClassifier model
├── preprocessing_utils.py    # Preprocessing functions
├── movie-success-prediction.ipynb                  # NoteBook Code
```

---
