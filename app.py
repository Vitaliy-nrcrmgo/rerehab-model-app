
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Навчальні дані (усічений набір для демо)
data = {
    'IMT': [18.4, 22.3, 27.1, 21.0],
    'WT': [0.37, 0.40, 0.55, 0.41],
    'ZSU': [2, 1, 1, 2],
    'Eva': [2, 2, 1, 1],
    'Сприйнятий дистрес': [8, 12, 20, 14],
    'GSI самоефективність': [31, 26, 21, 23],
    'Stress': [2, 1, 2, 2]
}
df = pd.DataFrame(data)

X = df.drop("Stress", axis=1)
y = df["Stress"]

# Навчаємо модель у коді (невелика для демонстрації)
model = GradientBoostingClassifier(random_state=42)
model.fit(X, y)

st.title("Прогноз потреби у реабілітації у дітей (демо-версія)")

st.markdown("Введіть значення для кожної з ознак:")

IMT = st.number_input("IMT – Індекс маси тіла", min_value=10.0, max_value=40.0, step=0.1)
WT = st.number_input("WT – Талія/Зріст (WHtR)", min_value=0.2, max_value=1.0, step=0.01)
ZSU = st.selectbox("ZSU – Перебування на окупованій території", [1, 2])
Eva = st.selectbox("Eva – Евакуація з території", [1, 2])
distress = st.slider("Сприйнятий дистрес", 0, 30, 15)
gsi = st.slider("GSI самоефективність", 0, 30, 15)

if st.button("Прогнозувати"):
    X_input = np.array([[IMT, WT, ZSU, Eva, distress, gsi]])
    prediction = model.predict(X_input)[0]
    if prediction == 2:
        st.error("Результат: Дитина ПОТРЕБУЄ реабілітаційних заходів.")
    else:
        st.success("Результат: Дитина НЕ потребує реабілітаційних заходів.")
