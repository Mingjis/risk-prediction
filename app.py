import streamlit as st
import pandas as pd
import pickle
import os
import requests
from catboost import CatBoostClassifier

# 🎯 Google Drive에서 모델 다운로드 함수
def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)

# ✅ Google Drive 파일 ID 설정
injury_type_model_id = "1mYGG3lZQDJwsaqSXgvC8lB0BHJmqHSap"
injury_type_model_path = "injury_type_model.cbm"

# 🎯 모델 및 리소스 로딩
@st.cache_resource
def load_models():
    # 📥 부상유형 모델 다운로드
    download_model_from_drive(injury_type_model_id, injury_type_model_path)

    # 📦 기인물 모델 (로컬 포함 가정)
    cause_model = CatBoostClassifier()
    cause_model.load_model("cause_material_model.cbm")

    # 📦 부상유형 모델
    injury_model = CatBoostClassifier()
    injury_model.load_model(injury_type_model_path)

    # 📦 위험도 딕셔너리
    with open("risk_model_average.pkl", "rb") as f:
        risk_data = pickle.load(f)

    # 📦 인코더
    with open("encoders_cause.pkl", "rb") as f:
        encoders_cause = pickle.load(f)
    with open("encoders_injury.pkl", "rb") as f:
        encoders_injury = pickle.load(f)

    return cause_model, injury_model, risk_data, encoders_cause, encoders_injury

# 📦 로드
cause_model, injury_model, risk_data, encoders_cause, encoders_injury = load_models()

# 🎛️ 사용자 입력
st.title("🏗️ 건설 재해 사망 위험도 예측기")
st.markdown("**아래 정보를 입력하면 사고유형, 기인물, 위험도를 예측해줍니다**")

project_scale = st.selectbox("Project scale", encoders_cause['Project scale'].classes_)
facility_type = st.selectbox("Facility type", encoders_cause['Facility type'].classes_)
work_type = st.selectbox("Work type", encoders_cause['Work type'].classes_)

if st.button("위험도 예측"):
    # 🧩 인코딩
    encoded_values = [
        encoders_cause['Project scale'].transform([project_scale])[0],
        encoders_cause['Facility type'].transform([facility_type])[0],
        encoders_cause['Work type'].transform([work_type])[0]
    ]
    columns = ["Project scale", "Facility type", "Work type"]
    x_input = pd.DataFrame([encoded_values], columns=columns)

    # 🧠 기인물 예측
    x_input_cause = x_input[columns]  # 그대로 사용
    pred_cause = cause_model.predict(x_input_cause)[0]
    decoded_cause = encoders_cause["Original cause material"].inverse_transform([int(pred_cause)])[0]

    # 🧠 부상유형 예측 (컬럼 순서 및 이름 강제 맞춤)
    expected_cols = injury_model.feature_names_
    x_input_injury = x_input.reindex(columns=expected_cols)
    pred_injury = injury_model.predict(x_input_injury)[0]
    decoded_injury = encoders_injury["Injury type"].inverse_transform([int(pred_injury)])[0]

    # ☠️ 위험도 계산
    cause_risk = risk_data['cause'].get(decoded_cause, 0)
    injury_risk = risk_data['injury'].get(decoded_injury, 0)
    final_risk = (cause_risk + injury_risk) / 2

    # ✅ 결과 출력
    st.success("예측 결과")
    st.write(f"**예측 기인물:** {decoded_cause}")
    st.write(f"**예측 부상유형:** {decoded_injury}")
    st.write(f"**기인물 위험도:** {cause_risk * 100:.2f}%")
    st.write(f"**부상유형 위험도:** {injury_risk * 100:.2f}%")
    st.markdown(f"### 💀 최종 사망 위험도: **{final_risk * 100:.2f}%**")
