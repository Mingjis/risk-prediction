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

# ✅ Google Drive 파일 ID 및 경로
injury_model_id = "1mYGG3lZQDJwsaqSXgvC8lB0BHJmqHSap"  # 부상유형 모델
injury_model_path = "injury_type_model.cbm"

# 🎯 모델 및 리소스 로딩
@st.cache_resource
def load_models():
    # 📥 Google Drive에서 부상유형 모델 다운로드
    download_model_from_drive(injury_model_id, injury_model_path)

    # 📦 CatBoost 모델 로딩
    cause_model = CatBoostClassifier()
    cause_model.load_model("cause_material_model.cbm")

    injury_model = CatBoostClassifier()
    injury_model.load_model(injury_model_path)

    # 📦 위험도 데이터
    with open("risk_model_average.pkl", "rb") as f:
        risk_data = pickle.load(f)

    # 📦 인코더
    with open("encoders_cause.pkl", "rb") as f:
        encoders_cause = pickle.load(f)

    with open("encoders_injury.pkl", "rb") as f:
        encoders_injury = pickle.load(f)

    return cause_model, injury_model, risk_data, encoders_cause, encoders_injury

# 🔧 로딩
cause_model, injury_model, risk_data, encoders_cause, encoders_injury = load_models()

# 🎛️ 사용자 입력
st.title("🏗️ 건설 재해 사망 위험도 예측기")
st.markdown("**아래 정보를 입력하면 사고유형, 기인물, 위험도를 예측해줍니다**")

project_scale = st.selectbox("Project scale", encoders_cause['Project scale'].classes_)
facility_type = st.selectbox("Facility type", encoders_cause['Facility type'].classes_)
work_type = st.selectbox("Work type", encoders_cause['Work type'].classes_)

if st.button("위험도 예측"):
    # 각각 인코딩
    x_input_cause = pd.DataFrame([[
        encoders_cause['Project scale'].transform([project_scale])[0],
        encoders_cause['Facility type'].transform([facility_type])[0],
        encoders_cause['Work type'].transform([work_type])[0]
    ]], columns=["Project scale", "Facility type", "Work type"])

    x_input_injury = pd.DataFrame([[
        encoders_injury['Project scale'].transform([project_scale])[0],
        encoders_injury['Facility type'].transform([facility_type])[0],
        encoders_injury['Work type'].transform([work_type])[0]
    ]], columns=["Project scale", "Facility type", "Work type"])

    st.write("입력 데이터 확인 (기인물용)", x_input_cause)
    st.write("입력 데이터 확인 (부상유형용)", x_input_injury)

    # 예측
    pred_cause = cause_model.predict(x_input_cause)[0]
    pred_injury = injury_model.predict(x_input_injury)[0]

    # 역변환
    decoded_cause = encoders_cause["Original cause material"].inverse_transform([int(pred_cause)])[0]
    decoded_injury = encoders_injury["Injury type"].inverse_transform([int(pred_injury)])[0]

    # 위험도 계산
    cause_risk = risk_data['cause'].get(decoded_cause, 0)
    injury_risk = risk_data['injury'].get(decoded_injury, 0)
    final_risk = (cause_risk + injury_risk) / 2

    # 출력
    st.success("✅ 예측 결과")
    st.write(f"**예측 기인물:** {decoded_cause}")
    st.write(f"**예측 부상유형:** {decoded_injury}")
    st.write(f"**기인물 위험도:** {cause_risk * 100:.2f}%")
    st.write(f"**부상유형 위험도:** {injury_risk * 100:.2f}%")
    st.markdown(f"### 💀 최종 사망 위험도: **{final_risk * 100:.2f}%**")
