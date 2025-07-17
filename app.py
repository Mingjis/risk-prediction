import streamlit as st
import pandas as pd
import pickle
import os
import requests
from catboost import CatBoostClassifier

# 🎯 Google Drive에서 모델 다운로드 함수
def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"📥 Downloading model from Google Drive: {output_path}")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(response.content)

# 🎯 모델 및 리소스 로딩
@st.cache_resource
def load_models():
    # ▶️ Google Drive 모델 파일 ID (수정해서 넣기)
    injury_model_id = "1mYGG3lZQDJwsaqSXgvC8lB0BHJmqHSap"
    injury_model_path = "injury_type_model.cbm"
    download_model_from_drive(injury_type_model_id, injury_type_model_path)

    # ▶️ 기인물 예측 모델은 용량 작으면 그냥 포함
    cause_model = CatBoostClassifier()
    cause_model.load_model("cause_material_model.cbm")

    injury_model = CatBoostClassifier()
    injury_model.load_model(injury_tpye_model_path)

    with open("risk_model_average.pkl", "rb") as f:
        risk_data = pickle.load(f)

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return cause_model, injury_model, risk_data, encoders

# 📦 모델 불러오기
cause_model, injury_model, risk_data, encoders = load_models()

# 🎛️ 사용자 입력
st.title("건설 재해 사망 위험도 예측기")
st.markdown("**아래 정보를 입력하면 사고유형, 기인물, 위험도를 예측해줍니다**")

project_scale = st.selectbox("Project scale", encoders['Project scale'].classes_)
facility_type = st.selectbox("Facility type", encoders['Facility Type'].classes_)
work_type = st.selectbox("Work type", encoders['Work type'].classes_)

if st.button("위험도 예측"):
    x_input = pd.DataFrame([[  # ⛓️ 인코딩
        encoders['Project scale'].transform([project_scale])[0],
        encoders['Facility Type'].transform([facility_type])[0],
        encoders['Work type'].transform([work_type])[0]
    ]], columns=["Project scale", "Facility Type", "Work type"])

    # 🔮 예측
    pred_cause = cause_model.predict(x_input)[0]
    pred_injury = injury_model.predict(x_input)[0]

    # 🧠 역변환
    decoded_cause = encoders["Original cause material"].inverse_transform([int(pred_cause)])[0]
    decoded_injury = encoders["Injury type"].inverse_transform([int(pred_injury)])[0]

    # ☠️ 위험도 계산
    cause_risk = risk_data['cause'].get(decoded_cause, 0)
    injury_risk = risk_data['injury'].get(decoded_injury, 0)
    final_risk = (cause_risk + injury_risk) / 2

    # 🎯 출력
    st.success("예측 결과")
    st.write(f"**예측 기인물:** {decoded_cause}")
    st.write(f"**예측 부상유형:** {decoded_injury}")
    st.write(f"**기인물 위험도:** {cause_risk * 100:.2f}%")
    st.write(f"**부상유형 위험도:** {injury_risk * 100:.2f}%")
    st.markdown(f"### 💀 최종 사망 위험도: **{final_risk * 100:.2f}%**")
