import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier

# 🎯 모델 및 리소스 로딩
@st.cache_resource
def load_models():
    cause_model = CatBoostClassifier()
    cause_model.load_model("cause_model.cbm")

    injury_model = CatBoostClassifier()
    injury_model.load_model("injury_type_model.cbm")

    with open("risk_dict.pkl", "rb") as f:
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
facility_type = st.selectbox("Facility type", encoders['Facility type'].classes_)
work_type = st.selectbox("Work type", encoders['Work type'].classes_)

if st.button("위험도 예측"):
    # ⛓️ 인코딩
    x_input = pd.DataFrame([[
        encoders['Project scale'].transform([project_scale])[0],
        encoders['Facility type'].transform([facility_type])[0],
        encoders['Work type'].transform([work_type])[0]
    ]], columns=["Project scale", "Facility type", "Work type"])

    # 🔮 예측
    pred_cause = cause_model.predict(x_input)[0]
    pred_injury = injury_model.predict(x_input)[0]

    # 🧠 역변환
    decoded_cause = encoders["Original cause material"].inverse_transform([int(pred_cause)])[0]
    decoded_injury = encoders["Injury type"].inverse_transform([int(pred_injury)])[0]

    # ☠️ 위험도 계산
    cause_risk = risk_data['cause'].get(decoded_cause, 0)
    injury_risk = risk_data['injury'].get(decoded_injury, 0)
    final_risk = max(cause_risk, injury_risk)

    # 🎯 출력
    st.success("예측 결과")
    st.write(f"**예측 기인물:** {decoded_cause}")
    st.write(f"**예측 부상유형:** {decoded_injury}")
    st.write(f"**기인물 위험도:** {cause_risk * 100:.2f}%")
    st.write(f"**부상유형 위험도:** {injury_risk * 100:.2f}%")
    st.markdown(f"### 💀 최종 사망 위험도: **{final_risk * 100:.2f}%**")
