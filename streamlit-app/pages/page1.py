# -*- coding: utf-8 -*-
"""

@author: QWB
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

#页面配置
st.set_page_config(layout="wide")

#path
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # streamlit-app
model_dir = base_dir / "models"
example_data_dir = base_dir / "data_example" / "data_example.csv"
font_dir = base_dir / "fonts"
font_path = font_dir / "simhei.ttf"

#标题和说明
st.title("骨关节炎风险评估（单个样本）")

st.divider()

#input data
st.header("请输入病人信息")
# patient name or id
patient_name = st.text_input("病人姓名或编号：")

col1, col2 = st.columns(2)
# patient gender
with col1:
    patient_gender = st.selectbox("病人的性别：", ["男", "女"])
# patient age
with col2:
    patient_age = st.number_input("年龄（岁）",min_value=0,max_value=120)

st.caption("  ")
st.caption("  ")

#blood indices
col1, col2, col3 = st.columns(3)
with col1:
    patient_Basophil_count = st.number_input("嗜碱性粒细胞数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Eosinophil_percentage = st.number_input("嗜酸性粒细胞比率 (%)", value=None,step=0.01,format="%.2f")
    patient_Lymphocyte_count = st.number_input("淋巴细胞数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Mean_corpuscular_haemoglobin_concentration = st.number_input("平均血红蛋白浓度 (g/L)", value=None,step=0.01,format="%.2f")
    patient_Monocyte_percentage = st.number_input("单核细胞比率 (%)", value=None,step=0.01,format="%.2f")
    patient_Platelet_crit = st.number_input("血小板压积 (%)", value=None,step=0.01,format="%.2f")
    patient_Red_blood_cell_distribution_width = st.number_input("红细胞分布宽度CV (%)", value=None,step=0.01,format="%.2f")
    patient_White_blood_cell_count = st.number_input("白细胞 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Alkaline_phosphatase = st.number_input("碱性磷酸酶 (U/L)", value=None,step=0.01,format="%.2f")
    patient_Aspartate_aminotransferase = st.number_input("天门冬氨酸氨基转移酶 (U/L)", value=None,step=0.01,format="%.2f")
    patient_Cholesterol = st.number_input("总胆固醇 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Direct_bilirubin = st.number_input("直接胆红素 (umol/L)", value=None,step=0.01,format="%.2f")
    patient_Glycated_haemoglobin = st.number_input("糖化血红蛋白 (%)", value=None,step=0.01,format="%.2f")
    patient_LDL_direct = st.number_input("低密度脂蛋白胆固醇 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Total_bilirubin = st.number_input("总胆红素 (umol/L)", value=None,step=0.01,format="%.2f")
    patient_Urate = st.number_input("尿酸 (umol/L)", value=None,step=0.01,format="%.2f")
    patient_Urea = st.number_input("尿素 (mmol/L)", value=None,step=0.01,format="%.2f")

with col2:
    patient_Basophil_percentage = st.number_input("嗜碱性粒细胞比率 (%)", value=None,step=0.01,format="%.2f")
    patient_Haematocrit_percentage = st.number_input("红细胞压积 (%)", value=None,step=0.01,format="%.2f")
    patient_High_light_scattervreticulocyte_percentage = st.number_input("高荧光网红比值 (%)", value=None,step=0.01,format="%.2f")
    patient_Lymphocyte_percentage = st.number_input("淋巴细胞比率 (%)", value=None,step=0.01,format="%.2f")
    patient_Mean_corpuscular_volume = st.number_input("红细胞平均体积 (fL)", value=None,step=0.01,format="%.2f")
    patient_Neutrophil_count = st.number_input("中性粒细胞数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Nucleated_red_blood_cell_percentage = st.number_input("有核红细胞比例 (%)", value=None,step=0.01,format="%.2f")
    patient_Platelet_distribution_width = st.number_input("血小板分布宽度 (fL)", value=None,step=0.01,format="%.2f")
    patient_Reticulocyte_count = st.number_input("网织红细胞计数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Alanine_aminotransferase = st.number_input("丙氨酸氨基转移酶 (U/L)", value=None,step=0.01,format="%.2f")
    patient_Apolipoprotein_A = st.number_input("载脂蛋白A (g/L)", value=None,step=0.01,format="%.2f")
    patient_C_reactive_protein = st.number_input("C-反应蛋白 (mg/L)", value=None,step=0.01,format="%.2f")
    patient_Creatinine = st.number_input("肌酐 (umol/L)", value=None,step=0.01,format="%.2f")
    patient_Gamma_glutamyltransferase = st.number_input("γ-谷氨酰基转移酶 (U/L)", value=None,step=0.01,format="%.2f")
    patient_HDL_cholesterol = st.number_input("高密度脂蛋白胆固醇 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Lipoprotein_A = st.number_input("脂蛋白(a) (nmol/L)", value=None,step=0.01,format="%.2f")
    patient_Total_protein = st.number_input("总蛋白 (g/L)", value=None,step=0.01,format="%.2f")
    

with col3:
    patient_Eosinophil_count = st.number_input("嗜酸性粒细胞数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Haemoglobin_concentration = st.number_input("血红蛋白 (g/L)", value=None,step=0.01,format="%.2f")
    patient_Immature_reticulocyte_fraction = st.number_input("未成熟网红比值 (%)", value=None,step=0.01,format="%.2f")
    patient_Mean_corpuscular_haemoglobin = st.number_input("平均血红蛋白量 (pg)", value=None,step=0.01,format="%.2f")
    patient_Mean_platelet_volume = st.number_input("平均血小板体积 (fL)", value=None,step=0.01,format="%.2f")
    patient_Monocyte_count = st.number_input("单核细胞数 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Neutrophil_percentage = st.number_input("中性粒细胞比率 (%)", value=None,step=0.01,format="%.2f")
    patient_Platelet_count = st.number_input("血小板 (10^9/L)", value=None,step=0.01,format="%.2f")
    patient_Red_blood_cell_count = st.number_input("红细胞 (10^12/L)", value=None,step=0.01,format="%.2f")
    patient_Reticulocyte_percentage = st.number_input("网织红细胞百分比 (%)", value=None,step=0.01,format="%.2f")
    patient_Albumin = st.number_input("白蛋白 (g/L)", value=None,step=0.01,format="%.2f")
    patient_Apolipoprotein_B = st.number_input("载脂蛋白B (g/L)", value=None,step=0.01,format="%.2f")
    patient_Calcium = st.number_input("钙 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Cystatin_C = st.number_input("胱抑素C (mg/L)", value=None,step=0.01,format="%.2f")
    patient_Glucose = st.number_input("葡萄糖测定 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Rheumatoid_factor = st.number_input("类风湿因子 (IU/mL)", value=None,step=0.01,format="%.2f")
    patient_Phosphate = st.number_input("磷 (mmol/L)", value=None,step=0.01,format="%.2f")
    patient_Triglycerides = st.number_input("甘油三脂 (mmol/L)", value=None,step=0.01,format="%.2f")

st.info("如有缺失值可直接留空，请注意各项指标的输入单位。")    
st.divider()

col1, col2, col3 = st.columns([3, 1, 1])
with col3:
    predict_button = st.button("评估风险", type="primary")

input_dict = {
    "姓名": patient_name,
    "性别": patient_gender,
    "年龄": patient_age,
    
    "嗜碱性粒细胞数": patient_Basophil_count,
    "嗜碱性粒细胞比率": patient_Basophil_percentage,
    "嗜酸性粒细胞数": patient_Eosinophil_count,
    "嗜酸性粒细胞比率": patient_Eosinophil_percentage,
    "淋巴细胞数": patient_Lymphocyte_count,
    "淋巴细胞比率": patient_Lymphocyte_percentage,
    "单核细胞数": patient_Monocyte_count,
    "单核细胞比率": patient_Monocyte_percentage,
    "中性粒细胞数": patient_Neutrophil_count,
    "中性粒细胞比率": patient_Neutrophil_percentage,
    "白细胞": patient_White_blood_cell_count,
    "红细胞": patient_Red_blood_cell_count,
    "血红蛋白": patient_Haemoglobin_concentration,
    "红细胞压积": patient_Haematocrit_percentage,
    "红细胞分布宽度CV": patient_Red_blood_cell_distribution_width,
    "平均血红蛋白量": patient_Mean_corpuscular_haemoglobin,
    "平均血红蛋白浓度": patient_Mean_corpuscular_haemoglobin_concentration,
    "红细胞平均体积": patient_Mean_corpuscular_volume,
    "血小板": patient_Platelet_count,
    "平均血小板体积": patient_Mean_platelet_volume,
    "血小板压积": patient_Platelet_crit,
    "血小板分布宽度": patient_Platelet_distribution_width,
    "网织红细胞计数": patient_Reticulocyte_count,
    "网织红细胞百分比": patient_Reticulocyte_percentage,
    "高荧光网红比值": patient_High_light_scattervreticulocyte_percentage,
    "未成熟网红比值": patient_Immature_reticulocyte_fraction,
    "糖化血红蛋白":patient_Glycated_haemoglobin,
    "有核红细胞比例":patient_Nucleated_red_blood_cell_percentage,

    "白蛋白": patient_Albumin,
    "总蛋白": patient_Total_protein,
    "钙": patient_Calcium,
    "磷": patient_Phosphate,
    "葡萄糖测定": patient_Glucose,
    "尿素": patient_Urea,
    "肌酐": patient_Creatinine,
    "胱抑素C": patient_Cystatin_C,
    "尿酸": patient_Urate,
    "丙氨酸氨基转移酶": patient_Alanine_aminotransferase,
    "天门冬氨酸氨基转移酶": patient_Aspartate_aminotransferase,
    "碱性磷酸酶": patient_Alkaline_phosphatase,
    "γ-谷氨酰基转移酶": patient_Gamma_glutamyltransferase,
    "总胆红素": patient_Total_bilirubin,
    "直接胆红素": patient_Direct_bilirubin,
    "总胆固醇": patient_Cholesterol,
    "甘油三脂": patient_Triglycerides,
    "高密度脂蛋白胆固醇": patient_HDL_cholesterol,
    "低密度脂蛋白胆固醇": patient_LDL_direct,
    "载脂蛋白A": patient_Apolipoprotein_A,
    "载脂蛋白B": patient_Apolipoprotein_B,
    "脂蛋白(a)": patient_Lipoprotein_A,
    "C-反应蛋白": patient_C_reactive_protein,
    "类风湿因子": patient_Rheumatoid_factor,
}    

#predict    
if predict_button:    
    
    st.divider()
    st.header("风险评估结果")
    
    
    #prepare data
    input_data=pd.DataFrame([input_dict])
    input_data_df = input_data.copy()
    input_data_df['性别'] = input_data_df['性别'].map({'男': 1, '女': 0}).astype(int)

    #set columns sequence
    example_df = pd.read_csv(example_data_dir,header=0)
    example_feature_order = example_df.drop(columns=['编号']).columns.tolist()
    input_data_df = input_data_df[example_feature_order]
    st.session_state["input_data_df"] = input_data_df
    X_test = input_data_df.drop(columns=['姓名'])
    
    
    #-----perform imputation on features-----
    features = [col for col in X_test.columns if col not in ['年龄','性别']]
    if len(features) !=0:
        # perform imputation
        imputer=joblib.load(model_dir / "imputer.pkl")
        X_test_imputed = imputer.transform(X_test[features])
        # merge imputed data
        X_test[features] = X_test_imputed
        
    st.session_state["X_test_1"] = X_test
    
    #-----features scaling-----
    scaler = joblib.load(model_dir / "scaler.pkl")
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled =pd.DataFrame(X_test_scaled,columns=X_test.columns)
    st.session_state["X_test_scaled_1"] = X_test_scaled        

    #predict
    ensemble_model=joblib.load(model_dir / "ensemble_model.pkl")
    risk_prob = ensemble_model.predict_proba(X_test_scaled)[:,1]
    
    #save results
    result_df = input_data.copy()
    result_df["风险概率"] = risk_prob
    st.session_state["result_df_1"] = result_df
        
if "result_df_1" in st.session_state:
   st.success("风险评估完成，详情请见“风险概率”列")

   #显示结果表
   show_result_df=st.session_state["result_df_1"].copy()
   show_result_df["风险概率"]=show_result_df["风险概率"].apply(lambda x: f"{x:.2f}")
   cols = list(show_result_df.columns)
   cols.remove("风险概率")
   cols.insert(1, "风险概率")   # 第 3 列（0-based）
   from matplotlib.colors import LinearSegmentedColormap
   single_color_cmap = LinearSegmentedColormap.from_list(
    "same_color", ["#FFD54F", "#FFD54F"])
   show_result_df = show_result_df[cols].style.background_gradient(subset=["风险概率"],cmap=single_color_cmap)
   st.dataframe(show_result_df,hide_index=True,width="content")

   
   #shap
   st.divider()
   st.header("解释分析")
   st.info("本系统会对缺失的血液指标自动填充，请谨慎解读解释结果")    
   explainer = joblib.load(model_dir / "shap_explainer.pkl")
   shap_values = explainer.shap_values(st.session_state["X_test_scaled_1"])
   
   #set font
   from matplotlib import font_manager, rcParams
   #手动注册字体
   font_manager.fontManager.addfont(font_path)
   font_prop = font_manager.FontProperties(fname=font_path)
   font_name = font_prop.get_name()
   # 设置为全局默认字体
   rcParams['font.family'] = font_name
   rcParams['axes.unicode_minus'] = False

   #plot
   force_plot = shap.plots.force(base_value=explainer.expected_value,
                   shap_values=shap_values[0],
                   features=np.round(st.session_state["X_test_1"].iloc[0,:],2),
                   feature_names=st.session_state["X_test_scaled_1"].columns,
                   plot_cmap=["#ff4777", "#00e079"])
   import streamlit.components.v1 as components
   html_str = shap.getjs() + force_plot.html()
   components.html(html_str ,width=1000)
   
   st.warning("若图片显示不全，可直接下载查看")
   col1, col2, col3 = st.columns([3, 1, 1])
   with col3:
       st.download_button(label='下载图片',data=html_str,
                      file_name="shap_force_plot.html",
                      type="primary")

    
    
    