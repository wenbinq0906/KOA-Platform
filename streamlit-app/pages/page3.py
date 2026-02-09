# -*- coding: utf-8 -*-
"""

@author: QWB
"""

import streamlit as st
import pandas as pd

#页面配置
st.set_page_config(layout="wide")


#path
from pathlib import Path
base_dir = Path(__file__).resolve().parents[1]  # streamlit-app
matrix_dir = base_dir / "data_example" / "Matrix.csv"
gene_function_dir = base_dir / "data_example" / "Gene_function.xlsx"

#read reference
matrix_df = pd.read_csv(matrix_dir, index_col=0)
matrix_df_filtered = matrix_df.drop(columns=['Sex','Age'])
name_mapping = {
    #blood count
    'Basophill count': '嗜碱性粒细胞数',
    'Basophill percentage': '嗜碱性粒细胞比率',
    'Eosinophill count': '嗜酸性粒细胞数',
    'Eosinophill percentage': '嗜酸性粒细胞比率',
    'Haematocrit percentage': '红细胞压积',
    'Haemoglobin concentration': '血红蛋白',
    'High light scatter reticulocyte count': '高荧光网红计数',
    'High light scatter reticulocyte percentage': '高荧光网红比值',
    'Immature reticulocyte fraction': '未成熟网红比值',
    'Lymphocyte count': '淋巴细胞数',
    'Lymphocyte percentage': '淋巴细胞比率',
    'Mean corpuscular haemoglobin': '平均血红蛋白量',
    'Mean corpuscular haemoglobin concentration': '平均血红蛋白浓度',
    'Mean corpuscular volume': '红细胞平均体积',
    'Mean platelet volume': '平均血小板体积',
    'Mean reticulocyte volume':'网织红细胞平均体积',
    'Mean sphered cell volume':'球形细胞平均体积',
    'Monocyte count':'单核细胞数',
    'Monocyte percentage': '单核细胞比率',
    'Neutrophill count': '中性粒细胞数',
    'Neutrophill percentage': '中性粒细胞比率',
    'Nucleated red blood cell count': '有核红细胞数',
    'Nucleated red blood cell percentage': '有核红细胞比例',
    'Platelet count': '血小板',
    'Platelet crit': '血小板压积',
    'Platelet distribution width': '血小板分布宽度',
    'Erythrocyte count': '红细胞',
    'Erythrocyte distribution width': '红细胞分布宽度CV',
    'Reticulocyte count': '网织红细胞计数',
    'Reticulocyte percentage': '网织红细胞百分比',
    'Leukocyte count': '白细胞',

    #blood biochemistry
    "Alanine aminotransferase": "丙氨酸氨基转移酶",
    "Albumin": "白蛋白",
    "Alkaline phosphatase": "碱性磷酸酶",
    "Apolipoprotein A": "载脂蛋白A",
    "Apolipoprotein B": "载脂蛋白B",
    "Aspartate aminotransferase": "天门冬氨酸氨基转移酶",
    "C-reactive protein": "C-反应蛋白", 
    "Calcium": "钙",
    "Cholesterol": "总胆固醇",
    "Creatinine": "肌酐",
    "Cystatin C": "胱抑素C",
    "Direct bilirubin": "直接胆红素",
    "Glutamyltransferase": "γ-谷氨酰基转移酶",  
    "Glucose": "葡萄糖测定",
    "Glycated haemoglobin (HbA1c)": "糖化血红蛋白",  
    "HDL cholesterol": "高密度脂蛋白胆固醇",  
    "IGF-1": "胰岛素样生长因子-1",  
    "LDL direct": "低密度脂蛋白胆固醇",  
    "Lipoprotein A": "脂蛋白(a)",  
    "Oestradiol": "雌二醇",
    "Phosphate": "磷",
    "Rheumatoid factor": "类风湿因子",
    "SHBG": "性激素结合球蛋白",  
    "Testosterone": "睾酮",
    "Total bilirubin": "总胆红素",
    "Total protein": "总蛋白",
    "Triglycerides": "甘油三脂",
    "Urate": "尿酸",
    "Urea": "尿素",
    "Vitamin D": "维生素D"
}
matrix_df_renamed = matrix_df_filtered.rename(columns=name_mapping)

#read gene function reference
gene_function_df= pd.read_csv(gene_function_dir, index_col=0, encoding="gbk")

#title
st.title("血液指标相关基因查询")
st.divider()

st.info("本系统记录了部分血液指标与部分膝骨关节炎软骨核心调控基因的关联，可供科学研究和病理分析使用。")
blood_indice = st.selectbox("请选择一个血液指标", options=matrix_df_renamed.columns.tolist())

if blood_indice:
    series = matrix_df_renamed[blood_indice]
    # 只保留系数 ≠ 0
    related_genes = series[series != 0]
    
    if related_genes.empty:
        st.warning("该指标未检测到相关基因")
    else:
        result_df = (related_genes.reset_index().
                     drop(columns=blood_indice).
                     rename(columns={"index": "关联基因"})
                     )
        merged_result_df = pd.merge(result_df, gene_function_df,left_index=True)
        st.dataframe(merged_result_df,hide_index=True,width='content')
        