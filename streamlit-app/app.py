# -*- coding: utf-8 -*-
"""
@author: QWB
"""

import streamlit as st



#path
from pathlib import Path
base_dir = Path(__file__).resolve().parent # streamlit-app

LOGO_path = base_dir / "art_materials" / "BIGC_logo.png"
LOGO_bigc = base_dir / "art_materials" / "BIGC.png"
LOGO_xjtu = base_dir / "art_materials" / "XJTU.png"


col1, col2, col3 = st.columns([1.4,2,2])
with col1:
    st.image(LOGO_xjtu)
with col3:
    st.image(LOGO_bigc)


#page definition
home = st.Page("pages/home.py", title="首页")
page1 = st.Page("pages/page1.py", title="单个病例输入")
page2 = st.Page("pages/page2.py", title="批量病例输入")
page3 = st.Page("pages/page3.py", title="血液指标——基因")
pg = st.navigation([home, page1, page2, page3 ],position="hidden")
pg.run()

with st.sidebar:
    st.image(LOGO_path)
    st.divider()
    st.page_link("pages/home.py", label="首页")
    st.page_link("pages/page1.py", label="单个样本输入")
    st.page_link("pages/page2.py", label="批量样本输入")
    st.page_link("pages/page3.py", label="血液指标——基因")


