# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap

import streamlit as st

def show_navigation():
    with st.container(border=True):
        col1,col2,col3,col4,col5=st.columns(5)
        col1.page_link("Home.py", label="Home", icon="🏠")
        col2.page_link("pages/0_upload_pdf.py", label="Upload PDF", icon="1️⃣")
        col3.page_link("pages/1_chat_with_AI.py", label="Chat", icon="2️⃣")
        col4.page_link("pages/2_retreival_augmented_chat.py", label="RAG", icon="🌎")
        col5.page_link("pages/3_unstructured.py", label="Unstructured")

def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))
