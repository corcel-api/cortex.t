import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
from cortext import CONFIG
from openai import OpenAI

# Configuration
API_BASE_URL = f"http://{CONFIG.organic.host}:{CONFIG.organic.port}"

OPENAI_CLIENT = OpenAI(api_key="", base_url=API_BASE_URL + "/api/v1")


def check_admin_key(admin_key):
    """Validate admin key by trying to fetch all API keys"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/keys",
            headers={"Authorization": f"Bearer {admin_key}"},
        )
        return response.status_code == 200
    except:
        return False


# Session state initialization
if "admin_key" not in st.session_state:
    st.session_state.admin_key = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# Admin Authentication
with st.sidebar:
    st.title("Admin Authentication")
    admin_key = st.text_input("Enter Admin API Key", type="password")
    if st.button("Login"):
        if check_admin_key(admin_key):
            st.session_state.admin_key = admin_key
            st.session_state.is_admin = True
            st.success("Successfully authenticated!")
        else:
            st.error("Invalid admin key")

    if st.session_state.is_admin:
        st.success("Logged in as Admin")

# Main content
st.title("Cortex Admin Dashboard")
if not st.session_state.is_admin:
    st.warning("Please login with admin key to access the dashboard")
else:
    tab1, tab2 = st.tabs(["Test Completions", "API Key Management"])

    # Test Completions Tab
    with tab1:
        st.header("Test Chat Completions")

        model = st.selectbox(
            "Select Model",
            ["gpt-4o", "claude-3-5-sonnet-20241022"],  # Add more models as needed
        )

        api_key = st.text_input("Enter API Key", type="password")

        prompt = st.text_input("Enter Prompt")

        submit = st.button("Submit")

        if submit:
            if not api_key:
                st.info("Please enter an API key to continue.")
                st.stop()

            messages = [{"role": "user", "content": prompt}]

            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.1,
                }

                with st.spinner("Waiting for response..."):
                    OPENAI_CLIENT.api_key = api_key
                    response = OPENAI_CLIENT.chat.completions.create(**payload)

                    response_container = st.empty()

                    def stream_data():
                        for chunk in response:
                            yield chunk.choices[0].delta.content

                    st.write_stream(stream_data)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # API Key Management Tab
    with tab2:
        st.header("API Key Management")

        # Create new API key
        with st.expander("Create New API Key"):
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.text_input("User ID")
                initial_credits = st.number_input(
                    "Initial Credits", min_value=0.0, value=100.0
                )
            with col2:
                monthly_reset = st.checkbox("Monthly Reset", value=True)

            if st.button("Create API Key"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/keys",
                        params={
                            "user_id": user_id,
                            "initial_credits": initial_credits,
                            "monthly_reset": monthly_reset,
                        },
                        headers={
                            "Authorization": f"Bearer {st.session_state.admin_key}"
                        },
                    )
                    if response.status_code == 200:
                        st.success("API Key created successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # List all API keys
        st.subheader("Existing API Keys")
        if st.button("Refresh API Keys"):
            try:
                response = requests.get(
                    f"{API_BASE_URL}/api/v1/keys",
                    headers={"Authorization": f"Bearer {st.session_state.admin_key}"},
                )
                if response.status_code == 200:
                    keys_data = response.json()
                    df = pd.DataFrame(keys_data)
                    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    df["credit_reset_date"] = pd.to_datetime(
                        df["credit_reset_date"]
                    ).dt.strftime("%Y-%m-%d %H:%M:%S")
                    st.dataframe(df)

                    # Add actions for each key
                    for key in keys_data:
                        with st.expander(f"Actions for key: {key['key'][:10]}..."):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                new_credits = st.number_input(
                                    "Add Credits",
                                    min_value=0.0,
                                    value=0.0,
                                    key=f"credits_{key['key']}",
                                )
                                if st.button("Add", key=f"add_{key['key']}"):
                                    response = requests.post(
                                        f"{API_BASE_URL}/api/v1/keys/{key['key']}/add-credits",
                                        params={"amount": new_credits},
                                        headers={
                                            "Authorization": f"Bearer {st.session_state.admin_key}"
                                        },
                                    )
                                    if response.status_code == 200:
                                        st.success("Credits added successfully!")

                            with col2:
                                new_status = st.checkbox(
                                    "Active",
                                    value=key["is_active"],
                                    key=f"status_{key['key']}",
                                )
                                if st.button(
                                    "Update Status", key=f"update_{key['key']}"
                                ):
                                    response = requests.patch(
                                        f"{API_BASE_URL}/api/v1/keys/{key['key']}/status",
                                        json={"is_active": new_status},
                                        headers={
                                            "Authorization": f"Bearer {st.session_state.admin_key}"
                                        },
                                    )
                                    if response.status_code == 200:
                                        st.success("Status updated successfully!")

                            with col3:
                                if st.button("Delete", key=f"delete_{key['key']}"):
                                    if st.warning("Are you sure?"):
                                        response = requests.delete(
                                            f"{API_BASE_URL}/api/v1/keys/{key['key']}",
                                            headers={
                                                "Authorization": f"Bearer {st.session_state.admin_key}"
                                            },
                                        )
                                        if response.status_code == 200:
                                            st.success("Key deleted successfully!")

                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
