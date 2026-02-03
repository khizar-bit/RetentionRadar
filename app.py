import streamlit as st
import pandas as pd
import requests
import json
import google.generativeai as genai
import time
from urllib.parse import quote

# Page Config
st.set_page_config(page_title="Retention Radar", page_icon="📡", layout="wide")

# Title and Sidebar
st.title("📡 Retention Radar")
st.sidebar.header("Configuration")

# Credentials Inputs
SPUR_API_KEY = st.sidebar.text_input("Spur API Bearer Token", type="password", help="Found in Spur Settings > API")
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password", help="Get it from aistudio.google.com")

# Model & AI Configuration
with st.sidebar.expander("🤖 AI Settings", expanded=True):
    # Model Selection
    MODEL_NAME = st.text_input("Model Name", value="gemini-1.5-flash", help="e.g. gemini-1.5-flash, gemini-pro")
    
    if GEMINI_API_KEY and st.button("Check Available Models"):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.success(f"Found {len(models)} models:")
            st.code("\n".join(models))
        except Exception as e:
            st.error(f"Error listing models: {str(e)}")
            
    # Prompt Template
    st.markdown("### 📝 Prompt Template")
    DEFAULT_PROMPT = """Write a short, friendly, and persuasive WhatsApp recovery message (maximum 20 words) for a customer named '{name}' who hasn't logged into our app for {days} days.
Do not include hashtags. Do not include 'Subject:'. Just the message body."""
    
    PROMPT_TEMPLATE = st.text_area(
        "Edit the instructions for the AI:", 
        value=DEFAULT_PROMPT,
        height=200,
        help="Use {name} and {days} as placeholders. They will be replaced automatically."
    )

WHATSAPP_NUMBER_SOURCE = "whatsapp" # Default channel

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Metabase/Posthog CSV", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # 2. Pre-process Data
    if 'Last_Login_Days' in df.columns:
        # Filter for At Risk Users (> 7 days inactive)
        risk_df = df[df['Last_Login_Days'] > 7].copy()
        
        # Initialize 'Select' column for checkboxes
        if 'Select' not in risk_df.columns:
            risk_df.insert(0, 'Select', False)
        
        # 3. The AI Drafter
        st.write("---")
        st.subheader(f"🚨 At Risk Users ({len(risk_df)})")
        
        # AI Draft Button at the top
        col1, col2 = st.columns([1, 4])
        with col1:
            generate_ai = st.button("✨ Generate AI Drafts")
        
        # Logic to generate drafts if button clicked
        if generate_ai:
             if not GEMINI_API_KEY:
                 st.warning("⚠️ Please enter your Gemini API Key in the sidebar first.")
             else:
                 # Configure Gemini
                 genai.configure(api_key=GEMINI_API_KEY)
                 
                 try:
                     model = genai.GenerativeModel(MODEL_NAME)
                     
                     with st.spinner(f"Gemini ({MODEL_NAME}) is drafting using your custom prompt..."):
                         def draft_with_ai(row):
                             days = int(row['Last_Login_Days']) if pd.notnull(row['Last_Login_Days']) else 0
                             name = row.get('Name', 'there')
                             if pd.isna(name): name = 'there'
                             
                             # Use the Custom PROMPT_TEMPLATE
                             try:
                                 prompt = PROMPT_TEMPLATE.format(name=name, days=days)
                             except KeyError as e:
                                 return f"Error: Prompt template has invalid placeholder {e}"
                                 
                             try:
                                 response = model.generate_content(prompt)
                                 return response.text.strip()
                             except Exception as e:
                                 return f"Error: {e}"
                                 
                         # Apply AI to dataframe
                         risk_df['Draft_Message'] = risk_df.apply(draft_with_ai, axis=1)
                         st.success(f"Drafts generated!")
                         
                 except Exception as e:
                     st.error(f"Failed to initialize model '{MODEL_NAME}': {e}")
        
        # If no drafts yet, provide fallback so column exists
        if 'Draft_Message' not in risk_df.columns:
             def draft_standard(row):
                days = int(row['Last_Login_Days']) if pd.notnull(row['Last_Login_Days']) else 0
                name = row.get('Name', 'there')
                if pd.isna(name): name = 'there'
                return f"Hey {name}, noticed it's been {days} days since your last login. We miss you!"
             risk_df['Draft_Message'] = risk_df.apply(draft_standard, axis=1)

        # Show interactive table with Checkboxes
        edited_df = st.data_editor(
            risk_df,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Send?",
                    help="Select users to send messages to",
                    default=False,
                ),
                "Draft_Message": st.column_config.TextColumn(
                    "Message Content",
                    width="large"
                )
            },
            disabled=["Name", "Phone", "Last_Login_Days"], # Only let them edit Select and Message
            hide_index=True,
            num_rows="dynamic",
            key="editor"
        )
        
        # Determine how many selected
        selected_rows = edited_df[edited_df['Select'] == True]
        count_selected = len(selected_rows)

        # 4. The Sending Logic
        st.write(f"**Selected Users:** {count_selected}")
        
        if st.button(f"Send to {count_selected} Selected Users", type="primary", disabled=(count_selected == 0)):
            if not SPUR_API_KEY:
                st.error("Please enter your Spur API Key in the sidebar first!")
            else:
                progress_bar = st.progress(0)
                success_count = 0
                
                # Iterate ONLY through selected rows
                for i, (index, row) in enumerate(selected_rows.iterrows()):
                    phone = str(row['Phone']).replace("+", "").strip() 
                    if phone.endswith('.0'): phone = phone[:-2]
                        
                    message_text = row['Draft_Message']
                    
                    # 1. Encode message
                    encoded_text = quote(message_text)
                    image_url = f"https://api.spurnow.com/screenshot/text-message?text={encoded_text}"
                    
                    # 2. Construct Payload
                    payload = {
                        "channel": "whatsapp",
                        "to": phone,
                        "content": {
                            "type": "template",
                            "template": {
                                "name": "support_ticket_update_spur",
                                "language": {"code": "en"},
                                "components": [
                                    {
                                        "type": "header",
                                        "parameters": [
                                            {
                                                "type": "IMAGE",
                                                "image": {"link": image_url}
                                            }
                                        ]
                                    },
                                    {
                                        "type": "body",
                                        "parameters": [
                                            {"text": " ", "type": "text"}
                                        ]
                                    },
                                    {
                                        "type": "button",
                                        "index": 0,
                                        "sub_type": "QUICK_REPLY",
                                        "parameters": [
                                            {"type": "payload", "payload": "recovery_flow"}
                                        ]
                                    }
                                ]
                            }
                        }
                    }

                    # 3. Send Request
                    headers = {
                        'Authorization': f'Bearer {SPUR_API_KEY}',
                        'Content-Type': 'application/json'
                    }
                    
                    try:
                        response = requests.post(
                            'https://api.spurnow.com/send-message', 
                            headers=headers, 
                            data=json.dumps(payload)
                        )
                        
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            st.error(f"Failed for {phone}: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error sending to {phone}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / count_selected)

                st.success(f"Done! Sent {success_count} messages.")

    else:
        st.warning("CSV must contain 'Last_Login_Days' column to detect churn.")
