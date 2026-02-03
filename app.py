import streamlit as st
import pandas as pd
import requests
import json
import google.generativeai as genai
import time
from urllib.parse import quote
import os
import extra_streamlit_components as stx
import datetime

# Page Config
st.set_page_config(page_title="Retention Radar", page_icon="📡", layout="wide")

# Title and Sidebar
# Title and Sidebar
st.title("📡 Retention Radar")

# --- Authentication ---
def check_password():
    """Returns `True` if the user had the correct password."""
    
    if "APP_PASSWORD" not in st.secrets:
        st.warning("⚠️ ADD 'APP_PASSWORD' to .streamlit/secrets.toml")
        return False

    # Initialize Cookie Manager
    cookie_manager = stx.CookieManager()

    # Wait for cookies to load
    if "auth_token" not in st.session_state:
        # Check if cookie exists
        cookie_val = cookie_manager.get(cookie="auth_token")
        if cookie_val == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            return True
        st.session_state["auth_token"] = cookie_val

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            # Set cookie (expires in 30 days)
            cookie_manager.set("auth_token", st.secrets["APP_PASSWORD"], expires_at=datetime.datetime.now() + datetime.timedelta(days=30))
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()
    
st.sidebar.header("Configuration")

# Credentials Inputs
# Helper function to handle secrets vs manual input
def get_secret(key_name, label, help_text):
    # Check if key exists in secrets (case-insensitive try)
    secret_val = st.secrets.get(key_name)
    if not secret_val:
        # Try finding it in other sections or lowercase
        for k, v in st.secrets.items():
            if k.lower() == key_name.lower():
                secret_val = v
                break
    
    # Check validity
    if secret_val and "YOUR_" not in str(secret_val):
        st.sidebar.text_input(label, value="Securely Configured", disabled=True, type="password")
        return secret_val
    else:
        return st.sidebar.text_input(label, type="password", help=help_text)

SPUR_API_KEY = get_secret("SPUR_API_KEY", "Spur API Bearer Token", "Found in Spur Settings > API")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY", "Gemini API Key", "Get it from aistudio.google.com")
SLACK_WEBHOOK_URL = get_secret("SLACK_WEBHOOK_URL", "Slack Webhook URL", "Optional: For internal alerts")

# Model & AI Configuration
with st.sidebar.expander("🤖 AI Settings", expanded=True):
    # Hardcoded Model
    MODEL_NAME = "gemini-3-flash-preview"
    st.info(f"Using Model: `{MODEL_NAME}`")
            
    # Prompt Configuration with Persistence
    st.markdown("### 📝 Prompt Template")
    PROMPT_FILE = "prompt_config.json"

    def load_prompt():
        default_prompt = """You are Khizar, a Customer Success Exec at Spur (WhatsApp Automation SaaS).

CONTEXT - WHAT SPUR CAN DO:
[WhatsApp Broadcasts, Abandoned Checkout Recovery, Instagram Comment-to-DM Automation, AI Support Chatbots, Shopify Product Linking, Click-to-WhatsApp Ad Flows]

Analyze this specific user data:
{user_context}

YOUR THOUGHT PROCESS:
1. **Check Data Quality:** Do you have usage stats (like last_login, ROAS, broadcast_count)?
2. **IF DATA EXISTS:** Find the problem and pitch a solution from the Context list.
   - Example: High DMs but no automation? -> Pitch "AI Support Chatbots".
   - Example: Inactive > 7 days? -> Pitch "Ready-to-go Broadcast Templates".
   - Example: High Sales but no recovery? -> Pitch "Abandoned Checkout Flows".
3. **IF DATA IS MISSING (Only Name/Phone):** Use the "EGO PLAY".
   - Tell them they popped up on your internal "High Potential Brands" list.
   - Say you are personally checking in to help them scale/remove blockers.

TASK:
Write a text message (max 25 words).

TONE:
- Thumb-typing. Casual. Use "u" for "you" if it fits.
- No "Dear Sir". Start with "Hey [Name]" or just the message.
- Make it sound proactive, not robotic.

OUTPUT:
Just the message text."""
        if os.path.exists(PROMPT_FILE):
            try:
                with open(PROMPT_FILE, "r") as f:
                    return json.load(f).get("prompt", default_prompt)
            except:
                return default_prompt
        return default_prompt

    def save_prompt():
        # Callback to save prompt when changed
        new_prompt = st.session_state["prompt_input"]
        with open(PROMPT_FILE, "w") as f:
            json.dump({"prompt": new_prompt}, f)
        st.toast("Prompt saved!")

    def reset_prompt():
        if os.path.exists(PROMPT_FILE):
            os.remove(PROMPT_FILE)
        st.session_state["reset_prompt_requested"] = True
        st.toast("Prompt reset to default!")

    # Initialize session state for prompt if not exists
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = load_prompt()

    if st.session_state.get("reset_prompt_requested"):
        st.session_state["prompt_input"] = load_prompt()
        st.session_state["reset_prompt_requested"] = False

    PROMPT_TEMPLATE = st.text_area(
        "Edit the instructions for the AI:", 
        value=st.session_state["prompt_input"],
        height=200,
        help="Use {user_context} as the placeholder for the entire row of data.",
        key="prompt_input",
        on_change=save_prompt
    )

    st.button("🔄 Reset to Default Prompt", on_click=reset_prompt)

WHATSAPP_NUMBER_SOURCE = "whatsapp" # Default channel

# 1. File Uploader
uploaded_file = st.file_uploader("Upload Metabase/Posthog CSV", type=["csv"])

if uploaded_file:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # 2. Column Mapping
    st.subheader("📝 Map Your Columns")
    col1, col2, col3 = st.columns(3)
    
    all_cols = list(df.columns)
    
    # Helpers for defaults
    def get_default(options, keywords):
        for opt in options:
            if any(k in opt.lower() for k in keywords):
                return opt
        return options[0]

    with col1:
        name_col = st.selectbox("Name Column (Required)", all_cols, index=all_cols.index(get_default(all_cols, ["name", "customer"])))
    with col2:
        phone_col = st.selectbox("Phone Column (Required)", all_cols, index=all_cols.index(get_default(all_cols, ["phone", "mobile", "contact"])))
    with col3:
        # Optional Last Login
        login_opts = ["None"] + all_cols
        default_login = "None"
        for opt in all_cols:
            if any(k in opt.lower() for k in ["login", "days", "inactive"]):
                default_login = opt
                break
        login_col = st.selectbox("Days Inactive (Optional)", login_opts, index=login_opts.index(default_login))

    # Rename Columns
    rename_map = {name_col: 'Name', phone_col: 'Phone'}
    if login_col != "None":
        rename_map[login_col] = 'Last_Login_Days'
    
    # Avoid collisions: if target 'Name' exists but isn't being renamed, it will duplicate.
    # We rename the existing one out of the way.
    for source, target in rename_map.items():
        if target in df.columns and source != target:
            # If the target column name exists and is NOT being renamed itself (not in keys)
            if target not in rename_map:
                new_col = f"{target}_original"
                counter = 1
                while new_col in df.columns:
                    new_col = f"{target}_original_{counter}"
                    counter += 1
                df.rename(columns={target: new_col}, inplace=True)

    df = df.rename(columns=rename_map)
    
    # Final cleanup: remove any duplicates if they somehow persist
    df = df.loc[:, ~df.columns.duplicated()]

    # 3. Filtering Logic
    risk_df = None
    
    if 'Last_Login_Days' in df.columns:
        try:
            # Handle potential duplicate columns (returns DataFrame instead of Series)
            col_data = df['Last_Login_Days']
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
                
            # Filter for At Risk Users (> 7 days inactive)
            # Ensure numeric
            df['Last_Login_Days'] = pd.to_numeric(col_data, errors='coerce').fillna(0)
            risk_df = df[df['Last_Login_Days'] > 7].copy()
            st.info(f"Filtering for users inactive > 7 days. Found {len(risk_df)} users.")
        except Exception as e:
            st.warning(f"⚠️ Could not parse 'Last_Login_Days' column correctly ({e}). Proceeding with all users.")
            risk_df = None

    if risk_df is None:
        # No filter - treat EVERYONE as a target
        risk_df = df.copy()
        # Ensure Last_Login_Days exists for display if it wasn't there
        if 'Last_Login_Days' not in risk_df.columns:
             risk_df['Last_Login_Days'] = "Unknown" # Placeholder
        st.info(f"Selecting ALL {len(risk_df)} users for drafting.")
    
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
                        # 1. Turn the entire row into a readable string
                        try:
                            user_context = row.to_json()
                        except Exception as e:
                            return f"Error serializing row: {e}"
                        
                        # 2. Use the Custom PROMPT_TEMPLATE with the full context
                        try:
                            prompt = PROMPT_TEMPLATE.format(user_context=user_context)
                        except KeyError as e:
                            return f"Error: Prompt template has invalid placeholder {e}. Make sure to use {{user_context}}."
                            
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
            name = row.get('Name', 'there')
            if pd.isna(name):
                name = 'there'
            
            raw_days = row.get('Last_Login_Days')
            # Handle duplicate columns (if raw_days is a Series)
            if isinstance(raw_days, pd.Series):
                raw_days = raw_days.iloc[0]
            
            days_val = pd.to_numeric(raw_days, errors='coerce')
            if pd.isna(days_val):
                return f"Hey {name}, noticed it's been a while since your last login. We miss you!"
            return f"Hey {name}, noticed it's been {int(days_val)} days since your last login. We miss you!"
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
            success_details = []
            
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
                    
                    if response.ok:
                        success_count += 1
                        success_details.append({'Name': row.get('Name', 'Unknown'), 'Phone': phone})
                    else:
                        st.error(f"Failed for {phone}: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error sending to {phone}: {str(e)}")
                
                progress_bar.progress((i + 1) / count_selected)

            st.success(f"Done! Sent {success_count} messages.")
            
            if success_details:
                st.write("### 🔍 Verify Sent Messages")
                for item in success_details:
                    url = f"https://spur.chat/{item['Phone']}"
                    st.markdown(f"- **{item['Name']}**: [spur.chat/{item['Phone']}]({url})")

    st.write("---")
    if st.button(f"📢 Send Slack Alerts for {count_selected} Selected", disabled=(count_selected == 0), help="Send an internal alert to your team about these specific users."):
        if not SLACK_WEBHOOK_URL:
            st.error("Please enter a Slack Webhook URL in the sidebar.")
        else:
            progress_bar_slack = st.progress(0)
            slack_success = 0
            
            for i, (index, row) in enumerate(selected_rows.iterrows()):
                # Construct Slack Payload
                slack_msg = {
                    "text": f"🚨 *Churn Risk Detected*\n*Customer:* {row.get('Name', 'Unknown')}\n*Phone:* {row.get('Phone', 'N/A')}\n*Inactive Days:* {row['Last_Login_Days']}\n*Proposed Message:* _{row.get('Draft_Message', 'N/A')}_"
                }
                
                try:
                    requests.post(SLACK_WEBHOOK_URL, json=slack_msg)
                    slack_success += 1
                except Exception as e:
                    st.error(f"Slack Error: {e}")
                
                progress_bar_slack.progress((i + 1) / count_selected)
            
            st.success(f"Sent {slack_success} alerts to Slack!")

    # 5. Single Customer Outreach
    st.write("---")
    st.subheader("⚡ Single Customer Outreach")
    
    label_map = {
        f"{row.get('Name', 'Unknown')} | {row.get('Phone', '')} | Row {idx}": idx
        for idx, row in edited_df.iterrows()
    }
    
    if label_map:
        selected_label = st.selectbox("Choose a customer", list(label_map.keys()))
        selected_idx = label_map[selected_label]
        selected_row = edited_df.loc[selected_idx]
        
        if st.session_state.get("single_customer_label") != selected_label:
            st.session_state["single_customer_label"] = selected_label
            st.session_state["single_message"] = selected_row.get("Draft_Message", "")
            st.session_state["single_gen_error"] = None
            st.session_state["single_gen_success"] = None
        
        st.text_area(
            "Message for selected customer",
            key="single_message",
            height=120
        )
        
        def generate_single_callback(api_key, model_name, row, prompt_tmpl):
            if not api_key:
                st.session_state["single_gen_error"] = "⚠️ Please enter your Gemini API Key in the sidebar first."
                return

            genai.configure(api_key=api_key)
            try:
                model = genai.GenerativeModel(model_name)
                user_context = row.to_json()
                prompt = prompt_tmpl.format(user_context=user_context)
                response = model.generate_content(prompt)
                st.session_state["single_message"] = response.text.strip()
                st.session_state["single_gen_success"] = "AI message generated for this customer."
                st.session_state["single_gen_error"] = None
            except KeyError as e:
                st.session_state["single_gen_error"] = f"Error: Prompt template has invalid placeholder {e}. Make sure to use {{user_context}}."
                st.session_state["single_gen_success"] = None
            except Exception as e:
                st.session_state["single_gen_error"] = f"Error generating message: {e}"
                st.session_state["single_gen_success"] = None

        single_col1, single_col2 = st.columns([1, 1])
        with single_col1:
            st.button("✨ Generate AI Message", key="generate_single",
                      on_click=generate_single_callback,
                      args=(GEMINI_API_KEY, MODEL_NAME, selected_row, PROMPT_TEMPLATE))
            
            if st.session_state.get("single_gen_error"):
                if "⚠️" in st.session_state["single_gen_error"]:
                     st.warning(st.session_state["single_gen_error"])
                else:
                     st.error(st.session_state["single_gen_error"])
            
            if st.session_state.get("single_gen_success"):
                st.success(st.session_state["single_gen_success"])
        
        with single_col2:
            if st.button("📤 Send Message", key="send_single"):
                if not SPUR_API_KEY:
                    st.error("Please enter your Spur API Key in the sidebar first!")
                else:
                    message_text = st.session_state.get("single_message", "").strip()
                    if not message_text:
                        st.warning("Please generate or enter a message before sending.")
                    else:
                        phone = str(selected_row['Phone']).replace("+", "").strip()
                        if phone.endswith('.0'): phone = phone[:-2]
                        encoded_text = quote(message_text)
                        image_url = f"https://api.spurnow.com/screenshot/text-message?text={encoded_text}"
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
                            if response.ok:
                                st.success("Message sent!")
                                url = f"https://spur.chat/{phone}"
                                st.markdown(f"👉 [spur.chat/{phone}]({url})")
                            else:
                                st.error(f"Failed for {phone}: {response.text}")
                        except Exception as e:
                            st.error(f"Error sending to {phone}: {str(e)}")


