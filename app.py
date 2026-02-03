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
    # Check if key exists in secrets and is not the default placeholder
    if key_name in st.secrets and st.secrets[key_name] and "YOUR_" not in st.secrets[key_name]:
        st.sidebar.text_input(label, value="Securely Configured", disabled=True, type="password")
        return st.secrets[key_name]
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

    # Initialize session state for prompt if not exists
    if "prompt_input" not in st.session_state:
        st.session_state["prompt_input"] = load_prompt()

    PROMPT_TEMPLATE = st.text_area(
        "Edit the instructions for the AI:", 
        value=st.session_state["prompt_input"],
        height=200,
        help="Use {user_context} as the placeholder for the entire row of data.",
        key="prompt_input",
        on_change=save_prompt
    )
    
    if st.button("🔄 Reset to Default Prompt"):
        if os.path.exists(PROMPT_FILE):
            os.remove(PROMPT_FILE)
        st.session_state["prompt_input"] = load_prompt()
        st.rerun()

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
    
    df = df.rename(columns=rename_map)

    # 3. Filtering Logic
    if 'Last_Login_Days' in df.columns:
        # Filter for At Risk Users (> 7 days inactive)
        # Ensure numeric
        df['Last_Login_Days'] = pd.to_numeric(df['Last_Login_Days'], errors='coerce').fillna(0)
        risk_df = df[df['Last_Login_Days'] > 7].copy()
        st.info(f"Filtering for users inactive > 7 days. Found {len(risk_df)} users.")
    else:
        # No filter - treat EVERYONE as a target
        risk_df = df.copy()
        risk_df['Last_Login_Days'] = "Unknown" # Placeholder
        st.info(f"No 'Days Inactive' column mapped. Selecting ALL {len(risk_df)} users for drafting.")
        
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


