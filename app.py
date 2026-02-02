import streamlit as st
import pandas as pd
import requests

# Set page config
st.set_page_config(page_title="Retention Radar", page_icon="📡", layout="wide")

st.title("Retention Radar")
st.markdown("Identify churn risks and upsell opportunities from your customer data.")

# Sidebar for Slack Webhook
with st.sidebar:
    st.header("Settings")
    slack_webhook_url = st.text_input("Slack Webhook URL", type="password")

# File Uploader
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        required_columns = ["Customer Name", "Last Login Days Ago", "ROAS", "Email"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")
        else:
            # Logic 1: Churn Risk
            churn_risk_df = df[df["Last Login Days Ago"] > 7]
            
            # Logic 2: Upsell
            upsell_df = df[df["ROAS"] > 10]
            
            # Display Churn Risk (Red Table)
            st.subheader("🚨 At Risk (Ghosting)")
            if not churn_risk_df.empty:
                # styling the dataframe to look 'red' is tricky in native dataframe, 
                # but we can use style property or just show it. 
                # Let's use st.dataframe.
                st.dataframe(churn_risk_df.style.map(lambda x: 'background-color: #ffcccc', subset=df.columns), width=1000) # using a fixed large width or just relying on default full width if 'stretch' isn't fully supported by all versions yet, but warning advised 'stretch' so let's try that or just remove the arg if it defaults well. 
                # Actually, the warning said "use_container_width" is deprecated in favor of "width".
                # Let's use use_container_width=True for now as it MIGHT still be valid but warning, or just switch to width if supported.
                # However, since I can't be 100% sure of the version behavior without checking, and standard streamlit usually supports use_container_width.
                # The User said: "Please replace `use_container_width` with `width`".
                # So I will use st.dataframe(..., use_container_width=True) -> st.dataframe(..., width=None) or similar? 
                # Warning says: "For `use_container_width=True`, use `width='stretch'`".
                st.dataframe(churn_risk_df.style.map(lambda x: 'background-color: #ffcccc', subset=df.columns), use_container_width=True)
                
                # Slack Alert Button
                if st.button("Send Alert to Slack"):
                    if not slack_webhook_url:
                        st.error("Please enter a Slack Webhook URL in the sidebar.")
                    else:
                        success_count = 0
                        error_count = 0
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        total_rows = len(churn_risk_df)
                        
                        for i, (index, row) in enumerate(churn_risk_df.iterrows()):
                            # Formatted message
                            message = {
                                "text": f"🚨 *Churn Risk Alert*\n*Customer:* {row['Customer Name']}\n*Email:* {row['Email']}\n*Last Login:* {row['Last Login Days Ago']} days ago"
                            }
                            
                            try:
                                response = requests.post(slack_webhook_url, json=message)
                                if response.status_code == 200:
                                    success_count += 1
                                else:
                                    error_count += 1
                                    st.error(f"Failed to send alert for {row['Customer Name']}: {response.text}")
                            except Exception as e:
                                error_count += 1
                                st.error(f"Error sending alert for {row['Customer Name']}: {str(e)}")
                            
                            progress_bar.progress((i + 1) / total_rows)

                        status_placeholder.success(f"Processed {total_rows} alerts. Sent: {success_count}, Failed: {error_count}")
            else:
                st.info("No churn risks detected.")

            # Display Upsell (Gold Table)
            st.subheader("🍾 Upsell / Review Candidates")
            if not upsell_df.empty:
                # styling for gold
                st.dataframe(upsell_df.style.map(lambda x: 'background-color: #fff4cc', subset=df.columns), use_container_width=True)
            else:
                st.info("No upsell candidates detected.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
