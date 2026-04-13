import streamlit as st
import traceback

# Simple, crash-proof UI design
st.set_page_config(page_title="AI Text Generator", layout="centered")

st.title("AI Text Generator")
st.write("Generate text safely with AI.")

try:
    from inference import load_environment, generate_text
    
    @st.cache_resource
    def load_model_safe():
        try:
            return load_environment()
        except Exception as e:
            return None, None

    tokenizer, model = load_model_safe()
    
    if tokenizer is None or model is None:
        st.error("Failed to load model or tokenizer. Please check if tokenizer.pkl and best_model.pt exist.")
    else:
        user_input = st.text_input("Input Text:", "")
        
        if st.button("Generate"):
            try:
                if not user_input or user_input.strip() == "":
                    st.warning("Please enter some text.")
                else:
                    with st.spinner("Generating..."):
                        try:
                            # 100% try-catch guarded generation call
                            output = generate_text(model, tokenizer, str(user_input).strip())
                            
                            # Handle fallback/error messages cleanly
                            if str(output).startswith("Error") or "Safely recovered" in str(output):
                                st.warning(output)
                            else:
                                st.success("Generated Output:")
                                st.write(output)
                        except Exception as e:
                            st.error(f"Something broke, but app is safe now. Safe default text returned. Exception: {e}")
            except Exception as outer_e:
                st.error(f"A critical error occurred inside generation loop, safely caught: {outer_e}")
                
except Exception as global_e:
    st.error(f"A global error occurred, safely caught: {global_e}")
