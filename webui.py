import os
import fitz  # PyMuPDF
import streamlit as st
import plotly.express as px
import pandas as pd
from google.cloud import vision

from google.oauth2 import service_account
from google.cloud import vision
# --- LANGUAGE MAPS ---
LANGUAGE_MAP = {
    'af': 'Afrikaans', 'sq': 'Albanian', 'ar': 'Arabic', 'hy': 'Armenian',
    'be': 'Belarusian', 'bn': 'Bengali', 'bg': 'Bulgarian', 'ca': 'Catalan',
    'zh': 'Chinese', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
    'nl': 'Dutch', 'en': 'English', 'et': 'Estonian', 'fil': 'Filipino',
    'fi': 'Finnish', 'fr': 'French', 'de': 'German', 'el': 'Greek',
    'gu': 'Gujarati', 'iw': 'Hebrew', 'hi': 'Hindi', 'hu': 'Hungarian',
    'is': 'Icelandic', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese',
    'kn': 'Kannada', 'km': 'Khmer', 'ko': 'Korean', 'lo': 'Lao',
    'lv': 'Latvian', 'lt': 'Lithuanian', 'mk': 'Macedonian', 'ms': 'Malay',
    'ml': 'Malayalam', 'mr': 'Marathi', 'ne': 'Nepali', 'no': 'Norwegian',
    'fa': 'Persian', 'pl': 'Polish', 'pt': 'Portuguese', 'pa': 'Punjabi',
    'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian', 
    'es': 'Spanish', 'sv': 'Swedish', 'tl': 'Tagalog', 'ta': 'Tamil', 
    'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 
    'vi': 'Vietnamese'
}
REVERSE_LANGUAGE_MAP = {v: k for k, v in LANGUAGE_MAP.items()}

# --- CONFIGURATION ---


# ... (inside your analyze_document function where you define the client) ...

# Securely load the Google credentials from Streamlit Secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = vision.ImageAnnotatorClient(credentials=credentials)
st.set_page_config(page_title="Batch OCR Dashboard", layout="wide", page_icon="📄")

# --- CORE EXTRACTION LOGIC ---
def extract_data_from_response(response, language_hints):
    """Helper function to extract words and languages from a single Vision API response."""
    words = 0
    langs = {}
    
    if response.error.message:
        st.error(f"API Error: {response.error.message}")
        return words, langs

    if response.full_text_annotation:
        for v_page in response.full_text_annotation.pages:
            # Safely grab the auto-detected primary language of THIS specific page
            page_fallback_lang = "en" 
            if v_page.property and v_page.property.detected_languages:
                page_fallback_lang = v_page.property.detected_languages[0].language_code.split('-')[0]

            for block in v_page.blocks:
                for paragraph in block.paragraphs:
                    num_words = len(paragraph.words)
                    words += num_words
                    
                    para_text = "".join([symbol.text for word in paragraph.words for symbol in word.symbols])
                    has_letters = any(c.isalpha() for c in para_text)
                    lang_name = "Other (Numbers/Symbols)"
                    
                    if has_letters:
                        # FIX: Default to the auto-detected language of this page, NOT the global hint
                        clean_code = page_fallback_lang 
                        
                        if paragraph.property and paragraph.property.detected_languages:
                            for detected_lang in paragraph.property.detected_languages:
                                temp_code = detected_lang.language_code.split('-')[0]
                                
                                # If it matches a user hint, accept it with low confidence
                                if language_hints and temp_code in language_hints:
                                    if detected_lang.confidence > 0.05:
                                        clean_code = temp_code
                                        break
                                # If it's a completely different language, accept it only with high confidence
                                elif detected_lang.confidence > 0.10:
                                    clean_code = temp_code
                                    break
                        
                        lang_name = LANGUAGE_MAP.get(clean_code, clean_code.upper())
                    
                    langs[lang_name] = langs.get(lang_name, 0) + num_words
    return words, langs

# --- BATCH PROCESSING LOGIC ---
def analyze_document(file_bytes, file_name, file_type, language_hints, client, context, logs, terminal):
    """Processes a single file (PDF or Image) and logs it to the terminal."""
    total_words = 0
    language_counts = {}
    total_pages = 0
    
    # 1. HANDLE PDF FILES
    if "pdf" in file_type:
        doc = fitz.Document(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        logs.append(f"> 📄 Opened PDF: {file_name} ({total_pages} pages)")
        terminal.code("\n".join(logs[-10:]), language="bash")
        
        for i, page in enumerate(doc):
            logs.append(f"> ⏳ Scanning {file_name} - Page {i + 1}/{total_pages}...")
            terminal.code("\n".join(logs[-10:]), language="bash")
            
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = vision.Image(content=pix.tobytes("png"))
            response = client.document_text_detection(image=image, image_context=context)
            
            w_count, l_counts = extract_data_from_response(response, language_hints)
            total_words += w_count
            for k, v in l_counts.items():
                language_counts[k] = language_counts.get(k, 0) + v
                
            logs.append(f"> ✔️ {file_name} (Page {i + 1}) done: {w_count} words.")
            terminal.code("\n".join(logs[-10:]), language="bash")
            
    # 2. HANDLE IMAGE FILES
    else:
        total_pages = 1
        logs.append(f"> 🖼️ Scanning Image: {file_name}...")
        terminal.code("\n".join(logs[-10:]), language="bash")
        
        image = vision.Image(content=file_bytes)
        response = client.document_text_detection(image=image, image_context=context)
        
        w_count, l_counts = extract_data_from_response(response, language_hints)
        total_words += w_count
        for k, v in l_counts.items():
            language_counts[k] = language_counts.get(k, 0) + v
            
        logs.append(f"> ✔️ {file_name} done: {w_count} words.")
        terminal.code("\n".join(logs[-10:]), language="bash")
        
    return total_words, language_counts, total_pages

# --- WEB UI LAYOUT ---
st.title("📄 Batch OCR Document Dashboard")
st.markdown("Upload multiple PDFs or Images to aggregate total word counts and language distributions.")
st.divider()

left_col, right_col = st.columns([1, 1.5]) 

with left_col:
    st.subheader("1. Settings & Upload")
    selected_lang_names = st.multiselect(
        "Select expected languages (improves accuracy):",
        options=list(REVERSE_LANGUAGE_MAP.keys()),
        default=["English"] 
    )
    hint_codes = [REVERSE_LANGUAGE_MAP[name] for name in selected_lang_names]
    
    st.write("---")
    # UPDATED: Accept multiple files AND image formats
    uploaded_files = st.file_uploader(
        "Choose PDF or Image files", 
        type=["pdf", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    analyze_button = st.button(
        "🚀 Analyze All Documents", 
        type="primary", 
        use_container_width=True, 
        disabled=not uploaded_files # Disabled if list is empty
    )

with right_col:
    st.subheader("2. Analysis Results")
    
    if uploaded_files and analyze_button:
        # Grand Total Variables
        grand_total_words = 0
        grand_total_pages = 0
        aggregate_languages = {}
        file_results_table = []
        
        # Setup Terminal
        terminal_placeholder = st.empty()
        logs = [f"> Initializing Batch Analyzer for {len(uploaded_files)} files..."]
        terminal_placeholder.code("\n".join(logs), language="bash")
        
        # Setup API Client
        client = vision.ImageAnnotatorClient()
        context = vision.ImageContext(language_hints=hint_codes) if hint_codes else None
        
        progress_bar = st.progress(0)
        
        # --- BATCH PROCESSING LOOP ---
        for idx, file in enumerate(uploaded_files):
            file_bytes = file.read()
            
            # Process the individual file
            w_count, l_counts, p_count = analyze_document(
                file_bytes, file.name, file.type, hint_codes, 
                client, context, logs, terminal_placeholder
            )
            
            # Add to Grand Totals
            grand_total_words += w_count
            grand_total_pages += p_count
            for k, v in l_counts.items():
                aggregate_languages[k] = aggregate_languages.get(k, 0) + v
            
            # --- NEW: Calculate Prominent Language ---
            prominent_lang = "None"
            if l_counts:
                # Filter out "Other" so we get the actual prominent human language
                actual_langs = {k: v for k, v in l_counts.items() if "Other" not in k}
                if actual_langs:
                    prominent_lang = max(actual_langs, key=actual_langs.get) # Get language with max words
                else:
                    prominent_lang = "Other"
                
            # Log for the table (Now includes Prominent Language!)
            file_results_table.append({
                "File Name": file.name,
                "Type": "PDF" if "pdf" in file.type else "Image",
                "Pages": p_count,
                "Prominent Language": prominent_lang,
                "Word Count": f"{w_count:,}"
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        logs.append("> 🏁 Batch Analysis Complete. Rendering dashboard...")
        terminal_placeholder.code("\n".join(logs[-10:]), language="bash")
        progress_bar.empty()
        
        # --- DISPLAY RESULTS ---
        st.write("### 📁 Individual File Stats")
        # Display as a clean, structured table
        st.dataframe(pd.DataFrame(file_results_table),width="stretch")
        
        st.write("### 🏆 Grand Totals")
        m1, m2 = st.columns(2)
        m1.metric(label="Total Pages Processed", value=grand_total_pages)
        m2.metric(label="Total Words Detected", value=f"{grand_total_words:,}")
        
        if aggregate_languages:
            fig = px.pie(
                names=list(aggregate_languages.keys()), 
                values=list(aggregate_languages.values()), 
                title="Overall Language Distribution (All Files)",
                hole=0.4, 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=16)
            fig.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            
            st.plotly_chart(fig, width='stretch')
            
    elif not uploaded_files:
        st.info("👈 Please drag and drop your PDFs and Images on the left.")
    else:
        st.info("👆 Click **Analyze All Documents** when you are ready to begin.")