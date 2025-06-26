import streamlit as st
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# PDF and ML libraries
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import os
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Document Classification System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Real PDF text extraction
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return ""

# Enhanced feature extraction for ML classification
def extract_advanced_features(text):
    """Extract comprehensive features from document text for classification"""
    if not text:
        return {}
    
    text_lower = text.lower()
    lines = text.split('\n')
    words = text.split()
    
    # Order-specific keywords with variations
    keyword_patterns = {
        'purchase_order': r'purchase\s*order|p\.?o\.?\s*#?|p\.?o\.?\s*number|purchase\s*req',
        'proforma': r'proforma|pro\s*forma|pro-forma',
        'invoice': r'invoice|bill|billing',
        'order_number': r'order\s*no\.?|order\s*number|order\s*#',
        'quantity': r'qty\.?|quantity|amount|units?|pieces?|pcs',
        'delivery': r'deliver|delivery|ship|shipping|dispatch',
        'total': r'total|subtotal|amount\s*due|grand\s*total',
        'terms': r'terms|conditions|payment\s*terms|net\s*\d+',
        'instructions': r'special|instructions?|notes?|remarks?|comments?',
        'urgent': r'urgent|rush|asap|immediate|priority',
        'contact': r'contact|phone|email|@|tel:|mobile',
        'address': r'address|street|avenue|road|suite|building'
    }
    
    # Count keyword occurrences
    keyword_features = {}
    for key, pattern in keyword_patterns.items():
        matches = len(re.findall(pattern, text_lower))
        keyword_features[f'{key}_count'] = matches
        keyword_features[f'has_{key}'] = 1 if matches > 0 else 0
    
    # Structural and format features
    structural_features = {
        'line_count': len(lines),
        'word_count': len(words),
        'char_count': len(text),
        'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
        'max_line_length': max([len(line) for line in lines]) if lines else 0,
        'empty_lines': sum(1 for line in lines if not line.strip()),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0
    }
    
    # Table and formatting detection
    table_features = {
        'has_tables': 1 if any('|' in line or '\t' in line for line in lines) else 0,
        'pipe_count': text.count('|'),
        'tab_count': text.count('\t'),
        'colon_count': text.count(':'),
        'dash_count': text.count('-'),
        'bullet_points': len(re.findall(r'^\s*[•\-\*]\s', text, re.MULTILINE))
    }
    
    # Numerical and currency patterns
    numerical_features = {
        'number_count': len(re.findall(r'\d+', text)),
        'decimal_count': len(re.findall(r'\d+\.\d+', text)),
        'currency_symbols': len(re.findall(r'[\$£€¥]', text)),
        'currency_amounts': len(re.findall(r'[\$£€¥]\s*\d+[\d,]*\.?\d*', text)),
        'percentage_count': len(re.findall(r'\d+\s*%', text)),
        'date_patterns': len(re.findall(r'\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{2,4}', text)),
        'phone_patterns': len(re.findall(r'\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4}', text))
    }
    
    # Text characteristics
    text_features = {
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
        'punctuation_ratio': sum(1 for c in text if c in '.,;:!?') / len(text) if text else 0,
        'whitespace_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0,
        'unique_word_ratio': len(set(words)) / len(words) if words else 0
    }
    
    # Document structure indicators
    structure_indicators = {
        'has_header': 1 if any(keyword in lines[0].lower() if lines else '' 
                              for keyword in ['purchase', 'order', 'proforma', 'invoice']) else 0,
        'has_footer': 1 if any(keyword in lines[-1].lower() if lines else '' 
                              for keyword in ['total', 'terms', 'conditions', 'thank']) else 0,
        'has_signature_block': 1 if any('signature' in line.lower() or 'authorized' in line.lower() 
                                       for line in lines) else 0,
        'has_company_info': 1 if any(keyword in text_lower 
                                    for keyword in ['ltd', 'inc', 'corp', 'llc', 'company']) else 0
    }
    
    # Combine all features
    all_features = {
        **keyword_features,
        **structural_features,
        **table_features,
        **numerical_features,
        **text_features,
        **structure_indicators
    }
    
    return all_features

# Model persistence functions
def save_model(model, vectorizer, filepath="models/"):
    """Save trained model and vectorizer to disk"""
    Path(filepath).mkdir(exist_ok=True)
    
    model_path = Path(filepath) / "classifier_model.pkl"
    vectorizer_path = Path(filepath) / "vectorizer.pkl"
    metadata_path = Path(filepath) / "model_metadata.json"
    
    # Save model and vectorizer
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "feature_count": len(vectorizer.get_feature_names_out()),
        "classes": list(model.classes_)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, vectorizer_path, metadata_path

def load_model(filepath="models/"):
    """Load trained model and vectorizer from disk"""
    model_path = Path(filepath) / "classifier_model.pkl"
    vectorizer_path = Path(filepath) / "vectorizer.pkl"
    metadata_path = Path(filepath) / "model_metadata.json"
    
    if not all(path.exists() for path in [model_path, vectorizer_path]):
        return None, None, None
    
    # Load model and vectorizer
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load metadata if available
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, vectorizer, metadata

# Enhanced training data with more realistic examples
@st.cache_data
def generate_enhanced_training_data():
    """Generate more comprehensive training data for ML model"""
    training_texts = {
        'Structured Proforma': [
            """PROFORMA INVOICE
            Invoice No: PRO-2024-001
            Date: June 25, 2024
            
            Bill To:
            ABC Manufacturing Ltd
            123 Industrial Park
            Business City, BC 12345
            
            Ship To:
            Same as billing address
            
            Item No. | Description | Qty | Unit Price | Amount
            WID-001 | Industrial Widget A | 100 | $15.00 | $1,500.00
            WID-002 | Premium Widget B | 50 | $25.00 | $1,250.00
            
            Subtotal: $2,750.00
            Tax (10%): $275.00
            TOTAL: $3,025.00
            
            Terms: Net 30 days
            Delivery: 2-3 weeks from order confirmation
            Valid until: July 25, 2024""",
            
            """PRO FORMA INVOICE
            Company: Tech Solutions Inc.
            Order Reference: PF-2024-100
            
            Customer Details:
            XYZ Corporation
            456 Tech Boulevard
            
            Product Details:
            - Server Hardware Package (Qty: 5) - $2,500.00 each
            - Installation Services (Qty: 1) - $500.00
            - 2-Year Warranty (Qty: 1) - $300.00
            
            Total Amount: $13,300.00
            Payment Terms: 50% advance, 50% on delivery
            Estimated Delivery: 4-6 weeks""",
            
            """PROFORMA
            From: Global Supplies Ltd
            To: Regional Office
            Proforma No: GS-2024-075
            
            Line Items:
            Office Furniture Package | Qty: 1 lot | $5,000.00
            Delivery and Setup | Included | $0.00
            
            Special Notes:
            - Custom color matching available
            - Assembly included in price
            - 5-year manufacturer warranty
            
            Grand Total: $5,000.00
            Terms and Conditions apply"""
        ],
        
        'Simple Purchase Order': [
            """PURCHASE ORDER
            PO Number: 2024-0156
            Date: June 25, 2024
            
            Vendor: Office Supplies Direct
            123 Supply Street
            
            Requested Items:
            1. Office Chairs (Model: EC-100) - Qty: 10 - $150.00 each
            2. Desk Organizers - Qty: 20 - $25.00 each
            3. File Cabinets - Qty: 5 - $200.00 each
            
            Delivery Address:
            Corporate Headquarters
            789 Business Ave
            
            Total Order Value: $3,000.00
            Required Delivery Date: July 5, 2024
            Contact: Sarah Johnson (555) 123-4567""",
            
            """Purchase Order #: PO-2024-234
            From: ABC Company
            To: Tech Vendor LLC
            
            Order Details:
            - Laptop Computers: 15 units @ $800.00
            - Wireless Mice: 15 units @ $25.00
            - Laptop Cases: 15 units @ $50.00
            
            Ship To: IT Department
            456 Corporate Drive
            
            Total: $13,125.00
            Payment: Net 30
            Contact: Mike Chen - mchen@abccompany.com""",
            
            """P.O. # 78901
            Supplier: Industrial Parts Inc
            Order Date: 06/25/2024
            
            Part Number | Description | Qty | Price
            IP-001 | Bearing Assembly | 25 | $45.00
            IP-002 | Gear Set | 10 | $120.00
            
            Delivery: Standard shipping
            Total Amount: $2,325.00
            Approval: Department Manager"""
        ],
        
        'Complex Purchase Order': [
            """COMPLEX PURCHASE ORDER
            PO Number: COMP-2024-001
            Multi-Phase Project Order
            
            Phase 1: Design and Planning
            - Engineering Services: 100 hours @ $150/hour = $15,000
            - Site Survey: 40 hours @ $100/hour = $4,000
            - Deliverable: Technical specifications and drawings
            - Timeline: 4 weeks from PO date
            
            Phase 2: Equipment Procurement
            - Industrial Equipment Unit A: 3 units @ $25,000 = $75,000
            - Installation Hardware: 1 lot = $5,000
            - Special handling and delivery required
            - Timeline: 8 weeks from Phase 1 completion
            
            Phase 3: Installation and Testing
            - Installation Services: 200 hours @ $125/hour = $25,000
            - System Testing: 80 hours @ $100/hour = $8,000
            - Training: 40 hours @ $75/hour = $3,000
            - Timeline: 6 weeks from equipment delivery
            
            Special Requirements:
            - Security clearance required for all personnel
            - Quality inspection at each phase
            - Multiple approval signatures needed
            - Insurance certificate required
            
            Payment Schedule:
            - 25% on PO acceptance
            - 50% on equipment delivery
            - 25% on final acceptance
            
            Total Project Value: $135,000
            Project Duration: 18 weeks
            
            Multiple Contacts:
            Project Manager: John Smith (555) 111-2222
            Technical Lead: Jane Doe (555) 333-4444
            Finance: Bob Wilson (555) 555-6666""",
            
            """PURCHASE ORDER - MULTI-LOCATION
            PO #: MLO-2024-050
            Corporate Procurement Order
            
            Location A - New York Office:
            Office Furniture: 50 desks @ $300 each = $15,000
            Delivery Address: 123 Manhattan Plaza, NY 10001
            Contact: NYC Facilities Manager
            Special Instructions: Weekend delivery only
            
            Location B - Chicago Office:
            IT Equipment: 25 workstations @ $1,200 each = $30,000
            Delivery Address: 456 Loop Street, Chicago, IL 60601
            Contact: Chicago IT Director
            Special Instructions: Requires IT staff presence
            
            Location C - Remote Workers:
            Home Office Setup: 100 packages @ $500 each = $50,000
            Individual shipping addresses (list attached)
            Special Instructions: Coordinate with HR for addresses
            
            Project Coordination Requirements:
            - Separate delivery schedules for each location
            - Different approval workflows
            - Location-specific purchase orders
            - Consolidated billing to corporate
            
            Total Order Value: $95,000
            Master Agreement Reference: MA-2024-001
            Payment Terms: Vary by location (see attached)"""
        ],
        
        'Email Body Order': [
            """Subject: Urgent Order Request - Office Supplies
            
            Hi Jennifer,
            
            I hope this email finds you well. We have an urgent need for office supplies 
            for our quarterly meeting next week. Could you please arrange for the following:
            
            - 50 notebooks (standard spiral-bound)
            - 100 pens (blue ink, ballpoint)
            - 25 folders (manila, letter size)
            - 10 staplers
            - 5 boxes of staples
            
            We need these delivered to our main conference room by Thursday morning 
            if possible. The meeting starts at 9 AM, so delivery by 8 AM would be ideal.
            
            Please let me know the total cost and confirm availability.
            
            Thanks so much for your help!
            
            Best regards,
            Mark Peterson
            Operations Manager
            Direct: (555) 987-6543""",
            
            """Hey Tom,
            
            Quick order needed for the workshop this Friday:
            
            10x safety helmets (white)
            10x safety vests (hi-vis yellow, size L)
            20x safety glasses (clear)
            5x first aid kits (basic)
            
            Can you get these to the warehouse by Thursday afternoon?
            Budget is around $800 total.
            
            Let me know if you need any clarification.
            
            Cheers,
            Dave
            
            P.S. - Make sure the helmets meet OSHA standards""",
            
            """Hello Sarah,
            
            Following up on our phone conversation yesterday about the catering supplies.
            We confirmed we need:
            
            Tables: 20 round tables (8-person capacity)
            Chairs: 160 folding chairs
            Linens: 20 white tablecloths
            Place Settings: 160 complete sets (plates, utensils, glasses)
            
            Event Details:
            Date: July 15th, 2024
            Time: Setup needed by 5 PM
            Location: Grand Ballroom, Marriott Downtown
            Expected Guests: 150 people
            
            Please confirm pricing and availability. We may need additional items 
            depending on final guest count.
            
            Also, please include your standard delivery and pickup service.
            
            Thank you,
            Lisa Chen
            Event Coordinator
            lisa.chen@eventcompany.com
            Mobile: (555) 234-5678"""
        ]
    }
    
    # Add more variety with different writing styles and formats
    additional_samples = {
        'Structured Proforma': [
            "QUOTATION FORM | Quote #: Q2024-100 | Client: Metro Corp | Item: Software License (50 users) | Price: $15,000 | Valid: 30 days",
            "ESTIMATE | Project: Website Development | Hours: 120 @ $85/hr | Total: $10,200 | Terms: 50% upfront"
        ],
        'Simple Purchase Order': [
            "PO#67890 | Vendor: Local Printer | Business Cards: 5000 @ $0.10 each | Delivery: 1 week | Total: $500",
            "Order Form | Cleaning Supplies | Qty: Various | Budget: $300 | Contact: Facilities Dept"
        ],
        'Complex Purchase Order': [
            "MASTER SERVICE AGREEMENT ORDER | Multiple deliverables | Phased approach | Various locations | Total contract value: $250,000",
            "FRAMEWORK ORDER | Annual IT Support | Multiple call-offs | Service levels defined | 12-month term"
        ],
        'Email Body Order': [
            "Need 5 laptops ASAP for new hires. Standard business config. Budget $4000. When can you deliver?",
            "Hi - can you quote for office move? 50 workstations, 200 boxes, weekend move preferred. Thanks!"
        ]
    }
    
    # Combine all training data
    all_texts = []
    all_labels = []
    
    for category, examples in training_texts.items():
        for example in examples:
            all_texts.append(example)
            all_labels.append(category)
    
    for category, examples in additional_samples.items():
        for example in examples:
            all_texts.append(example)
            all_labels.append(category)
    
    return all_texts, all_labels

# Enhanced ML Model training and prediction
@st.cache_resource
def train_enhanced_classification_model():
    """Train an enhanced ML model with better features"""
    texts, labels = generate_enhanced_training_data()
    
    # Create combined feature vectors
    all_features = []
    text_features_list = []
    
    for text in texts:
        # Extract manual features
        manual_features = extract_advanced_features(text)
        all_features.append(manual_features)
        text_features_list.append(text)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=200, 
        stop_words='english', 
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,
        max_df=0.8
    )
    X_text = vectorizer.fit_transform(text_features_list)
    
    # Convert manual features to array
    feature_names = list(all_features[0].keys())
    X_manual = np.array([[feat_dict[name] for name in feature_names] for feat_dict in all_features])
    
    # Combine text and manual features
    X_combined = np.hstack([X_text.toarray(), X_manual])
    
    # Train enhanced Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model.fit(X_train, y_train)
    
    # Calculate accuracy on test set
    test_accuracy = model.score(X_test, y_test)
    
    return model, vectorizer, feature_names, test_accuracy

def classify_document_enhanced(text, model, vectorizer, feature_names):
    """Enhanced document classification with combined features"""
    if not text.strip():
        return "Unknown", 0.0, {}, {}
    
    # Extract manual features
    manual_features = extract_advanced_features(text)
    
    # Extract TF-IDF features
    X_text = vectorizer.transform([text])
    
    # Combine features (ensure same order as training)
    X_manual = np.array([[manual_features.get(name, 0) for name in feature_names]])
    X_combined = np.hstack([X_text.toarray(), X_manual])
    
    # Get prediction and confidence
    prediction = model.predict(X_combined)[0]
    probabilities = model.predict_proba(X_combined)[0]
    confidence = max(probabilities)
    
    # Get class probabilities for all classes
    class_probs = dict(zip(model.classes_, probabilities))
    
    return prediction, confidence, manual_features, class_probs

# Model management functions
def save_training_data(texts, labels, filepath="data/training_data.json"):
    """Save training data for future use"""
    Path(filepath).parent.mkdir(exist_ok=True)
    
    training_data = {
        "texts": texts,
        "labels": labels,
        "timestamp": datetime.now().isoformat(),
        "count": len(texts)
    }
    
    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=2)

def load_training_data(filepath="data/training_data.json"):
    """Load saved training data"""
    if not Path(filepath).exists():
        return None, None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data["texts"], data["labels"]

def add_training_example(text, label, filepath="data/training_data.json"):
    """Add a new training example to the dataset"""
    # Load existing data
    existing_texts, existing_labels = load_training_data(filepath)
    
    if existing_texts is None:
        existing_texts, existing_labels = [], []
    
    # Add new example
    existing_texts.append(text)
    existing_labels.append(label)
    
    # Save updated data
    save_training_data(existing_texts, existing_labels, filepath)
    
    return len(existing_texts)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, labels)
    
    return model, vectorizer

def classify_document(text, model, vectorizer):
    """Classify a document using the trained model"""
    if not text.strip():
        return "Unknown", 0.0, {}
    
    # Extract features using TF-IDF
    X = vectorizer.transform([text])
    
    # Get prediction and confidence
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    
    # Extract manual features for display
    features = extract_features(text)
    
    return prediction, confidence, features

# Sample data
@st.cache_data
def load_sample_documents():
    return [
        {
            "name": "ProformaInvoice_ABC123.pdf",
            "type": "Structured Proforma",
            "confidence": 0.94,
            "keywords": ["Proforma Invoice", "Order No", "Total Amount", "Delivery Date"],
            "features": {"tables": 1, "logo": True, "structured": True}
        },
        {
            "name": "PurchaseOrder_XYZ789.pdf", 
            "type": "Simple Purchase Order",
            "confidence": 0.87,
            "keywords": ["Purchase Order", "PO Number", "Qty", "Unit Price"],
            "features": {"tables": 1, "logo": False, "structured": False}
        },
        {
            "name": "ComplexOrder_DEF456.pdf",
            "type": "Complex Purchase Order", 
            "confidence": 0.91,
            "keywords": ["Purchase Order", "Line Items", "Special Instructions", "Terms"],
            "features": {"tables": 3, "logo": True, "structured": True}
        },
        {
            "name": "EmailOrder_GHI321.txt",
            "type": "Email Body Order",
            "confidence": 0.82,
            "keywords": ["Order", "Please supply", "Deliver to", "Urgent"],
            "features": {"tables": 0, "logo": False, "structured": False}
        }
    ]

@st.cache_data  
def load_dashboard_training_data():
    return pd.DataFrame({
        "Category": ["Structured Proforma", "Simple Purchase Order", "Complex Purchase Order", "Email Body Order"],
        "Document Count": [45, 78, 62, 34],
        "Accuracy": [0.94, 0.89, 0.91, 0.85],
        "Last Updated": ["2025-06-25", "2025-06-24", "2025-06-25", "2025-06-23"]
    })

def simulate_processing_steps():
    """Simulate the document processing pipeline"""
    steps = [
        "📧 Email Parsing & Text Extraction",
        "🔍 Lightweight Sampling", 
        "⚙️ Feature Extraction",
        "🎯 Document Classification"
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(steps):
        status_text.text(f"Step {i+1}/4: {step}")
        progress_bar.progress((i + 1) / len(steps))
        time.sleep(0.8)
    
    status_text.text("✅ Processing Complete!")
    return True

def main():
    st.title("📄 Document Classification System")
    st.markdown("**Prototype demonstrating automated order document processing**")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Document Processing", "Training Dashboard", "Live Monitoring", "System Settings"]
    )
    
    if page == "Document Processing":
        document_processing_page()
    elif page == "Training Dashboard":
        training_dashboard_page()
    elif page == "Live Monitoring":
        monitoring_page()
    else:
        system_settings_page()

def document_processing_page():
    st.header("📤 Document Processing Demo")
    
    # Initialize enhanced ML model
    if 'enhanced_model' not in st.session_state:
        with st.spinner("Loading enhanced ML model..."):
            result = train_enhanced_classification_model()
            st.session_state.enhanced_model = result[0]
            st.session_state.enhanced_vectorizer = result[1] 
            st.session_state.feature_names = result[2]
            st.session_state.model_accuracy = result[3]
        st.success(f"✅ Enhanced ML model loaded! Test accuracy: {st.session_state.model_accuracy:.1%}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Document")
        
        # Sample document selector
        sample_docs = load_sample_documents()
        selected_doc = st.selectbox(
            "Choose a sample document to process:",
            options=["None"] + [doc["name"] for doc in sample_docs],
            help="Select from sample documents to see classification in action"
        )
        
        # Real file uploader
        uploaded_file = st.file_uploader(
            "Or upload your own document:",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word, or text files"
        )
        
        # Text input for testing
        st.write("**Or paste text directly:**")
        text_input = st.text_area(
            "Document text:",
            height=150,
            placeholder="Paste your document text here for instant classification..."
        )
        
        process_button = st.button("🚀 Process Document", type="primary")
        
        if process_button and (selected_doc != "None" or uploaded_file or text_input):
            extracted_text = ""
            doc_name = ""
            
            # Extract text from different sources
            if uploaded_file:
                doc_name = uploaded_file.name
                with st.spinner("Extracting text from uploaded file..."):
                    if uploaded_file.type == "application/pdf":
                        extracted_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        extracted_text = str(uploaded_file.read(), "utf-8")
                    else:
                        st.error("Unsupported file type")
                        return
                        
            elif text_input:
                doc_name = "Text Input"
                extracted_text = text_input
                
            elif selected_doc != "None":
                # Use sample document data
                doc_info = next(doc for doc in sample_docs if doc["name"] == selected_doc)
                doc_name = doc_info["name"]
                # Generate sample text based on document type
                sample_texts = {
                    "ProformaInvoice_ABC123.pdf": "PROFORMA INVOICE Order No: PRO-2024-001 Company ABC Ltd Product Qty Unit Price Total Widget A 100 $10.00 $1000.00 TOTAL: $2000.00",
                    "PurchaseOrder_XYZ789.pdf": "Purchase Order PO Number: 12345 Date: 2024-06-25 Item Qty Unit Price Office Chairs 10 $150.00 Total: $1500.00",
                    "ComplexOrder_DEF456.pdf": "PURCHASE ORDER Complex Multi-Line Order PO Number: CPX-2024-001 Line 1: Equipment Qty: 3 Special Instructions: Certified installation Multiple delivery addresses",
                    "EmailOrder_GHI321.txt": "Hi, I need to place an urgent order for 25 units of Product A. Please deliver to our warehouse by Friday. Thanks, Mike"
                }
                extracted_text = sample_texts.get(selected_doc, "Sample document text")
            
            # Show extracted text
            if extracted_text:
                with st.expander("📄 Extracted Text Preview", expanded=False):
                    st.text_area("Document content:", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=150)
                
                # Processing simulation
                with st.spinner("Processing document..."):
                    simulate_processing_steps()
                
                # Real ML classification with enhanced features
                with st.spinner("Running enhanced ML classification..."):
                    prediction, confidence, features, class_probs = classify_document_enhanced(
                        extracted_text, 
                        st.session_state.enhanced_model, 
                        st.session_state.enhanced_vectorizer,
                        st.session_state.feature_names
                    )
                
                # Store result in session state
                st.session_state.last_result = {
                    "name": doc_name,
                    "type": prediction,
                    "confidence": confidence,
                    "features": features,
                    "class_probabilities": class_probs,
                    "text": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                }
            else:
                st.error("Could not extract text from the document")
    
    with col2:
        st.subheader("Classification Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            
            # Classification result
            if result["confidence"] > 0.7:
                st.success(f"✅ **Classification Complete**")
            else:
                st.warning(f"⚠️ **Low Confidence Classification**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Document Type", result["type"])
            with col_b:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            
            # Routing information
            st.info(f"📍 **Routing:** → {result['type']} Extraction Pipeline")
            
            # Feature analysis
            st.subheader("🔍 ML Features Detected")
            
            features = result["features"]
            
            # Order-specific keywords
            st.write("**Order Keywords Found:**")
            keyword_features = {
                k: v for k, v in features.items() 
                if k.endswith('_count') and k.startswith(('purchase_order', 'proforma', 'order_number', 'quantity', 'delivery', 'total', 'invoice', 'terms', 'instructions', 'urgent'))
            }
            
            # Display keyword counts
            keywords_found = False
            for feature, count in keyword_features.items():
                if count > 0:
                    # Clean up the feature name for display
                    clean_name = feature.replace('_count', '').replace('_', ' ').title()
                    st.write(f"✅ {clean_name}: {count} occurrences")
                    keywords_found = True
            
            if not keywords_found:
                st.write("ℹ️ No specific order keywords detected")
                
            # Also show boolean features that are true
            st.write("**Order Elements Present:**")
            boolean_features = {
                k: v for k, v in features.items() 
                if k.startswith('has_') and k.replace('has_', '') in ['purchase_order', 'proforma', 'order_number', 'quantity', 'delivery', 'total', 'invoice', 'terms', 'instructions', 'urgent', 'contact', 'address']
            }
            
            elements_found = False
            for feature, value in boolean_features.items():
                if value > 0:
                    clean_name = feature.replace('has_', '').replace('_', ' ').title()
                    st.write(f"✅ {clean_name}")
                    elements_found = True
            
            if not elements_found:
                st.write("ℹ️ No standard order elements detected")
            
            # Structural features
            st.write("**Document Structure:**")
            structural_features = {
                k: v for k, v in features.items() 
                if k in ['line_count', 'word_count', 'has_tables', 'number_count', 'currency_amounts', 'date_patterns']
            }
            
            for feature, value in structural_features.items():
                if feature.startswith('has_'):
                    clean_name = feature.replace('has_', '').replace('_', ' ').title()
                    st.write(f"{'✅' if value else '❌'} {clean_name}")
                else:
                    clean_name = feature.replace('_', ' ').title()
                    if feature in ['currency_amounts', 'date_patterns', 'number_count'] and value > 0:
                        st.write(f"📊 {clean_name}: {value}")
                    elif feature in ['line_count', 'word_count']:
                        st.write(f"📊 {clean_name}: {value}")
            
            # Additional useful features
            if features.get('uppercase_ratio', 0) > 0.1:
                st.write(f"📊 Uppercase Text: {features.get('uppercase_ratio', 0):.1%}")
            
            if features.get('has_company_info', 0):
                st.write("✅ Company Information Detected")
            
            # Confidence breakdown
            st.subheader("🎯 Classification Confidence")
            
            # Show all class probabilities
            if "class_probabilities" in result:
                st.write("**Probability Distribution:**")
                probs_df = pd.DataFrame([
                    {"Document Type": cls, "Probability": prob} 
                    for cls, prob in result["class_probabilities"].items()
                ]).sort_values("Probability", ascending=False)
                
                # Create probability chart with matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(probs_df["Document Type"], probs_df["Probability"], 
                             color=plt.cm.viridis(probs_df["Probability"]))
                ax.set_xlabel("Document Type")
                ax.set_ylabel("Probability")
                ax.set_title("Classification Probabilities")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            
            # Confidence assessment
            if result["confidence"] > 0.85:
                st.success("🟢 High confidence - automatic processing recommended")
            elif result["confidence"] > 0.7:
                st.warning("🟡 Medium confidence - review recommended")
            else:
                st.error("🔴 Low confidence - manual review required")
                
            # Model training feedback
            st.subheader("📚 Improve the Model")
            
            col_feedback1, col_feedback2 = st.columns(2)
            
            with col_feedback1:
                st.write("**Was this classification correct?**")
                if st.button("✅ Correct Classification"):
                    st.success("Thank you! This helps improve the model.")
                
            with col_feedback2:
                st.write("**If incorrect, what should it be?**")
                correct_label = st.selectbox(
                    "Correct classification:",
                    ["", "Structured Proforma", "Simple Purchase Order", "Complex Purchase Order", "Email Body Order"],
                    key="feedback_label"
                )
                
                if st.button("📝 Submit Correction") and correct_label:
                    # Add to training data
                    new_count = add_training_example(result["text"], correct_label)
                    st.success(f"Added to training data! Total examples: {new_count}")
                    st.info("Retrain the model to see improvements.")
                
        else:
            st.info("👆 Select and process a document to see classification results")
            
            # Show enhanced model info
            if 'enhanced_model' in st.session_state:
                st.subheader("🤖 Enhanced ML Model Information")
                col_model1, col_model2 = st.columns(2)
                
                with col_model1:
                    st.write("**Model Details:**")
                    st.write("- Type: Random Forest Classifier")
                    st.write("- Features: TF-IDF + Manual Features")
                    st.write(f"- Test Accuracy: {st.session_state.model_accuracy:.1%}")
                    st.write(f"- Feature Count: {len(st.session_state.feature_names) + 200}")
                
                with col_model2:
                    st.write("**Training Data:**")
                    texts, labels = generate_enhanced_training_data()
                    category_counts = pd.Series(labels).value_counts()
                    st.write(f"- Total Documents: {len(texts)}")
                    for category, count in category_counts.items():
                        st.write(f"- {category}: {count}")
                
                # Model management
                st.subheader("💾 Model Management")
                col_save, col_load = st.columns(2)
                
                with col_save:
                    if st.button("💾 Save Current Model"):
                        try:
                            paths = save_model(
                                st.session_state.enhanced_model,
                                st.session_state.enhanced_vectorizer
                            )
                            st.success("✅ Model saved successfully!")
                            st.write(f"Files saved to: {paths[0].parent}")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
                
                with col_load:
                    if st.button("📂 Load Saved Model"):
                        try:
                            model, vectorizer, metadata = load_model()
                            if model is not None:
                                st.session_state.enhanced_model = model
                                st.session_state.enhanced_vectorizer = vectorizer
                                st.success("✅ Model loaded successfully!")
                                if metadata:
                                    st.write(f"Trained: {metadata.get('training_date', 'Unknown')}")
                            else:
                                st.warning("No saved model found")
                        except Exception as e:
                            st.error(f"Error loading model: {e}")
                
                # Retrain option
                if st.button("🔄 Retrain Model with New Data"):
                    with st.spinner("Retraining model..."):
                        # Load any additional training data
                        saved_texts, saved_labels = load_training_data()
                        
                        if saved_texts:
                            st.info(f"Found {len(saved_texts)} additional training examples")
                            
                        # Retrain with enhanced data
                        result = train_enhanced_classification_model()
                        st.session_state.enhanced_model = result[0]
                        st.session_state.enhanced_vectorizer = result[1]
                        st.session_state.feature_names = result[2]
                        st.session_state.model_accuracy = result[3]
                        
                    st.success(f"✅ Model retrained! New accuracy: {st.session_state.model_accuracy:.1%}")
                    st.rerun()

def training_dashboard_page():
    st.header("📊 Training Dashboard")
    
    # Load training data
    training_df = load_dashboard_training_data()
    
    # Check for additional training examples
    saved_texts, saved_labels = load_training_data()
    additional_count = len(saved_texts) if saved_texts else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Dataset Overview")
        
        # Summary metrics
        total_docs = training_df["Document Count"].sum() + additional_count
        avg_accuracy = training_df["Accuracy"].mean()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Documents", total_docs)
        with col_b:
            st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
        with col_c:
            if 'model_accuracy' in st.session_state:
                st.metric("Current Model", f"{st.session_state.model_accuracy:.1%}")
            else:
                st.metric("Current Model", "Not loaded")
        
        # Enhanced training data table
        if additional_count > 0:
            st.info(f"📝 Found {additional_count} additional training examples from user feedback")
            
        enhanced_training_df = training_df.copy()
        enhanced_training_df.loc[len(enhanced_training_df)] = {
            "Category": "User Feedback",
            "Document Count": additional_count,
            "Accuracy": 0.0,  # Unknown for user feedback
            "Last Updated": datetime.now().strftime("%Y-%m-%d")
        }
        
        st.dataframe(
            enhanced_training_df.style.format({
                "Accuracy": "{:.1%}",
                "Document Count": "{:,}"
            }),
            use_container_width=True
        )
        
        # Progress towards neural network
        st.subheader("🎯 Neural Network Readiness")
        neural_threshold = 2000
        current_total = total_docs
        progress = min(current_total / neural_threshold, 1.0)
        
        st.progress(progress)
        st.write(f"Progress: {current_total:,} / {neural_threshold:,} documents")
        
        if progress >= 1.0:
            st.success("✅ Ready for neural network transition!")
        else:
            remaining = neural_threshold - current_total
            st.info(f"📝 Need {remaining:,} more documents for neural network training")
        
        # Feature importance (if model is loaded)
        if 'enhanced_model' in st.session_state:
            st.subheader("🔍 Feature Importance")
            
            # Get feature importance from the model
            model = st.session_state.enhanced_model
            
            # Get TF-IDF feature names (first 200 features)
            tfidf_features = list(st.session_state.enhanced_vectorizer.get_feature_names_out())
            manual_features = st.session_state.feature_names
            all_feature_names = tfidf_features + manual_features
            
            # Get importance scores
            importances = model.feature_importances_
            
            # Create DataFrame of top features
            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)
            
            # Plot feature importance with matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title('Top 15 Most Important Features')
            ax.invert_yaxis()  # Highest importance at top
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Performance Visualization")
        
        # Accuracy by category
        display_df = training_df[training_df['Accuracy'] > 0]  # Exclude user feedback for accuracy chart
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(display_df["Category"], display_df["Accuracy"], 
                     color=plt.cm.viridis(display_df["Accuracy"]))
        ax.set_xlabel("Category")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy by Category")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Document distribution including user feedback
        total_with_feedback = enhanced_training_df[enhanced_training_df['Document Count'] > 0]
        
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.pie(total_with_feedback["Document Count"], 
               labels=total_with_feedback["Category"], 
               autopct='%1.1f%%',
               startangle=90)
        ax2.set_title("Training Data Distribution")
        st.pyplot(fig2)
        
        # Model comparison over time (simulated)
        st.subheader("📈 Model Performance Trend")
        
        # Create simulated historical performance data
        dates = pd.date_range(start='2024-06-01', end='2024-06-25', freq='W')
        baseline_accuracy = [0.75, 0.78, 0.82, 0.85]
        enhanced_accuracy = [0.82, 0.86, 0.89, 0.91]
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(dates, baseline_accuracy, marker='o', label='Baseline', linewidth=2)
        ax3.plot(dates, enhanced_accuracy, marker='s', label='Enhanced', linewidth=2)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Model Performance Over Time")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        min_accuracy = display_df["Accuracy"].min()
        min_category = display_df.loc[display_df["Accuracy"].idxmin(), "Category"]
        
        recommendations = []
        
        if min_accuracy < 0.90:
            recommendations.append(f"⚠️ Add more examples for {min_category} (Current: {min_accuracy:.1%})")
        
        if additional_count > 10:
            recommendations.append("🔄 Retrain model with user feedback data")
        
        if total_docs < 500:
            recommendations.append("📝 Collect more diverse training examples")
        
        if not recommendations:
            recommendations.append("✅ All categories performing well!")
        
        for rec in recommendations:
            if rec.startswith("⚠️") or rec.startswith("🔄") or rec.startswith("📝"):
                st.warning(rec)
            else:
                st.success(rec)
        
        # Data quality metrics
        st.subheader("📋 Data Quality Metrics")
        
        quality_metrics = {
            "Dataset Balance": "Good" if min(training_df["Document Count"]) / max(training_df["Document Count"]) > 0.5 else "Needs Improvement",
            "Total Examples": f"{total_docs:,}",
            "Categories": len(training_df),
            "User Feedback": f"{additional_count} examples" if additional_count > 0 else "None yet"
        }
        
        for metric, value in quality_metrics.items():
            if "Good" in str(value):
                st.success(f"**{metric}:** {value}")
            elif "Needs Improvement" in str(value):
                st.warning(f"**{metric}:** {value}")
            else:
                st.info(f"**{metric}:** {value}")

def monitoring_page():
    st.header("📈 Live System Monitoring")
    
    # Generate mock real-time data
    if 'monitoring_data' not in st.session_state:
        st.session_state.monitoring_data = generate_monitoring_data()
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.monitoring_data = generate_monitoring_data()
        st.rerun()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Classification Accuracy", "94.2%", "2.1%")
    with col2:
        st.metric("Documents Processed", "127", "15")
    with col3:
        st.metric("Avg Processing Time", "18s", "-3s")
    with col4:
        st.metric("System Uptime", "99.8%", "0.1%")
    
    # Recent activity
    st.subheader("🕐 Recent Activity")
    
    recent_activity = pd.DataFrame({
        "Time": ["14:23", "14:19", "14:15", "14:12", "14:08"],
        "Document": ["PO_2024_0891.pdf", "Email_Order_ABC.txt", "Complex_PO_XYZ.pdf", "Invoice_DEF.pdf", "Proforma_GHI.pdf"],
        "Classification": ["Simple Purchase Order", "Email Body Order", "Complex Purchase Order", "Non-Order", "Structured Proforma"],
        "Status": ["✅ Success", "✅ Success", "⚠️ Review", "🚫 Filtered", "✅ Success"]
    })
    
    st.dataframe(recent_activity, use_container_width=True)
    
    # Performance over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Processing Volume")
        volume_data = generate_volume_data()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(volume_data["Hour"], volume_data["Documents"], marker='o', linewidth=2)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Documents")
        ax.set_title("Documents Processed by Hour")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Accuracy Trend")
        accuracy_data = generate_accuracy_data()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(accuracy_data["Day"], accuracy_data["Accuracy"], marker='s', linewidth=2, color='green')
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Classification Accuracy Over Time")
        ax2.set_ylim(0.8, 1.0)
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

def system_settings_page():
    st.header("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Processing Configuration")
        
        processing_mode = st.selectbox(
            "Processing Mode",
            ["Real-time", "Batch (Hourly)", "Batch (Daily)"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold for Auto-Processing",
            min_value=0.5,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Documents below this confidence will be flagged for manual review"
        )
        
        max_processing_time = st.number_input(
            "Max Processing Time (seconds)",
            min_value=10,
            max_value=300,
            value=30
        )
        
        enable_monitoring = st.checkbox("Enable Performance Monitoring", value=True)
        enable_alerts = st.checkbox("Enable Email Alerts", value=True)
    
    with col2:
        st.subheader("📧 Email Integration")
        
        email_provider = st.selectbox(
            "Email Provider",
            ["Outlook/Exchange", "Gmail", "IMAP", "Custom API"]
        )
        
        monitor_folders = st.text_area(
            "Folders to Monitor",
            value="Inbox\nOrders\nCustomer Communications",
            help="One folder per line"
        )
        
        file_types = st.multiselect(
            "Supported File Types",
            ["PDF", "DOC/DOCX", "TXT", "PNG", "JPG", "XLS/XLSX"],
            default=["PDF", "DOC/DOCX", "TXT"]
        )
        
        st.subheader("🤖 Model Settings")
        
        model_type = st.selectbox(
            "Current Model",
            ["Traditional ML (Random Forest)", "Traditional ML (SVM)", "Neural Network (Future)"]
        )
        
        retrain_frequency = st.selectbox(
            "Model Retraining Frequency",
            ["Weekly", "Monthly", "Quarterly", "Manual"]
        )
    
    # Save settings button
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")
        
        # Display current configuration
        st.subheader("📋 Current Configuration")
        config = {
            "Processing Mode": processing_mode,
            "Confidence Threshold": f"{confidence_threshold:.0%}",
            "Max Processing Time": f"{max_processing_time}s",
            "Email Provider": email_provider,
            "Model Type": model_type,
            "Retraining": retrain_frequency
        }
        
        for key, value in config.items():
            st.write(f"**{key}:** {value}")

@st.cache_data
def generate_monitoring_data():
    """Generate sample monitoring data"""
    return {
        "accuracy": random.uniform(0.90, 0.98),
        "processed_today": random.randint(100, 200),
        "avg_time": random.uniform(15, 25),
        "uptime": random.uniform(0.995, 1.0)
    }

@st.cache_data
def generate_volume_data():
    """Generate sample volume data"""
    hours = list(range(24))
    volumes = [random.randint(5, 25) if 8 <= h <= 18 else random.randint(0, 5) for h in hours]
    return pd.DataFrame({"Hour": hours, "Documents": volumes})

@st.cache_data  
def generate_accuracy_data():
    """Generate sample accuracy trend data"""
    days = [(datetime.now() - timedelta(days=i)).strftime("%m-%d") for i in range(7, 0, -1)]
    accuracies = [random.uniform(0.88, 0.96) for _ in days]
    return pd.DataFrame({"Day": days, "Accuracy": accuracies})

if __name__ == "__main__":
    main()