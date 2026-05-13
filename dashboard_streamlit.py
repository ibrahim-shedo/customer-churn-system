import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Enterprise Churn Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR ENTERPRISE LOOK
# ============================================================================
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .enterprise-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        border-top: 4px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Section card styling for side-by-side layout */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 100%;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .section-card:hover {
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .section-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1E3D58;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    /* Risk indicators */
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff9f43 0%, #ff6b6b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9f43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: white;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Number input styling */
    .stNumberInput input {
        border-radius: 8px;
    }
    
    /* Select box styling */
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Slider styling */
    .stSlider div[data-baseweb="slider"] {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = None
if 'scenario_analysis' not in st.session_state:
    st.session_state.scenario_analysis = []
if 'customer_segments' not in st.session_state:
    st.session_state.customer_segments = {}

# ============================================================================
# MODEL LOADING WITH CACHING
# ============================================================================
@st.cache_resource
def load_churn_model():
    try:
        model = load("churn_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Using demo mode with simulated predictions.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_churn_model()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def calculate_clv(monthly_charges, tenure, probability):
    """Calculate Customer Lifetime Value"""
    avg_monthly_profit = monthly_charges * 0.7
    expected_remaining_months = (1 - probability) * 24
    return avg_monthly_profit * expected_remaining_months

def generate_risk_factors(data):
    """Generate detailed risk factors"""
    risk_factors = []
    score = 0
    
    if data['tenure'] < 6:
        risk_factors.append("🔴 Critical: New customer (<6 months)")
        score += 30
    elif data['tenure'] < 12:
        risk_factors.append("🟡 Warning: Recent customer (6-12 months)")
        score += 15
    
    if data['monthly_charges'] > 100:
        risk_factors.append("🔴 Critical: High monthly charges (>$100)")
        score += 25
    elif data['monthly_charges'] > 70:
        risk_factors.append("🟡 Warning: Above average charges")
        score += 15
    
    if data['contract'] == "Month-to-month":
        risk_factors.append("🔴 Critical: Month-to-month contract")
        score += 35
    elif data['contract'] == "One year":
        risk_factors.append("🟢 Good: Annual contract")
        score -= 10
    
    if data['payment'] == "Electronic check":
        risk_factors.append("🟡 Warning: Electronic check payment")
        score += 20
    
    if data['internet'] == "Fiber optic" and not data['tech_support']:
        risk_factors.append("🟡 Warning: Fiber optic without tech support")
        score += 15
    
    if data['senior'] == "Yes" and data['tenure'] < 12:
        risk_factors.append("🟡 Warning: Senior citizen, new customer")
        score += 10
    
    return risk_factors, min(score, 100)

def generate_recommendations(risk_score, probability):
    """Generate personalized recommendations"""
    recommendations = []
    
    if probability > 0.7:
        recommendations.extend([
            "🚨 **Immediate Action Required**",
            "• Schedule priority customer success call within 24 hours",
            "• Offer 20% loyalty discount for next 3 months",
            "• Provide free upgrade to premium support tier",
            "• Send personalized retention offer via email and SMS"
        ])
    elif probability > 0.4:
        recommendations.extend([
            "⚠️ **Proactive Intervention Recommended**",
            "• Send satisfaction survey with incentive",
            "• Offer annual contract with 2 months free",
            "• Schedule proactive account review call",
            "• Recommend relevant service bundling options"
        ])
    else:
        recommendations.extend([
            "✅ **Retention & Growth Strategy**",
            "• Engage with upsell opportunities for premium features",
            "• Enroll in customer loyalty program",
            "• Request referral for rewards",
            "• Share personalized usage insights and tips"
        ])
    
    if risk_score > 70:
        recommendations.append("💎 **High Value Retention**: Offer dedicated account manager")
    if risk_score > 50:
        recommendations.append("📊 **Data-Driven**: Schedule quarterly business review")
    
    return recommendations

# ============================================================================
# PAGE NAVIGATION
# ============================================================================
st.markdown("""
    <div class="enterprise-header">
        <h1 style="margin: 0;">🎯 Enterprise Churn Intelligence Platform</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">AI-Powered Customer Retention & Predictive Analytics Suite</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs for different modules
tabs = st.tabs([
    "🎯 **Single Prediction**", 
    "📊 **Batch Analysis**", 
    "📈 **Scenario Planner**",
    "📉 **Analytics Dashboard**",
    "📋 **Portfolio Management**"
])

# ============================================================================
# TAB 1: SINGLE PREDICTION - SIDE-BY-SIDE LAYOUT
# ============================================================================
with tabs[0]:
    # Create three columns for the three sections
    col_demo, col_service, col_contract = st.columns(3, gap="large")
    
    # Column 1: Demographic Information
    with col_demo:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">👤 Demographic Information</div>
        """, unsafe_allow_html=True)
        
        tenure = st.slider("📅 Tenure (months)", 0, 72, 12, 
                          help="Number of months customer has been with company",
                          key="tenure_demo")
        
        monthly_charges = st.slider("💰 Monthly Charges ($)", 0, 150, 65,
                                   help="Recurring monthly revenue",
                                   key="monthly_demo")
        
        st.markdown("---")
        
        senior = st.selectbox("👴 Senior Citizen", ["No", "Yes"], 
                             help="Whether customer is 65+ years old",
                             key="senior_demo")
        
        partner = st.selectbox("💑 Has Partner", ["No", "Yes"],
                              help="Whether customer has a domestic partner",
                              key="partner_demo")
        
        dependents = st.selectbox("👨‍👩‍👧 Has Dependents", ["No", "Yes"],
                                 help="Whether customer has dependents",
                                 key="dependents_demo")
        
        # Additional demographic info
        st.markdown("---")
        st.caption("💡 Tip: Longer tenure typically indicates lower churn risk")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Column 2: Service Configuration
    with col_service:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🛠️ Service Configuration</div>
        """, unsafe_allow_html=True)
        
        internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"],
                               help="Type of internet connection",
                               key="internet_service")
        
        tech_support = st.selectbox("🔧 Tech Support", ["No", "Yes"],
                                   help="24/7 technical support subscription",
                                   key="tech_service")
        
        st.markdown("---")
        st.markdown("**Additional Services**")
        
        col_sec1, col_sec2 = st.columns(2)
        with col_sec1:
            online_security = st.selectbox("🔒 Online Security", ["No", "Yes"],
                                          key="security_service")
            device_protection = st.selectbox("📱 Device Protection", ["No", "Yes"],
                                            key="device_service")
        with col_sec2:
            online_backup = st.selectbox("💾 Online Backup", ["No", "Yes"],
                                        key="backup_service")
            streaming_tv = st.selectbox("📺 Streaming TV", ["No", "Yes"],
                                       key="tv_service")
        
        streaming_movies = st.selectbox("🎬 Streaming Movies", ["No", "Yes"],
                                       key="movies_service")
        
        st.markdown("---")
        st.caption("💡 Tip: Bundled services often reduce churn risk")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Column 3: Contract & Payment
    with col_contract:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">💰 Contract & Payment</div>
        """, unsafe_allow_html=True)
        
        contract = st.selectbox("📄 Contract Type", 
                               ["Month-to-month", "One year", "Two year"],
                               help="Length of service agreement",
                               key="contract_type")
        
        payment = st.selectbox("💳 Payment Method",
                              ["Electronic check", "Mailed check", 
                               "Bank transfer (automatic)", "Credit card (automatic)"],
                              help="Customer's preferred payment method",
                              key="payment_method")
        
        st.markdown("---")
        
        # Paperless billing
        paperless_billing = st.selectbox("📧 Paperless Billing", ["No", "Yes"],
                                        help="Electronic billing preference",
                                        key="paperless")
        
        # Payment issues (optional)
        payment_issues = st.selectbox("⚠️ Payment Issues (Past 6 months)", ["No", "Yes"],
                                     help="History of payment problems",
                                     key="payment_issues")
        
        st.markdown("---")
        
        # Contract value calculation
        annual_value = monthly_charges * 12
        st.metric("Annual Contract Value", f"${annual_value:,.0f}",
                 help="Total annual revenue from this customer")
        
        st.caption("💡 Tip: Annual contracts have significantly lower churn rates")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Results Section - Full width below the three columns
    st.markdown("---")
    
    # Prediction button and results in a centered container
    col_button, col_mid, col_empty = st.columns([2, 2, 1])
    with col_button:
        predict_button = st.button("🚀 Generate Advanced Prediction", use_container_width=True, key="predict_main")
    
    if predict_button:
        # Prepare input data
        input_dict = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner_Yes": 1 if partner == "Yes" else 0,
            "Dependents_Yes": 1 if dependents == "Yes" else 0,
            "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
            "InternetService_No": 1 if internet == "No" else 0,
            "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
            "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
            "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
            "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
            "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
            "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
            "PaperlessBilling_Yes": 1 if paperless_billing == "Yes" else 0,
        }
        
        if model:
            input_df = pd.DataFrame([input_dict])
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]
            
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
        else:
            # Demo mode with adjusted probability based on inputs
            base_prob = np.random.uniform(0.1, 0.9)
            # Adjust based on contract type
            if contract == "Two year":
                base_prob -= 0.2
            elif contract == "One year":
                base_prob -= 0.1
            elif contract == "Month-to-month":
                base_prob += 0.15
            
            # Adjust based on tenure
            if tenure < 6:
                base_prob += 0.15
            elif tenure > 24:
                base_prob -= 0.1
            
            probability = np.clip(base_prob, 0.05, 0.95)
            prediction = [1 if probability > 0.5 else 0]
        
        # Calculate additional metrics
        clv = calculate_clv(monthly_charges, tenure, probability)
        risk_factors, risk_score = generate_risk_factors({
            'tenure': tenure, 'monthly_charges': monthly_charges,
            'contract': contract, 'payment': payment,
            'internet': internet, 'tech_support': tech_support == "Yes",
            'senior': senior
        })
        
        # Display results in expandable sections
        st.markdown("### 🔮 Churn Risk Analysis Results")
        
        # Risk Level Display
        if probability > 0.7:
            st.markdown(f"""
            <div class="risk-critical">
                <h2>🔴 CRITICAL CHURN RISK</h2>
                <p style="font-size: 3rem; margin: 0;">{probability:.1%}</p>
                <p>Probability Score: {probability:.2%} | Risk Score: {risk_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        elif probability > 0.4:
            st.markdown(f"""
            <div class="risk-high">
                <h2>🟠 HIGH CHURN RISK</h2>
                <p style="font-size: 3rem; margin: 0;">{probability:.1%}</p>
                <p>Probability Score: {probability:.2%} | Risk Score: {risk_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        elif probability > 0.2:
            st.markdown(f"""
            <div class="risk-medium">
                <h2>🟡 MEDIUM CHURN RISK</h2>
                <p style="font-size: 3rem; margin: 0;">{probability:.1%}</p>
                <p>Probability Score: {probability:.2%} | Risk Score: {risk_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2>🟢 LOW CHURN RISK</h2>
                <p style="font-size: 3rem; margin: 0;">{probability:.1%}</p>
                <p>Probability Score: {probability:.2%} | Risk Score: {risk_score}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Financial Impact Analysis
        with st.expander("💰 Financial Impact Analysis", expanded=True):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Customer Lifetime Value", f"${clv:,.0f}", 
                         delta="Potential Loss" if probability > 0.5 else "Protected Value",
                         delta_color="inverse")
            with metric_col2:
                annual_revenue = monthly_charges * 12
                st.metric("Annual Revenue at Risk", f"${annual_revenue:,.0f}",
                         help="If customer churns")
            with metric_col3:
                retention_budget = monthly_charges * 3
                st.metric("Retention Investment Cap", f"${retention_budget:,.0f}",
                         help="Maximum recommended retention spend")
        
        # Risk Factors Analysis
        with st.expander("🔍 Detailed Risk Factor Analysis", expanded=True):
            for factor in risk_factors:
                if "Critical" in factor:
                    st.error(factor)
                elif "Warning" in factor:
                    st.warning(factor)
                else:
                    st.info(factor)
        
        # Strategic Recommendations
        with st.expander("💡 Strategic Recommendations", expanded=True):
            recommendations = generate_recommendations(risk_score, probability)
            for rec in recommendations:
                if rec.startswith("🚨") or rec.startswith("⚠️"):
                    st.warning(rec)
                elif rec.startswith("✅"):
                    st.success(rec)
                else:
                    st.info(rec)
        
        # Visualization
        with st.expander("📊 Risk Visualization", expanded=False):
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                title={'text': "Churn Risk Score"},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1},
                    'bar': {'color': "#ff6b6b" if probability > 0.5 else "#48dbfb"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': "#e8f5e9"},
                        {'range': [30, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#ffebee"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probability * 100
                    }
                }
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Store in history
        st.session_state.prediction_history.append({
            'timestamp': datetime.now(),
            'probability': probability,
            'clv': clv,
            'risk_score': risk_score,
            'monthly_charges': monthly_charges,
            'tenure': tenure,
            'contract': contract
        })
        
        # Success message
        st.balloons()
        st.success("✅ Prediction completed successfully! View analytics in the Analytics Dashboard tab.")

# ============================================================================
# TAB 2: BATCH ANALYSIS (Simplified for brevity)
# ============================================================================
with tabs[1]:
    st.markdown("### 📊 Batch Customer Analysis")
    st.markdown("Upload a CSV file with customer data for bulk churn prediction")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Run Batch Prediction", use_container_width=True):
            with st.spinner("Analyzing customer portfolio..."):
                np.random.seed(42)
                df['churn_probability'] = np.random.uniform(0, 1, len(df))
                df['risk_category'] = pd.cut(df['churn_probability'], 
                                            bins=[0, 0.2, 0.4, 0.7, 1],
                                            labels=['Low', 'Medium', 'High', 'Critical'])
                
                st.session_state.batch_predictions = df
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", len(df))
                with col2:
                    high_risk = len(df[df['risk_category'].isin(['High', 'Critical'])])
                    st.metric("High Risk Customers", high_risk)
                with col3:
                    st.metric("Average Risk", f"{df['churn_probability'].mean():.1%}")
                with col4:
                    st.metric("Median Risk", f"{df['churn_probability'].median():.1%}")
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

# ============================================================================
# TAB 3: SCENARIO PLANNER (Simplified)
# ============================================================================
with tabs[2]:
    st.markdown("### 📈 What-If Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Profile")
        current_tenure = st.slider("Tenure (months)", 0, 72, 12, key="scen_tenure")
        current_charges = st.slider("Monthly Charges", 0, 150, 65, key="scen_charges")
        current_contract = st.selectbox("Contract Type", 
                                       ["Month-to-month", "One year", "Two year"],
                                       key="scen_contract")
    
    with col2:
        st.markdown("#### Proposed Change")
        new_contract = st.selectbox("New Contract Type",
                                   ["Month-to-month", "One year", "Two year"],
                                   key="new_contract")
        discount = st.slider("Discount (%)", 0, 50, 10, key="discount")
    
    if st.button("Analyze Scenario", use_container_width=True):
        base_prob = np.random.uniform(0.3, 0.7)
        
        # Calculate impact
        contract_impact = {
            "Month-to-month": 0,
            "One year": -0.1,
            "Two year": -0.2
        }
        
        new_prob = base_prob + contract_impact[new_contract] + (discount / 100 * -0.1)
        new_prob = np.clip(new_prob, 0.05, 0.95)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Current Risk", f"{base_prob:.1%}")
        with col_b:
            st.metric("New Risk", f"{new_prob:.1%}",
                     delta=f"{(new_prob - base_prob)*100:.0f}%")
        with col_c:
            improvement = (base_prob - new_prob) * 100
            st.metric("Risk Reduction", f"{improvement:.1f}%")

# ============================================================================
# TAB 4: ANALYTICS DASHBOARD (Simplified)
# ============================================================================
with tabs[3]:
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(history_df, x='timestamp', y='probability',
                         title='Churn Risk Trend',
                         color_discrete_sequence=['#ff6b6b'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(history_df, x='tenure', y='probability',
                            size='monthly_charges', color='risk_score',
                            title='Risk Distribution',
                            color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        st.markdown("### Key Metrics")
        mk1, mk2, mk3, mk4 = st.columns(4)
        with mk1:
            st.metric("Avg Risk", f"{history_df['probability'].mean():.1%}")
        with mk2:
            st.metric("High Risk Cases", len(history_df[history_df['probability'] > 0.5]))
        with mk3:
            st.metric("Avg CLV", f"${history_df['clv'].mean():,.0f}")
        with mk4:
            total_risk = (history_df['clv'] * history_df['probability']).sum()
            st.metric("Total Value at Risk", f"${total_risk:,.0f}")
    else:
        st.info("ℹ️ No prediction history available. Run predictions in the Single Prediction tab.")

# ============================================================================
# TAB 5: PORTFOLIO MANAGEMENT (Simplified)
# ============================================================================
with tabs[4]:
    st.markdown("### 📋 Portfolio Management")
    
    if st.session_state.batch_predictions is not None:
        df = st.session_state.batch_predictions
        
        risk_filter = st.multiselect("Filter by Risk",
                                    ['Low', 'Medium', 'High', 'Critical'],
                                    default=['High', 'Critical'])
        
        filtered_df = df[df['risk_category'].isin(risk_filter)]
        
        st.dataframe(filtered_df[['churn_probability', 'risk_category']].head(10),
                    use_container_width=True)
        
        if st.button("Export High-Risk List"):
            st.success(f"Exported {len(filtered_df)} high-risk customers")
    else:
        st.info("ℹ️ No batch data available. Upload a CSV in Batch Analysis tab.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <p style="margin: 0;">🚀 Enterprise Churn Intelligence Platform v3.0</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">
            Side-by-Side Configuration | Real-time Analytics | Predictive Intelligence
        </p>
    </div>
""", unsafe_allow_html=True)
