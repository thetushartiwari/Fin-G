# Fin-G (Finance Guardian) - Complete Project Analysis

## 1. COMPLETE TECHNICAL SUMMARY

### Core Idea
**Fin-G (Finance Guardian)** is a web-based financial advisory platform that uses machine learning to assess investor risk profiles and provide personalized investment recommendations. The system combines unsupervised learning (K-Means clustering) with real-time stock market data to generate risk-based investment allocations and stock suggestions.

### Workflow
1. **User Input Collection**: User provides financial data (income, savings, expenses, age) and stock sector preference via web form
2. **Risk Assessment**: K-Means clustering model assigns user to one of 10 risk clusters (risk scores 1-10)
3. **Investment Allocation**: Based on risk score, system allocates savings across 4 asset classes:
   - Low Risk (1-3): 50% Mutual Funds, 20% Stocks, 20% SIPs, 10% Govt Bonds
   - Moderate Risk (4-6): 30% Mutual Funds, 40% Stocks, 20% SIPs, 10% Govt Bonds
   - High Risk (7-10): 10% Mutual Funds, 60% Stocks, 20% SIPs, 10% Govt Bonds
4. **Stock Recommendations**: Fetches real-time stock data via yfinance API, calculates 6-month returns, and recommends top 3 stocks from selected sector based on risk category
5. **Results Display**: Presents risk score, allocation breakdown, and stock recommendations in formatted HTML

### Algorithms Used
- **K-Means Clustering** (scikit-learn): Unsupervised learning algorithm for risk profiling
  - Configuration: 10 clusters, random_state=42, n_init=10
  - Input features: Income, Savings, Expenses, Age (4 features)
  - Output: Risk score (1-10, mapped from cluster labels)
- **StandardScaler**: Feature normalization for clustering
- **Simple Rule-Based Allocation**: Deterministic investment allocation based on risk score ranges

### Code Flow
```
User Request → Django URL Router (prediction/urls.py)
    ↓
views.py: analyze_user_investment()
    ↓
Extract POST data (income, savings, expenses, age, stockPreference)
    ↓
k_mean.py: predict_risk(user_input)
    ├─→ StandardScaler.transform() [uses in-memory scaler]
    └─→ KMeans.predict() [uses in-memory model]
    ↓
views.py: classify_investment(savings, risk_score)
    └─→ Returns allocation dictionary
    ↓
st_perform.py: get_top_stocks(interest, risk_score)
    ├─→ Maps risk_score to category (low/moderate/high)
    ├─→ Selects stocks from SECTOR_MAPPING
    ├─→ yfinance API calls for each stock (6-month history)
    ├─→ Calculates return percentage: ((end_price - start_price) / start_price) * 100
    └─→ Returns top 3 stocks sorted by return
    ↓
Render template with results (risk_score, allocation, top_stocks)
```

### Model Training Pipeline
**Location**: `prediction/k_mean.py`

**Training Process**:
1. **Data Generation**: Synthetic dataset of 500 samples
   - Income: random integers 30,000-200,000
   - Savings: random integers 5,000-100,000
   - Expenses: random integers 10,000-150,000
   - Age: random integers 20-70
   - Random seed: 42 (for reproducibility)

2. **Preprocessing**:
   - StandardScaler.fit_transform() on all 4 features
   - Normalizes features to zero mean and unit variance

3. **Model Training**:
   - KMeans.fit() on scaled data
   - 10 clusters (n_clusters=10)
   - Cluster labels mapped to risk scores (labels + 1, so 0-9 → 1-10)

4. **Model Persistence**:
   - `kmeans_risk_model.pkl`: Saved KMeans model
   - `scaler.pkl`: Saved StandardScaler

**Note**: The code has a design issue - models are trained on module import and saved, but `predict_risk()` uses in-memory objects rather than loading from pickle files. This means the model retrains on every server restart.

### Evaluation Method
**No formal evaluation metrics are implemented**. The system:
- Uses unsupervised learning (clustering) without ground truth labels
- No validation/test split
- No silhouette score, inertia, or other clustering metrics
- No backtesting of investment recommendations
- No performance tracking of recommended stocks

### APIs/Packages Used
**Backend**:
- **Django 5.1.6**: Web framework
- **scikit-learn 1.6.1**: KMeans, StandardScaler
- **yfinance 0.2.52**: Real-time stock market data API
- **joblib 1.4.2**: Model serialization
- **pandas 2.2.3**: Data manipulation
- **numpy 2.2.3**: Numerical operations

**Frontend**:
- **Tailwind CSS 2.2.19** (CDN): Styling framework
- **GSAP 3.12.2** (CDN): Animation library
- **Django Templates**: Server-side rendering

### Frontend/Backend Details

**Backend Architecture**:
- **Framework**: Django 5.1.6 (MVC pattern)
- **Database**: SQLite3 (default Django setup, but no custom models used)
- **URL Routing**: Django URL dispatcher
- **Views**: Function-based views with CSRF exemption
- **Template Engine**: Django template language
- **Static Files**: CSS served via Django static files

**Frontend Architecture**:
- **Template**: `template/index.html` - Main application page
- **Template**: `template/success_stories.html` - Marketing page
- **Styling**: Custom CSS (`static/style.css`) + Tailwind CSS
- **Features**:
  - Glassmorphism UI design
  - GSAP animations (fade-in effects)
  - Responsive form with dropdowns
  - Dynamic content display based on results
- **No JavaScript Framework**: Vanilla JS with GSAP for animations

**Key Frontend Components**:
- Risk assessment form (income, savings, expenses, age, sector preference)
- Results display section (risk score, allocation, stock recommendations)
- Success stories marketing page
- Future features dropdown (placeholder)

---

## 2. EXACT TECH STACK

### Programming Languages
- **Python 3.x** (inferred from Django 5.1.6 compatibility)

### Frameworks
- **Django 5.1.6**: Web application framework
- **Tailwind CSS 2.2.19**: Utility-first CSS framework (via CDN)

### ML Libraries
- **scikit-learn 1.6.1**: Machine learning library
  - `sklearn.cluster.KMeans`: Clustering algorithm
  - `sklearn.preprocessing.StandardScaler`: Feature scaling
- **joblib 1.4.2**: Model serialization
- **pandas 2.2.3**: Data manipulation
- **numpy 2.2.3**: Numerical computing

### Datasets Used
- **Synthetic Data**: Generated in `k_mean.py`
  - 500 samples, 4 features (Income, Savings, Expenses, Age)
  - Random generation with seed=42
  - **No real-world financial dataset used**
- **Real-Time Stock Data**: Fetched via yfinance API
  - Sectors: Technology, Finance, Healthcare, FMCG
  - Stocks: Pre-defined list in `SECTOR_MAPPING` (st_perform.py)
  - Data period: 6 months historical data
  - **No historical training dataset for stocks**

### Architectures Used
- **MVC (Model-View-Controller)**: Django's architecture pattern
- **Unsupervised Learning**: K-Means clustering for risk profiling
- **RESTful Web Application**: Django URL routing with POST requests

### Tools Used
- **Git**: Not explicitly visible, but standard for version control
- **Docker**: Not used
- **Virtual Environment**: Not explicitly shown, but recommended
- **Package Manager**: pip (requirements.txt present)

### External APIs
- **yfinance API**: Yahoo Finance API wrapper for stock market data
  - Used for: Fetching 6-month historical stock prices
  - Endpoint: `stock.history(period="6mo")`

---

## 3. ORIGINAL vs BOILERPLATE/AI-GENERATED ANALYSIS

### **ORIGINAL WORK** (What You Can Claim):

1. **Core Business Logic** (`prediction/views.py`):
   - `classify_investment()` function: Custom risk-based allocation algorithm
   - Integration of risk prediction with stock recommendations
   - Form handling and data validation logic

2. **Stock Recommendation System** (`prediction/st_perform.py`):
   - `SECTOR_MAPPING` dictionary: Curated stock lists by sector and risk
   - `get_top_stocks()` function: Real-time stock data fetching and return calculation
   - Integration of yfinance API with risk-based filtering

3. **Frontend Design** (`template/index.html`, `static/style.css`):
   - Glassmorphism UI design
   - Custom CSS styling
   - GSAP animation integration
   - User experience flow (form → results display)

4. **Project Concept**:
   - Combining ML risk profiling with real-time stock recommendations
   - Financial advisory web application concept

### **BOILERPLATE/AI-GENERATED** (What You Should Acknowledge):

1. **Django Project Structure**:
   - `manage.py`: Standard Django boilerplate (unchanged)
   - `fin/settings.py`: Standard Django settings (mostly default)
   - `fin/urls.py`: Standard Django URL configuration
   - `prediction/apps.py`: Standard Django app config
   - `prediction/admin.py`: Empty, standard Django file
   - `prediction/models.py`: Empty, standard Django file
   - `prediction/tests.py`: Empty, standard Django file

2. **K-Means Implementation** (`prediction/k_mean.py`):
   - Standard scikit-learn KMeans usage (no custom algorithm)
   - Synthetic data generation is basic (random integers)
   - **No feature engineering, no domain expertise in risk modeling**
   - Model training code is straightforward sklearn pipeline

3. **Template Structure**:
   - Basic Django template syntax
   - Tailwind CSS classes (standard utility classes)
   - GSAP usage is minimal (basic fade-in)

4. **README.md**: Minimal (just project name)

### **HONEST ASSESSMENT**:

**Strengths (Original)**:
- ✅ Functional end-to-end system
- ✅ Real-time stock data integration
- ✅ Custom business logic for investment allocation
- ✅ Working web application with UI

**Weaknesses (Limitations)**:
- ❌ No real financial dataset (synthetic data only)
- ❌ No model evaluation or validation
- ❌ K-Means may not be the best choice for risk profiling (supervised learning would be more appropriate)
- ❌ No feature engineering or domain knowledge in risk modeling
- ❌ Model retrains on every import (design flaw)
- ❌ No database persistence of user data or predictions
- ❌ No authentication or user management
- ❌ Limited error handling

**What You Can Safely Claim**:
- Built a web-based financial advisory platform using Django
- Implemented ML-based risk profiling using K-Means clustering
- Integrated real-time stock market data via yfinance API
- Developed custom investment allocation algorithm based on risk scores
- Created responsive web UI with modern design patterns

**What You Should Be Honest About**:
- Used synthetic data for model training (not real financial data)
- No formal model evaluation or validation metrics
- Standard scikit-learn implementations (no custom ML algorithms)
- Django project structure is mostly boilerplate

---

## 4. CV-READY PROJECT DESCRIPTION (3-5 Bullet Points)

**Fin-G: AI-Powered Financial Advisory Platform**

• Developed a full-stack web application using Django and machine learning to provide personalized investment recommendations based on user risk profiles, integrating K-Means clustering for risk assessment and real-time stock market data via yfinance API

• Designed and implemented a custom investment allocation algorithm that dynamically distributes user savings across mutual funds, stocks, SIPs, and government bonds based on ML-derived risk scores (1-10 scale)

• Built a responsive web interface with modern UI/UX design (glassmorphism effects, GSAP animations) and integrated real-time stock performance analysis to recommend top-performing stocks by sector and risk category

• Engineered an end-to-end ML pipeline using scikit-learn (K-Means clustering, StandardScaler) to process user financial data (income, savings, expenses, age) and generate actionable investment insights

• Implemented a stock recommendation system that fetches 6-month historical data for 30+ stocks across 4 sectors (Technology, Finance, Healthcare, FMCG), calculates returns, and filters recommendations by risk tolerance

---

## 5. RESEARCH-INTERNSHIP-READY DESCRIPTION

**Fin-G: Machine Learning-Driven Financial Risk Profiling and Investment Recommendation System**

**Motivation**:
Traditional financial advisory services are often inaccessible to retail investors due to high costs and complexity. Additionally, investors frequently make suboptimal decisions based on market hype rather than data-driven risk assessment. This project addresses the need for an automated, accessible system that combines unsupervised learning for risk profiling with real-time market data analysis to provide personalized investment guidance.

**Methodology**:
The system employs a two-stage approach: (1) **Risk Profiling**: K-Means clustering (k=10) is applied to user financial features (income, savings, expenses, age) after standardization, mapping users to risk categories (1-10). The clustering approach groups users with similar financial profiles, enabling unsupervised discovery of risk patterns without requiring labeled training data. (2) **Investment Recommendation**: Based on the assigned risk score, a rule-based allocation algorithm distributes investments across asset classes (mutual funds, stocks, SIPs, government bonds) with proportions inversely correlated to risk tolerance. For stock-level recommendations, the system integrates with yfinance API to fetch 6-month historical data for sector-specific stocks, calculates percentage returns, and ranks stocks to recommend the top 3 performers within the user's risk category.

**Technical Implementation**:
Built on Django 5.1.6 with scikit-learn for ML components, the system processes user inputs through a StandardScaler-normalized pipeline, applies K-Means clustering (random_state=42, n_init=10), and generates recommendations via real-time API calls. The frontend uses Tailwind CSS and GSAP for responsive, animated UI components.

**Impact & Future Work**:
The platform demonstrates the feasibility of automated financial advisory systems using unsupervised learning, though current limitations include synthetic training data and lack of formal evaluation metrics. Future research directions include: (1) validating the clustering approach against expert-labeled risk profiles, (2) incorporating additional features (debt-to-income ratio, investment history), (3) implementing supervised learning models (Random Forest, XGBoost) for risk prediction, (4) backtesting recommendation performance, and (5) integrating sentiment analysis of financial news to enhance fraud detection capabilities.

---

## 6. SUGGESTED IMPROVEMENTS (Research-Oriented, Achievable)

### 1. **Add Model Evaluation Metrics** (2-3 hours)
**What**: Implement clustering validation metrics to assess model quality
**How**: 
- Add silhouette score calculation: `from sklearn.metrics import silhouette_score`
- Calculate within-cluster sum of squares (inertia)
- Visualize clusters using PCA/t-SNE (2D projection)
- Add these metrics to a model evaluation script
**Why**: Demonstrates understanding of ML evaluation and provides quantitative justification for the clustering approach

### 2. **Implement Feature Engineering** (3-4 hours)
**What**: Create derived financial features that better capture risk
**How**:
- Calculate: savings_rate = savings/income, expense_ratio = expenses/income, years_to_retirement = 65 - age
- Add these engineered features to the clustering model
- Compare model performance with/without engineered features
**Why**: Shows domain knowledge and understanding of feature engineering importance in ML

### 3. **Add Model Persistence & Loading** (1-2 hours)
**What**: Fix the current bug where models retrain on every import
**How**:
- Modify `predict_risk()` to load models from pickle files: `joblib.load("kmeans_risk_model.pkl")`
- Add error handling for missing model files
- Create a separate training script that runs once to generate models
**Why**: Professional code quality, prevents unnecessary retraining, demonstrates software engineering best practices

### 4. **Implement Basic Backtesting** (4-5 hours)
**What**: Track performance of recommended stocks over time
**How**:
- Store user predictions with timestamps in Django models
- Create a background task (or manual script) that checks stock performance after 1/3/6 months
- Calculate accuracy metrics: % of recommendations that outperformed market average
- Display backtesting results on a dashboard
**Why**: Adds validation of the recommendation system, shows research-oriented thinking about model performance

### 5. **Add Comparative Analysis Section** (2-3 hours)
**What**: Compare K-Means with alternative approaches
**How**:
- Implement a simple supervised baseline (e.g., Logistic Regression with synthetic labels: risk_score = f(income, savings, age))
- Compare clustering results with the supervised approach
- Add a section in README or documentation explaining trade-offs
**Why**: Demonstrates critical thinking about algorithm selection and understanding of ML methodology

### **Bonus: Quick Win - Enhanced README** (30 minutes)
- Add installation instructions
- Document the API endpoints
- Explain the model architecture
- Include screenshots of the UI
- Add a "Future Work" section referencing the improvements above

---

## SUMMARY OF KEY FINDINGS

**Project Type**: Full-stack web application with ML components
**Core Technology**: Django + scikit-learn + yfinance
**ML Approach**: Unsupervised learning (K-Means) for risk profiling
**Data**: Synthetic training data, real-time stock data via API
**Originality**: Custom business logic and integration work; ML implementation is standard
**Research Readiness**: Functional but needs evaluation metrics and validation
**Time to Improve**: 12-17 hours total for all 5 improvements

**Recommendation**: Focus on improvements #1 (evaluation metrics) and #3 (model persistence) first, as they're quick wins that significantly improve the project's research credibility.

