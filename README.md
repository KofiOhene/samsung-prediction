# **Samsung Revenue Prediction and Key Business Drivers Analysis**
### **Applying Machine Learning to Forecast Revenue and Identify Key Factors**

## **Project Overview**
This project aims to analyze Samsung smartphone sales data using machine learning techniques to predict revenue and extract meaningful business insights. The analysis focuses on identifying the most significant factors influencing revenue, such as **5G adoption, market share, regional sales, and product model trends**. 

### **Objectives**
- Develop **supervised machine learning models** to predict smartphone revenue.
- Analyze key drivers of revenue using feature importance methods.
- Compare multiple regression models to determine the most effective approach.

---

## **Methodology**
### **1. Data Collection & Preprocessing**
- The dataset consists of **smartphone sales data from 2019 to 2024**, covering **various product models, regions, and technical specifications**.
- **Data Cleaning:** Addressed missing values, removed inconsistencies, and ensured categorical values were standardized.
- **Feature Engineering:** Encoded categorical variables, introduced interaction terms, and scaled numerical data.

### **2. Model Training & Evaluation**
- Implemented the following machine learning models:
  - **Linear Regression** (Baseline Model)
  - **Decision Tree Regressor**
  - **Random Forest Regressor**
  - **XGBoost Regressor** (Best-performing model)
- Evaluation Metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R² Score**

### **3. Feature Importance Analysis**
- Identified and ranked the most significant revenue-driving factors using **Decision Tree, Random Forest, and XGBoost** models.
- Examined the impact of **5G adoption, sales volume, market share, and regional variations**.

---

## **Model Performance Comparison**
| Model              | MAE          | MSE           | R² Score |
|--------------------|-------------|--------------|----------|
| **Linear Regression** | 13,726,762 | 281,161,478,028,605 | 0.0479 |
| **Decision Tree** | 2,097,802 | 53,210,326,160,618 | 0.8198 |
| **Random Forest** | 4,358,664 | 45,202,652,343,150 | 0.8469 |
| **XGBoost** | **1,856,888** | **36,711,691,788,135** | **0.8757** |

### **Key Observations**
- **XGBoost outperformed all other models**, achieving the highest **R² score (0.8757)** and the lowest **MAE (1.85M USD)**.
- **Decision Tree performed well** but exhibited greater variance compared to Random Forest.
- **Linear Regression did not perform well**, suggesting revenue is influenced by non-linear relationships.

---

## **Key Business Insights**
Each model identified the **most significant factors impacting revenue**.

| Feature | Decision Tree | Random Forest | XGBoost |
|---------|--------------|---------------|---------|
| **Units Sold** | 15.6% | 14.1% | Not in top 10 |
| **Market Share (%)** | 8.7% | 12.2% | Not in top 10 |
| **Preference for 5G (%)** | 14.7% | 12.3% | 3.0% |
| **Avg 5G Speed (Mbps)** | 11.5% | 10.8% | Not in top 10 |
| **5G Subscribers (millions)** | 13.6% | 12.1% | Not in top 10 |
| **5G Capability (Yes/No)** | 7.2% | 5.4% | **43.3% (Most Important!)** |
| **Product Model (Galaxy S23, Z Fold3, etc.)** | Not in top 10 | Not in top 10 | **Dominates XGBoost rankings** |

### **Strategic Recommendations**
1. **5G Capability is the most significant revenue driver**  
   - Models with **5G connectivity** generate substantially higher revenue.
   - **Investment in 5G technology and marketing should be a priority**.

2. **Units Sold & Market Share remain fundamental revenue drivers**  
   - Marketing campaigns should focus on increasing **sales volume** and **expanding market share**.

3. **Product Model & Regional Variations are key considerations**  
   - **Galaxy S23, Z Fold3, and Z Flip3** models significantly contribute to revenue.
   - **North America and Latin America** are key revenue-generating regions.

---

## **How to Run the Project**
### **1. Environment Setup**
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/samsung-revenue-prediction.git

# Navigate into the project folder
cd samsung-revenue-prediction

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### **2. Run the Model Training Pipeline**
```bash
python main.py
```
This script will:
- Load and clean the dataset
- Perform feature engineering
- Train multiple machine learning models and compare their performance
- Save results to `models/model_comparison.csv`

### **3. View Model Results**
- Model predictions and evaluation metrics are stored in:
  - `models/`
  - `models/model_comparison.csv`

### **4. Generate Visualizations**
To generate feature importance and model comparison visualizations:
```bash
python src/visualization.py
```

---

## **Future Work and Improvements**
1. **Expanding the dataset** to include additional features such as **pricing strategies, customer sentiment analysis, and marketing spend**.
2. **Hyperparameter tuning** of XGBoost to further optimize performance.
3. **Developing a classification model** to predict **whether a product will be a top seller**.

---

## **Contributors**
- **Kevin Manu**  
- Contact: mansaohene@gmail.com

---

## **License**
This project is licensed under the **MIT License**.  

---

## **Final Notes**
This project provides **a data-driven approach to revenue forecasting** and identifies **key business drivers** in Samsung's smartphone sales. By leveraging machine learning, particularly **XGBoost**, the analysis offers valuable insights into **5G adoption, regional demand, and product-level performance**, assisting in strategic decision-making.

