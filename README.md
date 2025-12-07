# 🛒 **Supermart Grocery Sales – Retail Analytics & Machine Learning Project**

This repository contains a complete **end-to-end Data Science project** built using the  
**Supermart Grocery Sales – Retail Analytics Dataset**.  
The project includes **EDA**, **Feature Engineering**, **ML Modeling**, **CatBoost Optimization**,  
and a complete **Streamlit Deployment App**.

---

## 📌 **Project Overview**

This project analyzes a fictional grocery sales dataset from **Tamil Nadu, India**.  
The goal is to:

- Understand **sales performance** across categories, cities, months, and years  
- Engineer advanced features to improve predictive power  
- Build and evaluate multiple ML models  
- Deploy the final model using a **Streamlit application**

Dataset Reference:  
Training dataset includes:  
**Category, Sub Category, City, Region, Sales, Discount, Profit, Order Date, State, Month, Year**  
*(Full dataset description available in the included PDF)*

---

## 📁 **Project Structure**

```plaintext
supermart-grocery-sales-analytics
│
├── data/
│   └── Supermart Grocery Sales - Retail Analytics Dataset.csv
│
├── notebooks/
│   └── Supermart_Grocery_Sales-Retail_Analytics_Dataset.ipynb
│
├── reports/
│   ├── Supermart_Project_Summary.pdf
│   └── Supermart_Project_Summary.docx
│
├── model/
│   └── supermart_catboost_sales_model.pkl
│
├── app/
│   └── app.py        # Streamlit Application
│
├── requirements.txt
└── README.md




```


**📊 Key Exploratory Data Insights (EDA)**

✔ Category-wise Sales

- Eggs, Meat & Fish had the highest sales contribution (~15%)

✔ Monthly Sales Trend

- Clear upward trend across months showing business improvement

✔ Yearly Sales

- Sales increased significantly from 2016 → 2018

✔ Top Performing Cities

- Kanyakumari

- Vellore

- Bodi

- Tirunelveli

- Perambalur
------


**🔧 Feature Engineering**

The following features were engineered to boost model performance:

**🗓 Date Features**

- Order_Day

- Order_Month

- Order_Year

- Month_Name

- Is_Weekend

**💼 Business Features**

- Profit_Margin = Profit / Sales

- Discount_Impact = Discount × Profit

- Profit_to_Discount = Profit / Discount

- Outlier removal (1% top and bottom)

**🎯 Target Transformation**

* Log_Sales = log1p(Sales) applied for stable model training

  -------

**🤖 Modeling & Performance**

Multiple models were trained and evaluated.

| Model                                  | MAE         | RMSE        | R² Score  |
| -------------------------------------- | ----------- | ----------- | --------- |
| **Linear Regression**                  | 382.67      | 463.27      | 0.3584    |
| **Random Forest**                      | 387.32      | 472.67      | 0.3321    |
| ⭐ **CatBoost Regressor (Final Model)** | **200–260** | **300–350** | **0.60+** |

**📌 Notes**

CatBoost outperformed all baseline models due to its ability to handle:

- Categorical features

- Non-linear relationships

- Complex feature interactions

**Final model saved as:**

model/supermart_catboost_sales_model.pkl

------

**🚀 Streamlit Application**

https://share.streamlit.io/

A full Streamlit app is included to run real-time predictions.

**▶ Run locally:**

streamlit run app/app.py


The app allows users to:

- Select product category

- Choose city, region, state

- Specify discount, profit, order date

- Get predicted Sales instantly

---------


**📦 Installation**

Install all dependencies:

pip install -r requirements.txt

---------

**🔮 Business Value Delivered**

✔ Accurate sales forecasting

✔ Data-driven discount & promotion planning

✔ Optimized inventory management

✔ Clear insights for regional sales strategy

✔ Deployable prediction app for real-time usage

---------

**🏅 Tools & Technologies**

- Python

- Pandas

- NumPy

- Matplotlib / Seaborn

- Scikit-Learn

- CatBoost

- Streamlit

- Jupyter Notebook

- ReportLab / DOCX (Report Generation)

--------------

**📝 Project Authors**

*Kaushlendra Pratap Singh*
Data Analyst & Machine Learning& Data Scienist Practitioner

------------

**⭐ Support the Project**

If you like this project, please ⭐ star the repository!
