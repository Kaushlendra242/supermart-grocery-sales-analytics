import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ============================================
# 1. PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Supermart Grocery Sales Prediction",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Supermart Grocery Sales Prediction (CatBoost Model)")
st.write(
    """
This app uses a trained **CatBoost regression model** to predict **Sales**  
for a given grocery order, based on product, location, discount and date.
"""
)

# ============================================
# 2. LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load("supermart_catboost_sales_model.pkl")
    return model

model = load_model()

# ============================================
# 3. OPTIONAL: LOAD DATA FOR DROPDOWN OPTIONS
#    (If CSV not found, we fall back to default lists)
# ============================================

def load_reference_data():
    try:
        df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")
        return df
    except Exception:
        return None

ref_df = load_reference_data()

if ref_df is not None:
    categories = sorted(ref_df["Category"].dropna().unique().tolist())
    sub_categories = sorted(ref_df["Sub Category"].dropna().unique().tolist())
    cities = sorted(ref_df["City"].dropna().unique().tolist())
    regions = sorted(ref_df["Region"].dropna().unique().tolist())
    states = sorted(ref_df["State"].dropna().unique().tolist())
else:
    # Fallback values (update with real ones if needed)
    categories = [
        "Bakery", "Beverages", "Eggs, Meat & Fish", "Food Grains",
        "Fruits & Veggies", "Oil & Masala", "Snacks"
    ]
    sub_categories = ["Atta & Flour", "Biscuits", "Cold Drinks", "Namkeen"]
    cities = ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli"]
    regions = ["North", "South", "East", "West"]
    states = ["Tamil Nadu", "Karnataka", "Kerala", "Telangana"]

# ============================================
# 4. FEATURE PREPARATION FUNCTION
#    (Must match training-time logic)
# ============================================

def prepare_features_from_order(order_dict):
    """
    order_dict: dictionary with keys:
      Category, Sub Category, City, Region, State,
      Discount, Profit, Order_Date (YYYY-MM-DD string),
      optional: Sales (only for Profit_Margin)
    """
    temp = order_dict.copy()
    
    # Parse date
    order_date = pd.to_datetime(temp["Order_Date"])
    
    # Date features
    temp["Order_Day"] = order_date.day
    temp["Order_Month"] = order_date.month
    temp["Order_Year"] = order_date.year
    temp["Month"] = order_date.month_name()
    
    # Basic numeric features
    discount = float(temp.get("Discount", 0.0))
    profit = float(temp.get("Profit", 0.0))
    sales = float(temp.get("Sales", 0.0))  # optional, if known
    
    # Engineered features (must match training)
    if sales > 0:
        temp["Profit_Margin"] = profit / (sales + 1e-6)
    else:
        temp["Profit_Margin"] = 0.0

    temp["Is_Weekend"] = int(order_date.dayofweek in [5, 6])  # 5 = Sat, 6 = Sun
    temp["Discount_Impact"] = discount * profit
    temp["Profit_to_Discount"] = profit / (discount + 1e-6)
    
    # Build final feature row
    row = {
        # categorical
        "Category": str(temp["Category"]),
        "Sub Category": str(temp["Sub Category"]),
        "City": str(temp["City"]),
        "Region": str(temp["Region"]),
        "State": str(temp["State"]),
        "Month": str(temp["Month"]),
        # numeric
        "Discount": discount,
        "Profit": profit,
        "Order_Day": temp["Order_Day"],
        "Order_Month": temp["Order_Month"],
        "Order_Year": temp["Order_Year"],
        "Profit_Margin": temp["Profit_Margin"],
        "Is_Weekend": temp["Is_Weekend"],
        "Discount_Impact": temp["Discount_Impact"],
        "Profit_to_Discount": temp["Profit_to_Discount"],
    }
    
    return pd.DataFrame([row])

# ============================================
# 5. SIDEBAR INPUTS
# ============================================

st.sidebar.header("Enter Order Details")

category = st.sidebar.selectbox("Category", categories)
sub_category = st.sidebar.selectbox("Sub Category", sub_categories)
city = st.sidebar.selectbox("City", cities)
region = st.sidebar.selectbox("Region", regions)
state = st.sidebar.selectbox("State", states)

col1, col2 = st.sidebar.columns(2)
with col1:
    discount = st.number_input(
        "Discount (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )
with col2:
    profit = st.number_input(
        "Profit (₹)",
        value=200.0,
        step=10.0
    )

order_date = st.sidebar.date_input(
    "Order Date",
    value=datetime(2019, 8, 15)
)

# Optional: sales (only for Profit_Margin; keep 0 if unknown)
sales_optional = st.sidebar.number_input(
    "Known Sales (optional, only for more accurate Profit Margin)",
    value=0.0,
    step=50.0
)

# ============================================
# 6. PREDICTION
# ============================================

if st.button("🔮 Predict Sales"):
    # Prepare dictionary
    order_dict = {
        "Category": category,
        "Sub Category": sub_category,
        "City": city,
        "Region": region,
        "State": state,
        "Discount": discount,
        "Profit": profit,
        "Order_Date": order_date.strftime("%Y-%m-%d"),
        "Sales": sales_optional  # optional; 0 if unknown
    }

    # Prepare features
    features_df = prepare_features_from_order(order_dict)

    # Predict (model outputs log(Sales + 1))
    log_sales_pred = model.predict(features_df)[0]
    sales_pred = np.expm1(log_sales_pred)

    st.subheader("Predicted Sales")
    st.success(f"₹ {sales_pred:,.2f}")

    # Show the final feature vector (for learning / debugging)
    with st.expander("Show model input features"):
        st.write(features_df)

# ============================================
# 7. FOOTER
# ============================================
st.markdown("---")
st.caption(
    "Built with CatBoost & Streamlit • Supermart Grocery Sales Prediction • "
    "You can extend this app with authentication, logging, and database integration."
)
