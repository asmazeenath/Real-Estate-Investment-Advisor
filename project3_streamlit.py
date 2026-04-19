import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD FILES ---------------- #
df = pd.read_csv("C:/Users/DELL XPS/OneDrive/Desktop/AIML/project_3/data.py/india_housing_prices (3).csv")
model = joblib.load("model1.pkl")
scaler = joblib.load("scaler1.pkl")
columns = joblib.load("columns1.pkl")

# ---------------- UI ---------------- #
st.set_page_config(page_title="Real Estate Predictor", layout="centered")

st.title("🏡 Real Estate Price Predictor")
st.write("Predict property price based on features")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Introduction", "EDA Visualizations", "Prediction"])


# ---------------- INTRODUCTION ---------------- #
if page == "Introduction":
    st.title("🏠 Property Investment Analyzer")
    st.write("""
    Buying a property is a big financial decision.
             
     Many people feel confused about whether a property is worth investing in or if its value will increase in the future. This website helps you make that decision easily and confidently.""")
    st.subheader("💡 What You Can Do Here")
    st.write("""✔ Enter property details like size, price, BHK, and location
             
✔ Check if the property is a Good Investment or Not
             
✔ See the Estimated Price After 5 Years
             
✔ View simple charts to understand market trends
             
✔ Get a confidence score to know how reliable the prediction is
Our system uses smart data analysis and Machine Learning to give you clear insights, so you can invest wisely and reduce financial risk.
    """)


# ---------------- EDA ---------------- #
elif page == "EDA Visualizations":

    st.title("📊 EDA Visualizations")

    Query = st.selectbox("Select the Analysis",[
        "1.What is the distribution of property prices?",
        "2.What is the distribution of property sizes?",
        "3.How does the price per sq ft vary by property type?",
        "4.Is there a relationship between property type?",
        "5.Are there any outliers in price per sq ft or property size?",
        "6.What is the average price per sq ft by state?",
        "7.What is the average property price by city?",
        "8.What is the median age of properties by locality?",
        "9.How is BHK distributed across cities?",
        "10.What are the price trends for the top 5 most expensive localities?",
        "11.How are numeric features correlated with each other?",
        "12.How do nearby schools relate to price per sq ft?",
        "13.How do nearby hospitals relate to price per sq ft?",
        "14.How does price vary by furnished status?",
        "15.How does price per sq ft vary by property facing direction?",
        "16.How many properties belong to each owner type?",
        "17.How many properties are available under each availability status?",
        "18.Does parking space affect property price?",
        "19.How do amenities affect price per sq ft?",
        "20.How does public transport accessibility relate to price per sq ft or investment potential?"
    ])

    if st.button("Run Analysis"):

        if Query == "1.What is the distribution of property prices?":
            fig, ax = plt.subplots()
            sns.histplot(df['Price_in_Lakhs'], bins=30, kde=True)
            plt.title('Distribution of Price')
            st.pyplot(fig)

        elif Query == "2.What is the distribution of property sizes?":
            fig, ax = plt.subplots()
            plt.boxplot(df["Size_in_SqFt"])
            plt.xlabel("No of distribution")
            st.pyplot(fig)

        elif Query == "3.How does the price per sq ft vary by property type?":
            fig, ax = plt.subplots()
            variation = df.groupby('Property_Type')['Price_per_SqFt'].mean()
            plt.bar(variation.index.astype(str), variation.values)
            plt.xlabel('Property_Type')
            plt.ylabel('Average Price_per_SqFt')
            st.pyplot(fig)

        elif Query == "4.Is there a relationship between property type?":
            fig, ax = plt.subplots()
            sample_df = df.sample(1000)
            plt.bar(sample_df['Size_in_SqFt'], sample_df['Price_in_Lakhs'])
            plt.xlabel('size in sqft')
            plt.ylabel('price in lakhs')
            plt.title('size vs price')
            st.pyplot(fig)

        elif Query == "5.Are there any outliers in price per sq ft or property size?":
            fig, ax = plt.subplots()
            corr_value = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
            plt.bar(['Size vs Price'], [corr_value])
            plt.ylim(-1, 1)
            plt.title("Correlation between Size and Price")
            st.pyplot(fig)

        elif Query == "6.What is the average price per sq ft by state?":
            fig, ax = plt.subplots()
            average = df.groupby('State')['Price_per_SqFt'].mean()
            plt.barh(average.index.astype(str), average.values)
            plt.xlabel('Average price per sqft')
            plt.ylabel('state')
            st.pyplot(fig)

        elif Query == "7.What is the average property price by city?":
            fig, ax = plt.subplots(figsize=(10,6))
            average = df.groupby('City')['Price_per_SqFt'].mean()
            plt.barh(average.index.astype(str), average.values)
            plt.xlabel('price per property')
            plt.ylabel('city')
            st.pyplot(fig)

        elif Query == "8.What is the median age of properties by locality?":
            top_cities = df["City"].value_counts().head(5).index
            filtered_df = df[df["City"].isin(top_cities)]

            fig, ax = plt.subplots(figsize=(10,6))
            filtered_df.boxplot(column="Price_in_Lakhs", by="City", ax=ax)

            ax.set_title("Property Price Distribution (Top 5 Cities)")
            ax.set_xlabel("City")
            ax.set_ylabel("Price in Lakhs")
            plt.suptitle("")
            st.pyplot(fig)

        elif Query == "9.How is BHK distributed across cities?":
            fig, ax = plt.subplots()
            top_cities = df["City"].value_counts().head(5).index
            filtered_df = df[df["City"].isin(top_cities)]
            data = [filtered_df[filtered_df["City"] == city]["BHK"] for city in top_cities]
            plt.violinplot(data, showmeans=True)
            plt.xlabel("City")
            plt.ylabel("BHK Distribution")
            plt.title("BHK Distribution Across Top 5 Cities")
            st.pyplot(fig)

        elif Query == "10.What are the price trends for the top 5 most expensive localities?":
            fig, ax = plt.subplots()
            top_localities = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).index
            filtered_df = df[df["Locality"].isin(top_localities)]
            data = [filtered_df[filtered_df["Locality"] == loc]["Price_in_Lakhs"] for loc in top_localities]
            plt.violinplot(data, showmeans=True)
            plt.xticks(range(1, len(top_localities)+1), top_localities, rotation=30)
            plt.xlabel("Locality")
            plt.ylabel("Price in Lakhs")
            plt.title("Price Distribution in Top 5 Expensive Localities")
            st.pyplot(fig)

        elif Query == "11.How are numeric features correlated with each other?":
            fig, ax = plt.subplots()
            numeric = df.select_dtypes(include=['int64','float64'])
            corr = numeric.corr()
            sns.heatmap(corr, annot=True)
            st.pyplot(fig)


# ---------------- PREDICTION ---------------- #
elif page == "Prediction":

    st.sidebar.header("Enter Property Details")

    sqft = st.sidebar.number_input("Size (SqFt)", 300, 10000, 1000)
    bhk = st.sidebar.number_input("BHK", 1, 10, 2)
    floor = st.sidebar.number_input("Floor No", 0, 50, 1)
    age = st.sidebar.number_input("Age of Property", 0, 50, 5)

    schools = st.sidebar.slider("Nearby Schools", 0, 10, 2)
    hospitals = st.sidebar.slider("Nearby Hospitals", 0, 10, 2)

    city = st.sidebar.selectbox("City", ["Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad"])
    property_type = st.sidebar.selectbox("Property Type", ["Apartment", "Villa", "Independent House"])
    parking = st.sidebar.selectbox("Parking", ["Yes", "No"])
    security = st.sidebar.selectbox("Security", ["Low", "Medium", "High"])

    input_dict = {
        'Size_in_SqFt': sqft,
        'BHK': bhk,
        'Floor_No': floor,
        'Age_of_Property': age,
        'Nearby_Schools': schools,
        'Nearby_Hospitals': hospitals
    }

    input_df = pd.DataFrame([input_dict])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    if f"City_{city}" in input_df.columns:
        input_df[f"City_{city}"] = 1

    if f"Property_Type_{property_type}" in input_df.columns:
        input_df[f"Property_Type_{property_type}"] = 1

    if f"Parking_Space_{parking}" in input_df.columns:
        input_df[f"Parking_Space_{parking}"] = 1

    if f"Security_{security}" in input_df.columns:
        input_df[f"Security_{security}"] = 1

    input_df = input_df[columns]

    if st.button("Predict Price 💰"):
        try:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            st.success(f" Estimated Price: ₹ {prediction:.2f} Lakhs")

            if prediction < 70:
                st.success(" Good Investment ✅")
                st.write("This property is relatively low-priced → higher return potential.")

            elif prediction < 150:
                st.info(" Average Investment")
                st.write("Moderate pricing → stable but not high returns.")

            else:
                st.warning("⚠️ Risky / Overpriced Investment")
                st.write("High price → lower ROI and higher risk.")

        except Exception as e:
            st.error(f"Error: {e}")


        st.caption("Calculation assumed approx. 8% growth rate unless ML model is applied")
