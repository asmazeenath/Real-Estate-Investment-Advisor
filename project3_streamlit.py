import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("india_housing_prices cleaned.csv")

clf_model = joblib.load("xgb_classification_model.pkl")
reg_model = joblib.load("rf_future_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Introduction", "EDA Visualizations", "Prediction"])


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

elif page == "EDA Visualizations":

    st.title("📊 EDA Visualizations")

    # Size Distribution
    Query=st.selectbox("Select the Analysis",[
        
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
        Result=None

        if Query=="1.What is the distribution of property prices?":
                    fig, ax = plt.subplots()
                    sns.histplot(df['Price_in_Lakhs'], bins=30, kde=True)
                    plt.title('Distribution of Price')
                    st.pyplot(fig)
            
        elif Query=="2.What is the distribution of property sizes?":
                    fig, ax = plt.subplots()
                    plt.boxplot(df["Size_in_SqFt"])
                    plt.xlabel("No of distribution")
                    st.pyplot(fig)
        elif Query== "3.How does the price per sq ft vary by property type?":
                    fig, ax = plt.subplots()
                    variation=df.groupby('Property_Type')['Price_per_SqFt'].mean()
                    plt.bar(variation.index.astype(str),variation.values)
                    plt.xlabel('Property_Type')
                    plt.ylabel('Average Price_per_SqFt')
                    st.pyplot(fig)
            
          
        elif Query=="4.Is there a relationship between property type?":
            
                    fig, ax = plt.subplots()
                    sample_df=df.sample(1000)
                    plt.bar(sample_df['Size_in_SqFt'],sample_df['Price_in_Lakhs'])
                    plt.xlabel('size in sqft')
                    plt.ylabel('price in lakhs')
                    plt.title('size vs price')
                    st.pyplot(fig)
        elif Query=="5.Are there any outliers in price per sq ft or property size?":
                    fig, ax = plt.subplots()
                    corr_value = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
                    plt.bar(['Size vs Price'], [corr_value])
                    plt.ylim(-1, 1)
                    plt.title("Correlation between Size and Price")
                    st.pyplot(fig)
            
        elif Query=="6.What is the average price per sq ft by state?":
                    fig, ax = plt.subplots()
                    average =df.groupby('State')['Price_per_SqFt'].mean()
                    plt.barh(average.index.astype(str),average.values)
                    plt.xlabel('Average price per sqft')
                    plt.ylabel('state')
                    st.pyplot(fig)
        elif Query=="7.What is the average property price by city?":  
                    fig, ax = plt.subplots(figsize=(10,6))
                    average=df.groupby('City')['Price_per_SqFt'].mean()
                    plt.barh(average.index.astype(str),average.values)
                    plt.xlabel('price per property')
                    plt.ylabel('city')
                    st.pyplot(fig)
        elif Query=="8.What is the median age of properties by locality?":

                    top_cities = df["City"].value_counts().head(5).index
                    filtered_df = df[df["City"].isin(top_cities)]

                    fig, ax = plt.subplots(figsize=(10,6))
                    filtered_df.boxplot(
                    column="Price_in_Lakhs",
                    by="City",
                    ax=ax)

                    ax.set_title("Property Price Distribution (Top 5 Cities)")
                    ax.set_xlabel("City")
                    ax.set_ylabel("Price in Lakhs")

                    plt.suptitle("") 
                    st.pyplot(fig)
        elif Query=="9.How is BHK distributed across cities?":
                    fig, ax = plt.subplots()
                    top_cities = df["City"].value_counts().head(5).index
                    filtered_df = df[df["City"].isin(top_cities)]
                    data = [filtered_df[filtered_df["City"] == city]["BHK"] for city in top_cities]
                    plt.violinplot(data, showmeans=True)
                    plt.xlabel("City")
                    plt.ylabel("BHK Distribution")
                    plt.title("BHK Distribution Across Top 5 Cities")
                    st.pyplot(fig)
        elif Query=="10.What are the price trends for the top 5 most expensive localities?":
                    fig, ax = plt.subplots()
                    top_localities = (
                    df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).index)
                    filtered_df = df[df["Locality"].isin(top_localities)]
                    data = [filtered_df[filtered_df["Locality"] == loc]["Price_in_Lakhs"] 
                    for loc in top_localities]
                    plt.violinplot(data, showmeans=True)
                    plt.xticks(range(1, len(top_localities)+1), top_localities, rotation=30)
                    plt.xlabel("Locality")
                    plt.ylabel("Price in Lakhs")
                    plt.title("Price Distribution in Top 5 Expensive Localities")
                    st.pyplot(fig)
        elif Query=="11.How are numeric features correlated with each other?":
                    fig, ax = plt.subplots()
                    numeric=df.select_dtypes(include=['int64','float64'])
                    corr=numeric.corr()
                    sns.heatmap(corr,annot=True)
                    st.pyplot(fig)

        elif Query=="12.How do nearby schools relate to price per sq ft?":
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='Nearby_Schools',y='Price_per_SqFt',data=df)
                    st.pyplot(fig)
        elif Query=="13.How do nearby hospitals relate to price per sq ft?":
                    fig, ax = plt.subplots()
                    sns.scatterplot(x='Nearby_Hospitals',y='Price_per_SqFt',data=df)
                    st.pyplot(fig)  
        elif Query=="14.How does price vary by furnished status?":
                    fig, ax = plt.subplots()
                    sns.boxplot(x='Furnished_Status',y='Price_in_Lakhs',data=df)
                    st.pyplot(fig)
        elif Query=="15.How does price per sq ft vary by property facing direction?":
                    fig, ax = plt.subplots()
                    sns.boxplot(x='Facing',y='Price_per_SqFt',data=df)
                    st.pyplot(fig)
            
        elif Query=="16.How many properties belong to each owner type?":
                    fig, ax = plt.subplots()
                    sns.countplot(x='Owner_Type',data=df)
                    st.pyplot(fig)
        elif Query=="17.How many properties are available under each availability status?":
                    fig, ax = plt.subplots()
                    sns.countplot(x='Availability_Status',data=df)
                    st.pyplot(fig)
            
        elif Query=="18.Does parking space affect property price?":
                   fig, ax = plt.subplots()
                   sns.boxplot(x='Parking_Space', y='Price_in_Lakhs', data=df, ax=ax)
                   st.pyplot(fig)
        elif Query=="19.How do amenities affect price per sq ft?":
                   fig,ax= plt.subplots()
                   df['Amenities_Count'] = df['Amenities'].apply(lambda x: len(str(x).split(',')))
                   sns.scatterplot(x='Amenities_Count',y='Price_per_SqFt',data=df)
                   st.pyplot(fig)
        elif Query=="20.How does public transport accessibility relate to price per sq ft or investment potential?":
                   fig,ax= plt.subplots()
                   sns.violinplot(x='Public_Transport_Accessibility',y='Price_per_SqFt',data=df)
                   st.pyplot(fig)

            

elif page == "Prediction":

    st.title("🏠 Property Investment Prediction")
    st.write("Fill out the form below to view investment classification and price forecast.")

    with st.form("prediction_form"):

    
        city = st.selectbox("City", df["City"].unique())
        property_type = st.selectbox("Property Type", df["Property_Type"].unique())

        bhk = st.number_input("BHK", min_value=1)
        size = st.number_input("Size_in_SqFt", min_value=100)
        price = st.number_input("Price_in_Lakhs", min_value=1)

        Nearby_schools = st.number_input("Nearby_Schools", min_value=0)
        Nearby_hospitals = st.number_input("Nearby_Hospitals", min_value=0)
        transport = st.slider("Public_Transport_Accessibility", 1, 10)
        parking_Space = st.number_input("Parking_Space", min_value=0)


        submit = st.form_submit_button("Predict")

    if submit:

        # Create input dataframe
        input_data = pd.DataFrame([{
        
            "City": city,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Price_in_Lakhs": price,
            "Nearby_Schools": Nearby_schools,
            "Nearby_Hospitals": Nearby_hospitals,
            "Public_Transport_Accessibility": transport,
            "Parking_Space": parking_Space,
            
        }])

        # Convert categorical to numeric
        input_data = pd.get_dummies(input_data)

        # Match training columns
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Classification prediction
        class_pred = clf_model.predict(input_data)[0]
        class_prob = clf_model.predict_proba(input_data)[0][1]

        # Regression prediction
        future_price = reg_model.predict(input_data)[0]

        st.subheader("📊 Results")

        if class_pred == 1:
            st.success("✅ Good Investment")
        else:
            st.error("❌ Not a Good Investment")

        st.write("Confidence Score:", round(class_prob * 100, 2), "%")

        # Future price section
        st.subheader("💰 Estimated Future Price (5 Years)")
        st.title(f"₹ {round(future_price,2)} Lakhs")

        # Growth calculation
        growth = ((future_price - price) / price) * 100
        growth = round(growth, 2)

        st.write("📈 Price Growth:", growth, "%")

        st.caption("Calculation assumed approx. 8% growth rate unless ML model is applied")