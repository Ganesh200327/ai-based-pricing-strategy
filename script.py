import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
file_path = "C:\\Users\\cmadh\\Downloads\\final_cleaned_electronics_dataset.csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = ["Product Name", "Brand", "Category", "Price (INR)"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Remove outliers in Price
df = df[(df["Price (INR)"] > 500) & (df["Price (INR)"] < 100000)]

# Preserve original price before scaling
df["Price_Original"] = df["Price (INR)"].copy()

# Convert Price to numeric
df["Price (INR)"] = pd.to_numeric(df["Price (INR)"], errors="coerce")

# Encode categorical features
encoders = {}
for col in ["Brand", "Category", "Product Name"]:
    df[col] = df[col].astype(str).fillna("Unknown")
    encoders[col] = LabelEncoder()
    df[col + "_Encoded"] = encoders[col].fit_transform(df[col])

# Normalize Price
scaler = MinMaxScaler()
df["Price (INR)"] = scaler.fit_transform(df[["Price (INR)"]])

# Define features and target
features = ["Brand_Encoded", "Category_Encoded", "Product Name_Encoded"]
target = "Price (INR)"
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Function to encode unseen values
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Assign default value for unseen data (mean value)
        return int(np.mean(encoder.transform(encoder.classes_)))

# Function to predict optimal price
def predict_optimal_price(brand, category, product_name):
    try:
        # Check if product exists in the dataset
        product_data = df[(df["Brand"] == brand) &
                          (df["Category"] == category) &
                          (df["Product Name"] == product_name)]

        if not product_data.empty:
            # Use the actual price if product exists
            avg_price = product_data["Price_Original"].mean()
            category_avg = df[df["Category"] == category]["Price_Original"].mean()

            # Handle NaN values in category average
            if pd.isna(category_avg):
                category_avg = avg_price

            # Compute discount based on category average price
            discount = ((category_avg - avg_price) / category_avg) * 100
            discount = max(0, min(50, round(discount)))

            return round(avg_price, 2), discount
        else:
            # Encode values for unseen inputs
            brand_encoded = safe_encode(encoders["Brand"], brand)
            category_encoded = safe_encode(encoders["Category"], category)
            product_encoded = safe_encode(encoders["Product Name"], product_name)

            if brand_encoded == -1 or category_encoded == -1 or product_encoded == -1:
                return "❌ Invalid input. Please check the values."

            # Predict price
            predicted_price = model.predict([[brand_encoded, category_encoded, product_encoded]])[0]
            predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

            # Handle NaN values for category average
            category_avg = df[df["Category"] == category]["Price_Original"].mean()
            if pd.isna(category_avg):
                category_avg = predicted_price

            # Compute discount based on category average price
            discount = ((category_avg - predicted_price) / category_avg) * 100
            discount = max(0, min(50, round(discount)))

            return round(predicted_price, 2), discount

    except Exception as e:
        return f"Error: {str(e)}"

# User input and display
def main():
    print("\n🔥 AI Pricing Strategy System 🔥")

    available_brands = sorted(df["Brand"].unique())
    available_categories = sorted(df["Category"].unique())

    print("\n📌 Available Categories:", available_categories)
    print("📌 Available Brands:", available_brands)

    category = input("\nEnter product category: ").strip()
    brand = input("Enter product brand: ").strip()

    available_models = df[df["Brand"] == brand]["Product Name"].unique()
    if len(available_models) == 0:
        print("\n❌ No models found for this brand.")
        return

    print("\n📌 Available Models:", list(available_models))
    product_name = input("Enter product model: ").strip()

    result = predict_optimal_price(brand, category, product_name)

    if isinstance(result, tuple):
        price, discount = result
        print(f"\n💰 Suggested Price for {product_name}: ₹{price}")
        print(f"🔻 Recommended Discount: {discount}%")
        print(f"📉 Final Price After Discount: ₹{price * (1 - discount / 100):.2f}")
    else:
        print(result)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model for prediction
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

if __name__ == "__main__":
    main()
