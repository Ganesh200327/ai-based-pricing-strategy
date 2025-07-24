from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load dataset
file_path = "C:\\Users\\cmadh\\Downloads\\final_cleaned_electronics_dataset.csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = ["Product Name", "Brand", "Category", "Price (INR)"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing required columns: {missing_columns}")

# Remove outliers
df = df[(df["Price (INR)"] > 500) & (df["Price (INR)"] < 100000)]

# Preserve original price before scaling
df["Price_Original"] = df["Price (INR)"].copy()

# Encode categorical features
encoders = {}
for col in ["Brand", "Category", "Product Name"]:
    df[col] = df[col].astype(str).fillna("Unknown")
    encoders[col] = LabelEncoder()
    df[col + "_Encoded"] = encoders[col].fit_transform(df[col])

# Normalize numerical values
scaler = MinMaxScaler()
df["Price (INR)"] = scaler.fit_transform(df[["Price (INR)"]])

# Define features and target
features = ["Brand_Encoded", "Category_Encoded", "Product Name_Encoded"]
target = "Price (INR)"
X = df[features]
y = df[target]

# Train model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Function to encode unseen values
def safe_encode(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        # Assign default value for unseen data (mean value)
        return int(np.mean(encoder.transform(encoder.classes_)))


# ✅ Function to compute discount based on price
def compute_discount(price):
    if price < 10000:
        return 10  # 10% discount for low-priced products
    else:
        return 5   # 5% discount for high-priced products


# ✅ Function to predict optimal price
def predict_optimal_price(brand, category, product_name):
    try:
        # Check if product exists in dataset
        product_data = df[
            (df["Brand"] == brand) &
            (df["Category"] == category) &
            (df["Product Name"] == product_name)
        ]

        if not product_data.empty:
            avg_price = product_data["Price_Original"].mean()
            discount = compute_discount(avg_price)
            return round(avg_price, 2), discount
        else:
            # Encode new inputs
            brand_encoded = safe_encode(encoders["Brand"], brand)
            category_encoded = safe_encode(encoders["Category"], category)
            product_encoded = safe_encode(encoders["Product Name"], product_name)

            if brand_encoded == -1 or category_encoded == -1 or product_encoded == -1:
                return "Invalid input. Please check the values.", None

            predicted_price = model.predict([[brand_encoded, category_encoded, product_encoded]])[0]
            predicted_price = scaler.inverse_transform([[predicted_price]])[0][0]

            # Compute discount based on predicted price
            discount = compute_discount(predicted_price)

            return round(predicted_price, 2), discount

    except Exception as e:
        return f"Error: {str(e)}", None


# ✅ Flask routes
@app.route("/")
def home():
    brands = sorted(df["Brand"].unique())
    categories = sorted(df["Category"].unique())
    return render_template("index.html", brands=brands, categories=categories)


# ✅ Get brands based on category
@app.route("/get-brands", methods=["POST"])
def get_brands():
    try:
        data = request.get_json()
        category = data.get('category')

        if not category:
            return jsonify({'error': 'Category is required'}), 400

        brands = sorted(df[df['Category'] == category]['Brand'].unique().tolist())
        return jsonify({'brands': brands})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Get products based on brand
@app.route("/get-products", methods=["POST"])
def get_products():
    try:
        data = request.get_json()
        brand = data.get('brand')

        if not brand:
            return jsonify({'error': 'Brand is required'}), 400

        products = sorted(df[df['Brand'] == brand]['Product Name'].unique().tolist())
        return jsonify({'products': products})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ✅ Predict price based on inputs
@app.route("/predict", methods=["POST"])
def predict():
    try:
        category = request.form.get("category")
        brand = request.form.get("brand")
        product_name = request.form.get("product_name")

        if not category or not brand or not product_name:
            return jsonify({"error": "All fields are required!"}), 400

        price, discount = predict_optimal_price(brand, category, product_name)

        if discount is None:
            return jsonify({"error": price}), 400

        final_price = round(price * (1 - discount / 100), 2)

        return jsonify({
            "product": product_name,
            "predicted_price": price,
            "discount": discount,
            "final_price": final_price
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Run app
if __name__ == "__main__":
    app.run(debug=True)
