# 🤖 AI-Based Pricing Strategy for Electronic Products

A smart and scalable pricing recommendation system built using **Machine Learning** that suggests **optimal prices and discounts** for electronic products based on product name, brand, and category. This system leverages **Random Forest Regression** to bring intelligence to product pricing strategies.

---

## 📌 Project Overview

This project predicts electronic product prices and provides discount recommendations by analyzing historical market data. It is ideal for online retailers, pricing analysts, or product managers looking to optimize pricing dynamically.

---

## 🔧 Features

- 🎯 Predicts prices for known and unseen products
- 📉 Recommends discounts by comparing category averages
- 🤝 Accepts user input via terminal (brand, category, model)
- 💾 Includes model training, serialization (`.pkl`), and reloading
- 🔁 Handles missing or new product info gracefully

---

## 🧰 Tech Stack

| Layer                | Tools / Libraries       |
|---------------------|-------------------------|
| Programming Language| Python                  |
| ML Model            | Random Forest Regressor |
| Data Processing     | Pandas, NumPy           |
| Preprocessing       | LabelEncoder, MinMaxScaler |
| Model Persistence   | Pickle                  |
| IDE Used            | PyCharm                 |

---

## 🗃️ Dataset

- **File:** `final_cleaned_electronics_dataset.csv`
- **Columns Used:** `Product Name`, `Brand`, `Category`, `Price (INR)`
- **Details:** Contains cleaned and preprocessed records for mobiles, laptops, TVs, and more.
- **Location:** Included in this repository (project root).

> If the file is missing, [download here]("C:\Users\cmadh\Downloads\final_updated_ultra_large_electronics_dataset.csv") and place it in your project folder.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally:

---

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/Ganesh200327/ai-based-pricing-strategy.git
cd ai-based-pricing-strategy
```

---

### 📥 Step 2: Install Dependencies

Use a virtual environment (optional but recommended):

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Then install the required libraries:

```bash
pip install -r requirements.txt
```

> 📦 Example contents of `requirements.txt`:

```
pandas
numpy
scikit-learn
```

---

### ⚙️ Step 3: Run the Script

After installing dependencies, run the pricing system:

```bash
python script.py
```

You’ll be prompted to enter:

* Product Category
* Brand
* Product Model

The system will then output the suggested price and discount dynamically based on the trained machine learning model.

---

### ✅ Sample Output

```text
🔥 AI Pricing Strategy System 🔥

📌 Available Categories: ['Mobile', 'Laptop', 'TV']
📌 Available Brands: ['Samsung', 'Apple', 'Sony']

Enter product category: Laptop
Enter product brand: Apple
📌 Available Models: ['MacBook Air', 'MacBook Pro']

Enter product model: MacBook Pro

💰 Suggested Price for MacBook Pro: ₹89999.0
🔻 Recommended Discount: 7%
📉 Final Price After Discount: ₹83609.07
```

---

## 📁 Folder Structure

```
ai-based-pricing-strategy/
│
├── script.py                         # Main executable script
├── final_cleaned_electronics_dataset.csv  # Input dataset
├── model.pkl                         # Trained ML model file
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git ignored files
```

---

## 🌍 Future Enhancements

* 🌐 Web UI with **Streamlit** or **Flask**
* 📈 Interactive data visualizations
* 🔁 Auto-retraining pipeline
* 🧪 Unit tests for model and functions

---

## 👤 Author

**Cheruvu Madhu Ganesh**  
🎓 B.Tech – Computer Science and Business Systems (2025)  
📧 [angadidivya210@gmail.com](mailto:angadidivya210@gmail.com)  
📍 Peravali, Andhra Pradesh, India  
🔗 [LinkedIn](www.linkedin.com/in/divya-angadi *(update with your profile)*

---

## 🛡 License

This project is licensed under the **MIT License**.

---

## ⭐ Show Your Support

If you found this project helpful:

* ⭐ Star this repo
* 🍴 Fork and customize
* 📢 Share with your peers

---
