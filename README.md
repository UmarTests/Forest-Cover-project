---
🔹 **Author**: Mohammad Umar  
🔹 **Contact**: [umar.test.49@gmail.com](mailto:umar.test.49@gmail.com)  
---

# 🌲 Forest Cover Type Prediction – Machine Learning Project

---

## 📌 Section 1: Introduction and Objective  

Forests are essential ecosystems that support biodiversity, regulate climate, and provide crucial resources. Understanding and classifying different forest cover types allows researchers, environmental agencies, and land planners to make informed decisions for conservation, wildfire risk management, and sustainable land use.  

### 🔹 Problem Being Solved:  
Automatically classify forest cover type using geographic features to eliminate the need for manual classification through fieldwork or satellite imagery interpretation.

### 🔹 Why It Matters:  
- Reduces costs and time in forest type classification  
- Assists in ecological zone mapping and habitat modeling  
- Supports fire prevention strategies and reforestation planning  

### 🔹 Final Objective:  
Develop and deploy a machine learning model that predicts one of seven forest cover types from raw feature input, and make it accessible through a non-technical Streamlit web app.

---

## 📊 Section 2: Dataset  

- **Source**: [UCI Machine Learning Repository – Forest CoverType dataset](https://archive.ics.uci.edu/ml/datasets/covertype)  
- **Shape**: 15,120 rows × 55 columns  

### 🧾 Description of Features:
#### Numerical Features (10):
- Elevation, Aspect, Slope, Hillshade (9am, Noon, 3pm)
- Distances to: Hydrology, Roadways, Fire Points (horizontal and vertical)

#### Categorical/Binary Features:
- **Wilderness_Area** (4 one-hot encoded columns)
- **Soil_Type** (40 one-hot encoded columns)

#### Target Variable: `Cover_Type` (7 classes):
1. Spruce/Fir  
2. Lodgepole Pine  
3. Ponderosa Pine  
4. Cottonwood/Willow  
5. Aspen  
6. Douglas-fir  
7. Krummholz  

### 🧹 Preprocessing:
- Dropped `Id` column  
- Confirmed **no missing values**  
- Removed duplicates  
- Applied `StandardScaler` for consistent feature scaling  

### 🔍 Key Observations:
- Elevation is the **most important predictor**
- Balanced distribution of cover types
- Soil and Wilderness types already one-hot encoded
- Some features show correlations with specific forest types

---

## ⚙️ Section 3: Design / Workflow  

### 🔸 Step-by-Step Workflow:

1. **Data Loading & Cleaning**  
   - Loaded CSV into Pandas DataFrame  
   - Removed unnecessary columns and duplicates  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized target distribution  
   - Generated correlation heatmaps  
   - Identified impactful numerical features  

3. **Feature Engineering**  
   - One-hot encoded soil and wilderness types  
   - Normalized features using `StandardScaler`  

4. **Model Training & Testing**  
   - Baseline: **Random Forest Classifier**  
   - Stratified train-test split (80/20)

5. **Hyperparameter Tuning**  
   - `GridSearchCV` parameters:
     - `n_estimators`: [50, 100, 150]  
     - `max_depth`: [10, 20, 30]  
     - `min_samples_split`: [2, 5, 10]  
   - Best params: `n_estimators=150`, `max_depth=30`, `min_samples_split=2`

6. **Model Evaluation**  
   - Accuracy and Classification Report  
   - Feature importance visualization  

7. **Deployment**  
   - Built interactive **Streamlit UI**  
   - User input via dropdowns/sliders  
   - Plain-language predictions (e.g., "Aspen")

---

## 📈 Section 4: Results  

### ✅ Final Model: Random Forest Classifier (Tuned)
| Metric          | Performance |
|-----------------|-------------|
| Test Accuracy   | 85%         |
| Best Classes    | 4 & 7       |

### 📊 Key Insights:
- Elevation explains most variance  
- Distance features refine decisions  
- Generalizes well with minimal overfitting  
- Streamlit interface is user-friendly  

---

## ✅ Section 5: Conclusion  

### 🔹 Summary:
Built a robust forest cover predictor with deployable Streamlit interface for practical use.

### 🔹 Challenges:
- One-hot encoding ↔ user-friendly mapping  
- UI design for non-technical users  
- Meaningful result presentation  

### 🔹 Future Scope:
- Batch CSV predictions  
- Satellite imagery integration  
- XGBoost/SHAP explainability  

### 🔹 Learnings:
- ML-to-real-world deployment  
- Feature engineering strategies  
- Streamlit UI design  

✅ **Project Status: Complete & Deployed**
