---
🔹 Author: Mohammad Umar  
🔹 Contact: umar.test.49@gmail.com  
---

# 🌲 Forest Cover Type Prediction – Machine Learning Project

---

### 📌 Section 1: Introduction and Objective  

Forests are essential ecosystems that support biodiversity, regulate climate, and provide crucial resources. Understanding and classifying different forest cover types allows researchers, environmental agencies, and land planners to make informed decisions for conservation, wildfire risk management, and sustainable land use.  

This project assumes the role of a data science intern at a forestry research organization. The objective is to build a machine learning model that can accurately predict the type of forest cover for a 30m x 30m patch of land based on environmental and topographical features such as elevation, slope, soil type, proximity to water, and sunlight exposure.

🔹 **Problem Being Solved:**  
Automatically classify forest cover type using geographic features to eliminate the need for manual classification through fieldwork or satellite imagery interpretation.

🔹 **Why It Matters:**  
- Reduces costs and time in forest type classification  
- Assists in ecological zone mapping and habitat modeling  
- Supports fire prevention strategies and reforestation planning  

🔹 **Final Objective:**  
Develop and deploy a machine learning model that predicts one of seven forest cover types from raw feature input, and make it accessible through a non-technical Streamlit web app.

---

### 📊 Section 2: Dataset  

- **Source:** UCI Machine Learning Repository – [Forest CoverType dataset](https://archive.ics.uci.edu/ml/datasets/covertype)  
- **Shape:** 15,120 rows × 55 columns  

#### 🧾 Description of Features:
- **Numerical Features (10):**
  - Elevation, Aspect, Slope, Hillshade (9am, Noon, 3pm)
  - Distances to: Hydrology, Roadways, Fire Points, both horizontal and vertical
- **Categorical/Binary Features:**
  - **Wilderness_Area** (4 one-hot encoded columns)
  - **Soil_Type** (40 one-hot encoded columns)
- **Target Variable:** `Cover_Type` (7 classes, labeled 1 to 7)
  - 1: Spruce/Fir  
  - 2: Lodgepole Pine  
  - 3: Ponderosa Pine  
  - 4: Cottonwood/Willow  
  - 5: Aspen  
  - 6: Douglas-fir  
  - 7: Krummholz  

#### 🧹 Preprocessing Performed:
- Dropped `Id` column  
- Checked and confirmed **no missing values**  
- Removed duplicates  
- Ensured consistent feature scaling using `StandardScaler`  

#### 🔍 Key Observations:
- Elevation is the **most important predictor**
- Cover types are relatively balanced across the dataset
- Soil and Wilderness types are already one-hot encoded
- Some features (e.g., slope, hillshade) show correlations with specific forest types

---

### ⚙️ Section 3: Design / Workflow  

The project followed a full ML lifecycle with the following steps:

#### 🔸 Step-by-Step Workflow:

1. **Data Loading & Cleaning**  
   - Loaded CSV into a Pandas DataFrame  
   - Removed unnecessary columns and duplicates  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized target distribution  
   - Generated correlation heatmaps  
   - Identified most impactful numerical features  

3. **Feature Engineering**  
   - Created one-hot encoded vectors for soil and wilderness types  
   - Normalized numeric features using `StandardScaler`  
   - Combined all features into a unified feature matrix  

4. **Model Training & Testing**  
   - Initial baseline using **Random Forest Classifier**  
   - Stratified train-test split (80/20)

5. **Hyperparameter Tuning**  
   - GridSearchCV on:
     - `n_estimators`: [50, 100, 150]  
     - `max_depth`: [10, 20, 30]  
     - `min_samples_split`: [2, 5, 10]  
   - Best parameters: `n_estimators=150`, `max_depth=30`, `min_samples_split=2`

6. **Model Evaluation**  
   - Evaluated using Accuracy and Classification Report  
   - Visualized feature importance  

7. **Deployment**  
   - Built an interactive **Streamlit UI**  
   - Accepts user input via dropdowns, sliders, and radio buttons  
   - Shows the predicted forest cover type in plain language (e.g., "Aspen", "Krummholz")

---

### 📈 Section 4: Results  

#### ✅ Final Model: Random Forest Classifier (Tuned)
- **Test Accuracy:** 85%  
- **Precision/Recall/F1-Score** per class:
  - All classes performed well, especially types 4 (Cottonwood/Willow) and 7 (Krummholz)
- **Confusion Matrix:** Most predictions lie on the diagonal; few misclassifications
- **Feature Importance Chart:**  
  - Top Features: `Elevation`, `Horizontal_Distance_To_Roadways`, `Horizontal_Distance_To_Fire_Points`, `Hillshade_Noon`, and `Wilderness Area`
  
#### 📊 Key Insights:
- Elevation alone captures a huge part of the variance — certain forest types are elevation-specific  
- Distance-based features further refine the model’s decision-making  
- The model generalizes well across all classes with minimal overfitting  
- The Streamlit interface is intuitive and ready for use by non-technical users

---

### ✅ Section 5: Conclusion  

#### 🔹 Summary:
This project successfully built a robust machine learning model that can predict the type of forest cover based on geospatial inputs. The final deployment via Streamlit makes it usable by ecologists, field workers, and even students.

#### 🔹 Challenges Faced:
- Mapping one-hot encoded binary columns back to user-friendly dropdowns  
- Making the interface intuitive for non-technical users  
- Ensuring prediction results were meaningful and readable  

#### 🔹 Future Scope:
- Add support for batch prediction through CSV uploads  
- Integrate satellite imagery or NDVI data for improved accuracy  
- Extend the model with XGBoost or deep learning alternatives  
- Add model explainability using SHAP or LIME  

#### 🔹 Personal Learnings:
- Learned how to bridge machine learning logic with real-world usability through deployment  
- Gained experience with feature preprocessing and encoding strategies  
- Developed strong intuition for classification workflows and tuning  
- Improved UI design skills with Streamlit for practical ML apps  

---

✅ **Project Status: Complete & Deployed**  
