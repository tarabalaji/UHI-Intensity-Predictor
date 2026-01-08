Urban Heat Island Prediction Modeling Framework

Overview
This project provides an interactive framework to predict and analyze Urban Heat Island (UHI) intensity using both a formula-based model and a machine learning (ML) model. Users can explore how urban characteristics—such as vegetation, elevation differences, albedo, and urban fraction—affect UHI intensity across different cities.

The software allows:
City-specific UHI data exploration
Sensitivity analysis of vegetation and urban variables
Comparison between formula-based and ML-based predictions
Evaluation of model accuracy (Mean Absolute Error)

Built with Python, Streamlit, Plotly, and XGBoost, this tool is designed for environmental research, urban planning, and educational purposes.

Features
Interactive City Selection: View urban-specific data in a table format.
Simulate Urban Changes: Adjust sliders for NDVI, elevation, albedo, and urban fraction to predict UHI.
Formula-based Prediction: Uses linear regression on physical variables to predict UHI.
ML-based Prediction: Uses an XGBoost regressor to predict UHI for a city based on other cities’ data.
Sensitivity Analysis: Visualize the impact of vegetation (NDVI) on predicted UHI.
Comparison Visualizations: Compare actual UHI, formula predictions, and ML predictions using interactive bar charts.
Model Accuracy Evaluation: Mean Absolute Error for formula-based and ML models.

Installation
Clone the repository:
git clone https://github.com/taroboba11/UHI-Intensity-Predictor.git
cd urban-heat-island
Install required packages (preferably in a virtual environment):
pip install streamlit pandas numpy plotly scikit-learn xgboost openpyxl
Place the dataset data.xlsx in the project directory. Ensure it contains the columns:
Urban_name, NDVI_urb_CT_act, DelNDVI_annual, DelDEM, UHI_annual_day

Usage
Run the Streamlit app with:
streamlit run app.py
Select a city from the dropdown menu.
Adjust sliders under “Simulate Changes” to explore the effect of urban variables.
View predicted UHI values from both formula-based and ML-based models.
Explore sensitivity analyses and comparison charts.
Check model accuracy with Mean Absolute Error metrics.

Project Structure
├─ app.py               # Main Streamlit application
├─ data.xlsx            # Dataset with urban variables and UHI values
├─ README.md            # Project documentation

Key Libraries
Streamlit: Interactive UI and dashboarding
Pandas & NumPy: Data handling and manipulation
Plotly Express: Interactive visualizations
Scikit-learn: Formula-based linear regression
XGBoost: ML regression for UHI prediction
