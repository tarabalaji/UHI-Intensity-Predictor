import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


@st.cache_data
def load_data():
    df = pd.read_excel("data.xlsx")
    columns = [
        "Urban_name",
        "NDVI_urb_CT_act",
        "DelNDVI_annual",
        "DelDEM",
        "UHI_annual_day",
    ]
    df = df[columns]
    df = df.dropna()
    return df


df = load_data()


@st.cache_resource
def train_models():
    df = load_data()
    X_phys = pd.DataFrame(
        {
            "inv_ndvi": 1 - df["NDVI_urb_CT_act"],
            "del_ndvi": df["DelNDVI_annual"],
            "del_dem": df["DelDEM"] / 100,
        }
    )
    y_phys = df["UHI_annual_day"]
    phys_model = LinearRegression()
    phys_model.fit(X_phys, y_phys)

    x_all = df[["NDVI_urb_CT_act", "DelNDVI_annual", "DelDEM"]].copy()
    x_all["NDVI_x_DelNDVI"] = x_all["NDVI_urb_CT_act"] * x_all["DelNDVI_annual"]
    y_all = df["UHI_annual_day"]
    x_all["DelDEM"] /= 100
    ml_model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=1.0,
        gamma=0,
        objective="reg:squarederror",
        random_state=42,
    )
    ml_model.fit(x_all, y_all)
    return phys_model, ml_model


phys_model, ml_model = train_models()

ALPHA, BETA, GAMMA = phys_model.coef_
INTERCEPT = phys_model.intercept_
st.title("UHIQ")
st.write(
    """
This urban heat island intensity modeling software predicts and compares
UHI intensity using a formula-based model and a ML-based model using
environmental variables such as vegetation indicators and elevation change.
"""
)

if "start_time" not in st.session_state:
    st.session_state["start_time"] = None
if "timings" not in st.session_state:
    st.session_state["timings"] = []
if "recorded_this_run" not in st.session_state:
    st.session_state["recorded_this_run"] = False


def record_start_time():
    st.session_state["start_time"] = time.time()
    st.session_state["recorded_this_run"] = False


city = st.selectbox(
    "Select an Urban Area:",
    sorted(df["Urban_name"].unique()),
    key="city",
    on_change=record_start_time,
)

city_data = df[df["Urban_name"] == city]
city_data = city_data.sort_values("UHI_annual_day")

low_row = city_data.loc[city_data["UHI_annual_day"].idxmin()]
high_row = city_data.loc[city_data["UHI_annual_day"].idxmax()]
typical_row = city_data.iloc[len(city_data) // 2]

st.subheader("City Data")
st.dataframe(city_data)

mode = st.radio("Select Scenario", ["Low Heat", "Typical", "High Heat"], index=1)

if mode == "Low Heat":
    row = low_row
elif mode == "High Heat":
    row = high_row
else:
    row = typical_row

ndvi = row["NDVI_urb_CT_act"]
del_ndvi = row["DelNDVI_annual"]
del_dem = row["DelDEM"]
del_dem_scale = del_dem / 100
uhi_actual = row["UHI_annual_day"]

ndvi_input = st.slider(
    "Urban NDVI",
    0.0,
    1.0,
    float(row["NDVI_urb_CT_act"]),
    0.01,
    key=f"ndvi_{city}_{mode}",
)

del_ndvi_input = st.slider(
    "ΔNDVI", -0.5, 0.5, float(row["DelNDVI_annual"]), 0.01, key=f"delndvi_{city}_{mode}"
)

del_dem_input = st.slider(
    "ΔDEM", -50.0, 50.0, float(row["DelDEM"]), 0.1, key=f"deldem_{city}_{mode}"
)

albedo_input = st.slider(
    "Albedo (0-1)", 0.1, 0.9, 0.3, 0.01, key="albedo_input", on_change=record_start_time
)

urban_frac_input = st.slider(
    "Urban Fraction (0-1)",
    0.0,
    1.0,
    0.5,
    0.01,
    key="urban_frac_input",
    on_change=record_start_time,
)

del_dem_input_scale = del_dem_input / 100

uhi_formula_input = (
    INTERCEPT
    + ALPHA * (1 - ndvi_input)
    + BETA * del_ndvi_input
    + GAMMA * del_dem_input_scale
)


st.subheader("UHI Calculation using Formula-based Model")
st.write(f"**Predicted UHI (Formula Model):** {uhi_formula_input:.2f} °C")
st.write(f"**Actual UHI:** {uhi_actual:.2f} °C")

ndvi_range = np.arange(0, 1, 0.05)
predicted_uhi_sensitivity = []

for test in ndvi_range:
    predicted_uhi_sensitivity.append(
        INTERCEPT
        + ALPHA * (1 - test)
        + BETA * del_ndvi_input
        + GAMMA * del_dem_input_scale
    )
sensitivity_table = pd.DataFrame(
    {"Simulated NDVI": ndvi_range, "Predicted UHI °C": predicted_uhi_sensitivity}
)
st.subheader("Sensitivity Analysis: Impact of Vegetation on UHI Intensity")
st.dataframe(sensitivity_table)

fig = px.line(
    sensitivity_table,
    x="Simulated NDVI",
    y="Predicted UHI °C",
    title="Predicted UHI vs NDVI",
    labels={"Simulated NDVI": "NDVI", "Predicted UHI °C": "UHI (°C)"},
)

st.plotly_chart(fig)

comparison_df = pd.DataFrame(
    {
        "Type": ["Formula Prediction", "Actual"],
        "UHI (°C)": [uhi_formula_input, uhi_actual],
    }
)

fig2 = px.bar(
    comparison_df,
    x="Type",
    y="UHI (°C)",
    text="UHI (°C)",
    title="Actual vs Formula Model UHI",
    color="Type",
)
st.plotly_chart(fig2)

ml_input = pd.DataFrame(
    [
        {
            "NDVI_urb_CT_act": ndvi_input,
            "DelNDVI_annual": del_ndvi_input,
            "DelDEM": del_dem_input,
        }
    ]
)

ml_input["NDVI_x_DelNDVI"] = ml_input["NDVI_urb_CT_act"] * ml_input["DelNDVI_annual"]
ml_input["DelDEM"] /= 100
raw_ml = ml_model.predict(ml_input)[0]
uhi_ml_input = raw_ml
st.subheader("UHI Calculation using ML-based Model")
st.write(f"**Predicted UHI (ML Model):** {uhi_ml_input:.2f} °C")
st.write(f"**Actual UHI:** {uhi_actual:.2f} °C")

predicted_ml_sensitivity = [
    ml_model.predict(
        pd.DataFrame(
            [
                {
                    "NDVI_urb_CT_act": ndvi_val,
                    "DelNDVI_annual": del_ndvi_input,
                    "DelDEM": del_dem_input,
                    "NDVI_x_DelNDVI": ndvi_val * del_ndvi_input,
                }
            ]
        )
    )[0]
    for ndvi_val in ndvi_range
]

ml_sensitivity_table = pd.DataFrame(
    {"Simulated NDVI": ndvi_range, "Predicted UHI °C": predicted_ml_sensitivity}
)

st.subheader("Sensitivity Analysis: Impact of Vegetation on UHI Intensity (ML Model)")
st.dataframe(ml_sensitivity_table)

fig4 = px.line(
    ml_sensitivity_table,
    x="Simulated NDVI",
    y="Predicted UHI °C",
    title="Predicted UHI vs NDVI (ML Model)",
    labels={"Simulated NDVI": "NDVI", "Predicted UHI °C": "UHI (°C)"},
)
st.plotly_chart(fig4)

st.subheader(
    "Comparison of Actual UHI, ML Model Prediction, and Formula Model Prediction"
)

comparison_df = pd.DataFrame(
    {
        "Type": ["Actual", "ML Prediction", "Formula Prediction"],
        "UHI (°C)": [uhi_actual, uhi_ml_input, uhi_formula_input],
    }
)

fig3 = px.bar(
    comparison_df,
    x="Type",
    y="UHI (°C)",
    text="UHI (°C)",
    title="Actual vs ML Model vs Formula Model Predicted UHI",
    color="Type",
)
st.plotly_chart(fig3)

x_eval = df[["NDVI_urb_CT_act", "DelNDVI_annual", "DelDEM"]].copy()
x_eval["NDVI_x_DelNDVI"] = x_eval["NDVI_urb_CT_act"] * x_eval["DelNDVI_annual"]
x_eval["DelDEM"] /= 100
y_eval = df["UHI_annual_day"]
X_train, X_test, y_train, y_test = train_test_split(
    x_eval, y_eval, test_size=0.2, random_state=42
)

evaluation_model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.01,
    reg_lambda=1.0,
    gamma=0,
    objective="reg:squarederror",
    random_state=42,
)
evaluation_model.fit(X_train, y_train)
y_pred_ml = evaluation_model.predict(X_test)


def formula_model(row):
    return (
        INTERCEPT
        + ALPHA * (1 - row["NDVI_urb_CT_act"])
        + BETA * row["DelNDVI_annual"]
        + GAMMA * row["DelDEM"]
    )


y_pred_formula = X_test.apply(formula_model, axis=1)
mae_ml = mean_absolute_error(y_test, y_pred_ml)
mae_formula = mean_absolute_error(y_test, y_pred_formula)
r2_ml = r2_score(y_test, y_pred_ml)
r2_formula = r2_score(y_test, y_pred_formula)

st.subheader("Model Accuracy Evaluation")
st.write(f"**ML Model MAE:** {mae_ml:.2f} °C")
st.write(f"**ML Model R²:** {r2_ml:.3f}")
st.write(f"**Formula Model MAE:** {mae_formula:.2f} °C")
st.write(f"**Formula Model R²:** {r2_formula:.3f}")

error_df = pd.DataFrame(
    {
        "Model": ["Formula Model", "ML Model"],
        "Mean Absolute Error (°C)": [mae_formula, mae_ml],
    }
)

fig_error = px.bar(
    error_df,
    x="Model",
    y="Mean Absolute Error (°C)",
    title="Average Prediction Error Comparison",
    text="Mean Absolute Error (°C)",
)
st.plotly_chart(fig_error)

perm = permutation_importance(
    evaluation_model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error",
)

perm_df = pd.DataFrame(
    {"Feature": X_test.columns, "Permutation Importance": perm.importances_mean}
).sort_values("Permutation Importance", ascending=False)

st.subheader("Permutation Importance (ML Features)")
st.dataframe(perm_df)

fig_perm = px.bar(
    perm_df,
    x="Feature",
    y="Permutation Importance",
    title="Permutation Importance of Features",
    text_auto=".3f",
)
st.plotly_chart(fig_perm)

start = st.session_state.get("start_time")
if start:
    elapsed = time.time() - start
    st.write(f"**Execution time:** {elapsed:.3f} seconds")
    if not st.session_state.get("recorded_this_run", False):
        st.session_state["timings"].append(elapsed)
        st.session_state["recorded_this_run"] = True
if st.session_state.get("timings"):
    avg = sum(st.session_state["timings"]) / len(st.session_state["timings"])
    st.write(f"**Average loading time:** {avg:.3f} seconds")

# dashboard display
st.divider()
st.header("UHIQ Interactive Prediction Dashboard")

st.caption(
    "Scenario-based comparison of physics-informed linear regression "
    "and XGBoost urban heat island predictions."
)


st.subheader("Dashboard Controls")

control_col1, control_col2 = st.columns(2)

with control_col1:
    st.selectbox(
        "Urban Area",
        sorted(df["Urban_name"].unique()),
        index=sorted(df["Urban_name"].unique()).index(city),
    )

with control_col2:
    st.radio(
        "Scenario",
        ["Low Heat", "Typical", "High Heat"],
        index=["Low Heat", "Typical", "High Heat"].index(mode),
        horizontal=True,
    )

st.subheader("Environmental Inputs")

slider_col1, slider_col2, slider_col3 = st.columns(3)

with slider_col1:
    screenshot_ndvi = st.slider(
        "Urban NDVI",
        min_value=0.0,
        max_value=1.0,
        value=float(ndvi_input),
        step=0.01,
        key="screenshot_ndvi",
    )

with slider_col2:
    screenshot_del_ndvi = st.slider(
        "ΔNDVI",
        min_value=-0.5,
        max_value=0.5,
        value=float(del_ndvi_input),
        step=0.01,
        key="screenshot_del_ndvi",
    )

with slider_col3:
    screenshot_del_dem = st.slider(
        "ΔDEM (m)",
        min_value=-50.0,
        max_value=50.0,
        value=float(del_dem_input),
        step=0.1,
        key="screenshot_del_dem",
    )


st.subheader("Model Predictions")

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

with prediction_col1:
    st.metric(
        label="Actual UHI",
        value=f"{uhi_actual:.2f} °C",
    )

with prediction_col2:
    st.metric(
        label="Physics-Informed Prediction",
        value=f"{uhi_formula_input:.2f} °C",
        delta=f"{uhi_formula_input - uhi_actual:.2f} °C",
    )

with prediction_col3:
    st.metric(
        label="XGBoost Prediction",
        value=f"{uhi_ml_input:.2f} °C",
        delta=f"{uhi_ml_input - uhi_actual:.2f} °C",
    )


screenshot_comparison_df = pd.DataFrame(
    {
        "Prediction Type": [
            "Observed UHI",
            "Physics-Informed Model",
            "XGBoost Model",
        ],
        "UHI Intensity (°C)": [
            uhi_actual,
            uhi_formula_input,
            uhi_ml_input,
        ],
    }
)

screenshot_comparison_fig = px.bar(
    screenshot_comparison_df,
    x="Prediction Type",
    y="UHI Intensity (°C)",
    text="UHI Intensity (°C)",
    title="Observed and Predicted Urban Heat Island Intensity",
)

screenshot_comparison_fig.update_traces(
    texttemplate="%{text:.2f} °C",
    textposition="outside",
)

screenshot_comparison_fig.update_layout(
    showlegend=False,
    xaxis_title=None,
)

st.plotly_chart(
    screenshot_comparison_fig,
    use_container_width=True,
)
