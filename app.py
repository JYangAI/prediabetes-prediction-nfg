from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly
import plotly.express as px
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load
from tools.scaler import inference_scaler

# ====================== Load Model ======================
model = load("./models/model_files/xgboost_classifier_model.joblib")
model.feature_names = ["Age", "BMI", "Pulse", "SBP", "FPG", "TG", "TC", "TyG", "WBC", "RBC", "HGB", "PLT", "MHR"]
explainer = shap.Explainer(model)

# ========================== UI =========================
app_ui = ui.page_fluid(
    ui.h2("🩺 Prediabetes Prediction in Individuals with Normal Fasting Glucose"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("📋 Please input corresponding values in the textboxes below:"),
            ui.input_numeric("age", "Age (year)", value=45, min=20, max=99, step=1),
            ui.input_numeric("height", "Height (cm)", value=170.0, min=100.0, max=250.0, step=0.1),
            ui.input_numeric("weight", "Weight (kg)", value=65.0, min=30.0, max=300.0, step=0.1),
            ui.input_numeric("pulse", "Pulse (bpm)", value=75, min=30, max=200, step=1),
            ui.input_numeric("sbp", "Systolic Blood Pressure, SBP (mmHg)", value=125, min=50, max=250, step=1),
            ui.input_numeric("fpg", "Fasting Plasma Glucose, FPG (mmol/L)", value=5.00, min=2.00, max=6.09, step=0.01),
            ui.input_numeric("tg", "Triglyceride, TG (mmol/L)", value=1.50, min=0.10, max=30.00, step=0.01),
            ui.input_numeric("tc", "Total cholesterol, TC (mmol/L)", value=4.50, min=0.10, max=20.00, step=0.01),
            ui.input_numeric("hdl_c", "High-Density Lipoprotein Cholesterol, HDL-C (mmol/L)", value=1.20, min=0.10,
                             max=5.00, step=0.01),
            ui.input_numeric("wbc", "White Blood Cell Count, WBC (10^9/L)", value=5.50, min=0.10, max=50.00, step=0.01),
            ui.input_numeric("rbc", "Red Blood Cell Count, RBC (10^12/L)", value=4.50, min=0.10, max=20.00, step=0.01),
            ui.input_numeric("hgb", "Hemoglobin, HGB (g/L)", value=120, min=1, max=300, step=1),
            ui.input_numeric("plt_count", "Platelet, PLT (10^9/L)", value=200, min=1, max=1500, step=1),
            ui.input_numeric("mono", "Monocyte, MONO# (10^9/L)", value=0.40, min=0.00, max=10.00, step=0.01),
            ui.input_action_button("btn_run", "Run", class_="btn-primary"),
            ui.input_action_button("btn_example", "Example", class_="btn-primary"),
        ),

        ui.card(
            ui.h3("📉 Prediction Results"),
            ui.output_ui("predicted_risk"),
            output_widget("pie_chart"),
            ui.output_ui("suggestion"),
        ),

        ui.card(
            ui.h3("📊 Interpretation"),
            ui.output_plot("shap_waterfall"),
            ui.output_ui("feature_note"),
        ),
    ),
)


# ====================== Server ======================
def server(input, output, session):
    results = reactive.Value(None)

    @reactive.effect
    @reactive.event(input.btn_run)
    def compute():
        age = float(input.age())
        height = float(input.height())
        weight = float(input.weight())
        pulse = float(input.pulse())
        sbp = float(input.sbp())
        fpg = float(input.fpg())
        tg = float(input.tg())
        tc = float(input.tc())
        hdl_c = float(input.hdl_c())
        wbc = float(input.wbc())
        rbc = float(input.rbc())
        hgb = float(input.hgb())
        plt_count = float(input.plt_count())
        mono = float(input.mono())

        bmi = weight / ((height / 100) ** 2)
        tyg = np.log(tg * 88.57 * fpg * 18.02 / 2)
        mhr = mono / hdl_c

        input_df = pd.DataFrame([[age, bmi, pulse, sbp, fpg, tg, tc, tyg, wbc, rbc, hgb, plt_count, mhr]],
                                columns=model.feature_names)
        print("input_df", input_df.to_string())

        features_scaled = inference_scaler(input_df, "./tools/scaler.joblib")
        input_scaled_df = pd.DataFrame(features_scaled, columns=input_df.columns)
        # print("input df", input_scaled_df.to_string())

        prob = model.predict_proba(input_scaled_df)[0]
        # print("prob", prob)
        shap_val = explainer(input_scaled_df)[0]
        # print("shap_val", shap_val)
        feat_names = model.feature_names

        predicted_class = int(np.argmax(prob))
        feat_values = input_scaled_df.iloc[0].to_dict()

        results.set({"prob": prob,
                     "shap_val": shap_val,
                     "feat_names": feat_names,
                     "predicted_class": predicted_class,
                     "feat_values": feat_values
                     })
        print("results", results())

    @reactive.effect
    @reactive.event(input.btn_example)
    def _reset_values():
        ui.update_numeric("age", value=45)
        ui.update_numeric("height", value=170.0)
        ui.update_numeric("weight", value=65.0)
        ui.update_numeric("pulse", value=75)
        ui.update_numeric("sbp", value=125)
        ui.update_numeric("fpg", value=5.00)
        ui.update_numeric("tg", value=1.50)
        ui.update_numeric("tc", value=4.50)
        ui.update_numeric("hdl_c", value=1.20)
        ui.update_numeric("wbc", value=5.50)
        ui.update_numeric("rbc", value=4.50)
        ui.update_numeric("hgb", value=120)
        ui.update_numeric("plt_count", value=200)
        ui.update_numeric("mono", value=0.40)
        ui.notification_show("Reset to sample value", type="message")

    @render.ui
    @reactive.event(input.btn_run)
    def predicted_risk():
        return ui.markdown(
            "##### **Predicted Risk**"
        )

    # ==================== pie chart ====================
    @render_plotly
    @reactive.event(input.btn_run)
    def pie_chart():
        data = results()
        # print("pie_chart", data)
        if data is None:
            return px.pie()
        prob = data["prob"]
        df_pie = pd.DataFrame({"prob": prob, "label": ["Normal", "Prediabetes"]})
        fig = px.pie(
            df_pie,
            values="prob",
            names="label",
            color="label",
            color_discrete_map={"Normal": "green", "Prediabetes": "red"},
            category_orders={"label": ["Normal", "Prediabetes"]}
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    # ==================== Health advice ====================
    @render.ui
    @reactive.event(input.btn_run)
    def suggestion():
        data = results()
        if data is None:
            return ui.div()
        prob = data["prob"]
        cls = data["predicted_class"]

        if cls == 0:
            return ui.markdown("""
                #### ✅ Your glucose metabolism is in good condition.  
                Please maintain a healthy lifestyle with a balanced diet, regular exercise, 
                and periodic physical examinations."""
                               )
        else:
            return ui.markdown(f"""
                #### ⚠️ There is a **{prob[1] * 100:.1f}%** probability of prediabetes. 
                Prediabetes is reversible with timely intervention.

                **Recommended lifestyle adjustments:**
                - Modify your diet: reduce sugar and fat intake, increase vegetables, fruits and fiber.
                - Quit smoking and limit alcohol intake.
                - Engage in moderate aerobic exercise daily and maintain a proper sleep schedule.
                - Further diagnosis by preforming 2-hour OGTT and HbA1C tests, and follow up regularly with your doctor.

                Always consult your doctor before starting new exercise or major diet changes, 
                especially if you have other health conditions. They can monitor your progress with regular blood 
                glucose checks (e.g., HbA1C tests) and discuss if medication might be needed in some cases.  
                Early action works - stay consistent!
            """)

    # ==================== SHAP Waterfall ====================
    @render.plot
    @reactive.event(input.btn_run)
    def shap_waterfall():
        data = results()
        if data is None:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Click Run to see SHAP explanation", ha="center")
            return fig

        cls = data["predicted_class"]
        shap_val = data["shap_val"]
        shap.plots.waterfall(shap_val, max_display=13, show=False)
        return plt.gcf()

    # ==================== Note ====================
    @render.ui
    @reactive.event(input.btn_run)
    def feature_note():
        return ui.markdown(
            """##### Note on Feature Interpretation:

The model includes several closely related metabolic indicators-TG, FPG, and TyG. 
Because these measures share common biological information (e.g., lipid and glucose metabolism), their individual 
contribution scores shown above may appear distributed across these features.  
For clinical decision-making, we recommend interpreting TG, FPG, and TyG together as a combined signal 
of metabolic health, rather than focusing on any single value in isolation. This does not affect the model's predictive 
accuracy but helps ensure a holistic view of metabolic risk.

Abbreviation: BMI, body mass index; SBP, systolic blood pressure; FPG, fasting plasma glucose; TG, triglyceride; 
TC, total cholesterol;  WBC, white blood cell count; RBC, red blood cell count; HGB, hemoglobin; PLT, platelet count; 
TyG, triglyceride glucose index; MHR, monocyte to high-density lipoprotein cholesterol ratio."""
        )


# ====================== Start ======================
app = App(app_ui, server)

if __name__ == "__main__":
    print("Access http://127.0.0.1:8000")
    app.run()
