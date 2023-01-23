import pickle
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.SKETCHY])

app.layout = html.Div(
    className="container",
    children=[
        html.Div(
            children=[
                html.H1("Heart Disease Classification by Machine Learning"),
                html.P("Use this app to diagnose heart disease based on patient information"),
            ]
        ),
        dbc.Row([
            dbc.Col([
                dbc.Form([
                    html.Label("Age in years", className="text-danger"),
                    dbc.Input(id="age-ip", placeholder="Age", min=25, max=85, type="number", className="form-control"),
                ]),
                dbc.Form([
                    html.Label("Sex", className="text-danger"),
                    dcc.Dropdown(
                        id="sex-ip",
                        options=[
                            {"label": "Male", "value": "male"},
                            {"label": "Female", "value": "female"},
                        ],
                        value="male",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Chest Pain Type", className="text-danger"),
                    dcc.Dropdown(
                        id="cp-ip",
                        options=[
                            {"label": "Typical Angina", "value": "typical_angina"},
                            {"label": "Atypical Angina", "value": "atypical_angina"},
                            {"label": "Non-anginal Pain", "value": "non_anginal_pain"},
                            {"label": "Asymptomatic", "value": "asymptomatic"},
                        ],
                        value="typical_angina",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Resting Blood Pressure", className="text-danger"),
                    dbc.Input(
                        id="trestbps-ip",
                        placeholder="in mm/Hg on admission to the hospital",
                        min=80,
                        max=220,
                        type="number",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Serum Cholesterol", className="text-danger"),
                    dbc.Input(
                        id="chol-ip",
                        placeholder="in mg/dl",
                        min=80,
                        max=600,
                        type="number",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Fasting Blood Sugar > 120 mg/dl?", className="text-danger"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Yes", "value": 1},
                            {"label": "No", "value": 0},
                        ],
                        inline=True,
                        value=1,
                        id="fbs-ip",
                        className="form-check form-check-inline"
                    ),
                ]),
                dbc.Form([
                    html.Label("Resting Electro Cardiographic Results", className="text-danger"),
                    dcc.Dropdown(
                        id="restecg-ip",
                        options=[
                            {"label": "Normal", "value": "normal"},
                            {"label": "ST-T Wave Abnormality", "value": "st_t_wave_abnormality"},
                            {"label": "Left Ventricular Hypertrophy", "value": "left_ventricular_hypertrophy"},
                        ],
                        value="normal",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Maximum Heart Rate Achieved", className="text-danger"),
                    dbc.Input(
                        id="thalach-ip",
                        placeholder="in bpm",
                        min=80,
                        max=220,
                        type="number",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Exercise Induced Angina?", className="text-danger"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Yes", "value": 1},
                            {"label": "No", "value": 0},
                        ],
                        value=1,
                        inline=True,
                        id="exang-ip",
                        className="form-check form-check-inline"
                    ),
                ]),
                dbc.Form([
                    html.Label("ST Depression Induced by Exercise relative to rest", className="text-danger"),
                    dbc.Input(
                        id="oldpeak-ip",
                        min=0,
                        max=7,
                        type="number",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("The Slope of the Peak Exercise ST Segment", className="text-danger"),
                    dcc.Dropdown(
                        id="slope-ip",
                        options=[
                            {"label": "Upsloping", "value": "upsloping"},
                            {"label": "Flat", "value": "flat"},
                            {"label": "Downsloping", "value": "downsloping"},
                        ],
                        value="flat",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Number of Major Vessels (0-3) Colored by Flourosopy", className="text-danger"),
                    dbc.Input(
                        id="ca-ip",
                        min=0,
                        max=3,
                        type="number",
                        className="form-control"
                    ),
                ]),
                dbc.Form([
                    html.Label("Thalessemia", className="text-danger"),
                    dcc.Dropdown(
                        id="thal-ip",
                        options=[
                            {"label": "Normal", "value": "normal"},
                            {"label": "Fixed Defect", "value": "fixed_defect"},
                            {"label": "Reversable Defect", "value": "reversable_defect"},
                            {"label": "Permanent Defect", "value": "permanent_defect"},
                        ],
                        value="normal",
                        className="form-control"
                    ),
                ]),
                dbc.Button("Get Result", id="submit-button", name="Diagnose", n_clicks=0, color="primary", className="me-1"),
            ]),
            dbc.Col([
                html.Div(id="results", className="alert alert-success")
            ])
        ])
    ]
)


@app.callback(
    Output("results", "children"),
    Input("submit-button", "n_clicks"),
    Input("age-ip", "value"),
    Input("sex-ip", "value"),
    Input("cp-ip", "value"),
    Input("trestbps-ip", "value"),
    Input("chol-ip", "value"),
    Input("fbs-ip", "value"),
    Input("restecg-ip", "value"),
    Input("thalach-ip", "value"),
    Input("exang-ip", "value"),
    Input("oldpeak-ip", "value"),
    Input("slope-ip", "value"),
    Input("ca-ip", "value"),
    Input("thal-ip", "value"),
)
def get_results(n_clicks, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    if n_clicks == 0:
        return html.Div()

    sex = 0 if sex == "Female" else 1

    with open('heart_disease.pkl', 'rb') as f:
        model = pickle.load(f)

    sample = pd.DataFrame(
        {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,

        },
        index=[0]
    )
    try:
        prediction = round(model.predict_proba(sample).flatten()[1] * 100, 2)
    except:
        return html.Div("Please Complete the form before submitting")

    return f"Probability of Heart Disease is: {prediction} %"


if __name__ == "__main__":
    app.run_server()