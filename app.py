import io
import textwrap
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="Student Stress Predictor",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Stress Predictor")
st.write(
    "Explore student stress patterns and estimate a stress category "
    "from survey responses. This is an educational tool, not a diagnosis."
)

DATA_FILE = "StressLevelDataset.csv"
TARGET = "stress_level"

CLASS_NAMES = {
    0: "Low Stress",
    1: "Moderate Stress",
    2: "High Stress"
}

# Friendly labels and explanations for common dataset features.
FEATURE_INFO = {
    "anxiety_level": {
        "label": "Anxiety level",
        "help": "How much anxiety the student reports.",
        "low": "Lowest anxiety rating",
        "high": "Highest anxiety rating"
    },
    "self_esteem": {
        "label": "Self-esteem",
        "help": "The student's reported self-esteem score.",
        "low": "Lowest self-esteem rating",
        "high": "Highest self-esteem rating"
    },
    "mental_health_history": {
        "label": "Mental health history",
        "help": "Use the survey's coding for this field; check the dataset codebook.",
        "low": "Lowest coded value",
        "high": "Highest coded value"
    },
    "depression": {
        "label": "Depression-related score",
        "help": "The dataset's depression-related measure; this is not a diagnosis.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "headache": {
        "label": "Headache frequency / rating",
        "help": "The headache-related measure recorded in the survey.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "blood_pressure": {
        "label": "Blood pressure measure",
        "help": "Enter the value using the dataset's scale, not a medical reading unless the dataset specifies that.",
        "low": "Lowest recorded value",
        "high": "Highest recorded value"
    },
    "sleep_quality": {
        "label": "Sleep quality",
        "help": "The student's reported sleep-quality rating.",
        "low": "Lowest sleep-quality rating",
        "high": "Highest sleep-quality rating"
    },
    "breathing_problem": {
        "label": "Breathing-problem rating",
        "help": "The breathing-related measure from the survey.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "noise_level": {
        "label": "Noise level",
        "help": "The noise-related measure in the survey.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "living_conditions": {
        "label": "Living conditions",
        "help": "The student's survey rating for living conditions.",
        "low": "Lowest living-condition rating",
        "high": "Highest living-condition rating"
    },
    "safety": {
        "label": "Feeling of safety",
        "help": "The safety-related measure recorded in the survey.",
        "low": "Lowest safety rating",
        "high": "Highest safety rating"
    },
    "basic_needs": {
        "label": "Access to basic needs",
        "help": "The survey's measure of access to basic needs.",
        "low": "Lowest basic-needs rating",
        "high": "Highest basic-needs rating"
    },
    "academic_performance": {
        "label": "Academic performance",
        "help": "The student's academic-performance measure in the dataset.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "study_load": {
        "label": "Study workload",
        "help": "The student's reported study workload.",
        "low": "Lowest workload rating",
        "high": "Highest workload rating"
    },
    "teacher_student_relationship": {
        "label": "Teacher–student relationship",
        "help": "The student's survey rating of their relationship with teachers.",
        "low": "Lowest relationship rating",
        "high": "Highest relationship rating"
    },
    "future_career_concerns": {
        "label": "Future career concerns",
        "help": "The student's reported concern about future career plans.",
        "low": "Lowest concern rating",
        "high": "Highest concern rating"
    },
    "social_support": {
        "label": "Social support",
        "help": "The student's reported support from other people.",
        "low": "Lowest support rating",
        "high": "Highest support rating"
    },
    "peer_pressure": {
        "label": "Peer pressure",
        "help": "The student's reported peer-pressure measure.",
        "low": "Lowest pressure rating",
        "high": "Highest pressure rating"
    },
    "extracurricular_activities": {
        "label": "Extracurricular activities",
        "help": "The survey's measure of participation in extracurricular activities.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
    "bullying": {
        "label": "Bullying-related measure",
        "help": "The survey's bullying-related measure; it is not a judgement about the student.",
        "low": "Lowest recorded rating",
        "high": "Highest recorded rating"
    },
}


def friendly_name(column_name):
    info = FEATURE_INFO.get(column_name, {})
    return info.get("label", column_name.replace("_", " ").replace("-", " ").title())


def stress_label(value):
    try:
        numeric_value = int(float(value))
        return CLASS_NAMES.get(numeric_value, f"Class {value}")
    except (ValueError, TypeError):
        return str(value)


def get_stress_suggestions(predicted_value):
    """Provide supportive, non-clinical suggestions for each predicted category."""
    try:
        level = int(float(predicted_value))
    except (ValueError, TypeError):
        level = -1

    common = [
        "Keep a regular sleep and wake-up routine, aiming for enough rest for your age.",
        "Break schoolwork into small tasks and work on one task at a time.",
        "Take short breaks while studying; stretch, walk, or rest your eyes.",
        "Try slow, comfortable breathing for a minute or two when you feel overwhelmed.",
        "Move your body in a way you enjoy, such as walking, dancing, cycling, or a sport.",
        "Eat regular meals and drink water throughout the day.",
        "Talk with a trusted friend, family member, teacher, or school counsellor.",
        "Plan your day with a short, realistic to-do list and include free time.",
        "Reduce distractions while studying and take breaks from screens when possible.",
        "Be kind to yourself: one test, grade, or difficult day does not define you."
    ]

    if level == 0:
        tailored = [
            "Keep doing the healthy routines that help you feel balanced.",
            "Notice which activities help you relax and make time for them regularly."
        ]
    elif level == 1:
        tailored = [
            "Choose one main school task to start with, then divide it into smaller steps.",
            "Share what's been worrying you with someone you trust instead of handling everything alone."
        ]
    elif level == 2:
        tailored = [
            "Consider talking soon with a trusted adult or school counsellor about what feels difficult.",
            "If stress is affecting sleep, daily life, or schoolwork, ask a qualified professional for support."
        ]
    else:
        tailored = [
            "Use the suggestions that feel manageable, and consider talking with someone you trust."
        ]

    # Put two tailored recommendations first, then enough common tips to make ten.
    recommendations = tailored + [tip for tip in common if tip not in tailored]
    return recommendations[:10]


# ============================================================
# PDF REPORT
# ============================================================
def create_pdf_report(
    prediction_text,
    model_name,
    model_accuracy,
    all_model_results,
    user_values,
    suggestions,
    report_date
):
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#24476B"),
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#555555")
    ))

    story = [
        Paragraph("Student Stress Prediction Report", styles["ReportTitle"]),
        Paragraph(f"Generated: {report_date}", styles["Normal"]),
        Spacer(1, 0.35 * cm),
        Paragraph("Prediction result", styles["Heading2"]),
        Paragraph(f"<b>Predicted category:</b> {prediction_text}", styles["BodyText"]),
        Paragraph(f"<b>Selected model:</b> {model_name}", styles["BodyText"]),
        Paragraph(
            f"<b>Model test accuracy:</b> {model_accuracy:.2f}%",
            styles["BodyText"]
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Survey values entered", styles["Heading2"])
    ]

    input_rows = [["Feature", "Entered value"]]
    for feature, value in user_values.items():
        safe_feature = friendly_name(feature).replace("&", "&amp;")
        input_rows.append([safe_feature, str(value)])

    input_table = Table(input_rows, colWidths=[10 * cm, 6 * cm], repeatRows=1)
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DCE4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F6F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story.extend([input_table, Spacer(1, 0.35 * cm)])

    story.append(Paragraph("Accuracy comparison (held-out test set)", styles["Heading2"]))
    accuracy_rows = [["Model", "Test accuracy"]]
    for _, row in all_model_results.iterrows():
        accuracy_rows.append([
            str(row["Model"]),
            f'{float(row["Test Accuracy (%)"]):.2f}%'
        ])
    accuracy_table = Table(
        accuracy_rows,
        colWidths=[10 * cm, 6 * cm],
        repeatRows=1
    )
    accuracy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#287D73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6DCE4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F3F6F9")]),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story.extend([accuracy_table, Spacer(1, 0.35 * cm)])

    story.append(Paragraph("10 supportive suggestions", styles["Heading2"]))
    for index, suggestion in enumerate(suggestions, start=1):
        story.append(Paragraph(f"{index}. {suggestion}", styles["BodyText"]))
        story.append(Spacer(1, 0.08 * cm))

    story.extend([
        Spacer(1, 0.25 * cm),
        Paragraph(
            "Important note: This report is generated by a machine-learning model "
            "trained on survey data. Its output is an estimate, not a medical "
            "diagnosis or a measure of a person's worth. If stress feels difficult "
            "to manage or affects daily life, consider speaking with a trusted "
            "adult, school counsellor, or qualified health professional.",
            styles["SmallNote"]
        )
    ])

    document.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)


try:
    data = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find **{DATA_FILE}**. Put it in the same folder as `app.py`."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if TARGET not in data.columns:
    st.error(f"The dataset must contain a `{TARGET}` column.")
    st.stop()

data = data.dropna(subset=[TARGET]).copy()
X = data.drop(columns=[TARGET]).copy()
y = data[TARGET].copy()

non_numeric_columns = X.select_dtypes(exclude=np.number).columns.tolist()
if non_numeric_columns:
    st.error(
        "These input columns are not numeric and need encoding first: "
        + ", ".join(non_numeric_columns)
    )
    st.stop()

try:
    numeric_y = pd.to_numeric(y)
    if np.all(np.isclose(numeric_y, np.round(numeric_y))):
        y = numeric_y.astype(int)
except (ValueError, TypeError):
    pass

class_counts = y.value_counts()
if len(class_counts) < 2 or class_counts.min() < 2:
    st.error("Each target class needs at least two rows to train and test models.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(random_state=42)
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200, random_state=42)
    ),
    "Support Vector Machine (SVM)": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC()
    ),
    "K-Nearest Neighbors (KNN)": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

trained_models = {}
results = []

with st.spinner("Training and evaluating the five models..."):
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append({
            "Model": model_name,
            "Test Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2)
        })
        trained_models[model_name] = model

results_df = pd.DataFrame(results).sort_values(
    by="Test Accuracy (%)", ascending=False
).reset_index(drop=True)

# ============================================================
# TABS
# ============================================================
predict_tab, graph_tab = st.tabs(["🔮 Predict Stress", "📊 Graph"])

with predict_tab:
    st.subheader("Tell us about the student's survey responses")
    st.write(
        "Each control explains what it measures. The lowest and highest values "
        "are shown before you choose a response. The exact scale depends on the dataset."
    )

    selected_model_name = st.selectbox(
        "Choose a prediction model",
        list(models.keys()),
        index=2
    )

    user_values = {}

    with st.form("stress_prediction_form"):
        st.markdown("#### Student survey")
        st.caption(
            "Use the student's actual survey responses if available. "
            "Do not guess medical measurements."
        )

        columns = st.columns(2)
        for index, column_name in enumerate(X.columns):
            series = pd.to_numeric(X[column_name], errors="coerce").dropna()
            if series.empty:
                user_values[column_name] = 0
                continue

            minimum = float(series.min())
            maximum = float(series.max())
            median_value = float(series.median())
            unique_values = sorted(series.unique().tolist())
            feature_info = FEATURE_INFO.get(column_name, {})
            label = friendly_name(column_name)
            help_text = feature_info.get(
                "help",
                "This feature is recorded in the original dataset. Check the dataset documentation for its exact meaning."
            )
            low_label = feature_info.get("low", "Lowest recorded value")
            high_label = feature_info.get("high", "Highest recorded value")

            with columns[index % 2]:
                st.markdown(f"**{label}**")
                st.caption(help_text)
                st.caption(
                    f"⬅ {low_label}: **{minimum:g}**  |  "
                    f"{high_label}: **{maximum:g}** ➡"
                )

                is_integer_feature = np.all(
                    np.isclose(series.to_numpy(), np.round(series.to_numpy()))
                )

                if is_integer_feature and len(unique_values) <= 10:
                    options = [int(value) for value in unique_values]
                    default_index = min(
                        range(len(options)),
                        key=lambda j: abs(options[j] - median_value)
                    )
                    chosen = st.select_slider(
                        "Choose a level",
                        options=options,
                        value=options[default_index],
                        key=f"feature_{column_name}",
                        help=f"Available values in the dataset: {options}"
                    )
                    # Show a simple endpoint guide for ordinal-style coded fields.
                    st.caption(
                        f"Scale guide: {options[0]} = lowest available code; "
                        f"{options[-1]} = highest available code."
                    )
                    user_values[column_name] = chosen
                else:
                    step = 1.0 if is_integer_feature else max((maximum - minimum) / 100, 0.01)
                    chosen = st.slider(
                        "Choose a value",
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(median_value, minimum), maximum),
                        step=step,
                        key=f"feature_{column_name}",
                        help=f"Observed range in this dataset: {minimum:g} to {maximum:g}"
                    )
                    user_values[column_name] = chosen

        submitted = st.form_submit_button(
            "Predict Stress",
            type="primary",
            use_container_width=True
        )

    

    if submitted:
        input_df = pd.DataFrame([user_values], columns=X.columns)
        selected_model = trained_models[selected_model_name]
        prediction = selected_model.predict(input_df)[0]
        selected_accuracy = accuracy_score(
            y_test, selected_model.predict(X_test)
        ) * 100
        prediction_text = stress_label(prediction)
        suggestions = get_stress_suggestions(prediction)

        st.divider()
        st.subheader("Your result")
        st.markdown(f"## {prediction_text}")
        st.metric(
            f"{selected_model_name} test accuracy",
            f"{selected_accuracy:.2f}%"
        )
        st.info(
            "Model accuracy describes performance on a separate test set. "
            "It is not the probability that this individual prediction is correct."
        )

        st.subheader("10 supportive ways to manage stress")
        st.write(
            "Choose the suggestions that feel realistic for you. You do not need "
            "to do all ten at once."
        )
        for index, suggestion in enumerate(suggestions, start=1):
            st.markdown(f"**{index}.** {suggestion}")

        st.download_button(
            label="📄 Download stress report as PDF",
            data=create_pdf_report(
                prediction_text=prediction_text,
                model_name=selected_model_name,
                model_accuracy=selected_accuracy,
                all_model_results=results_df,
                user_values=user_values,
                suggestions=suggestions,
                report_date=datetime.now().strftime("%d %B %Y, %I:%M %p")
            ),
            file_name="student_stress_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    st.subheader("Test accuracy of all five models")
    st.dataframe(results_df, hide_index=True, use_container_width=True)
    best_row = results_df.iloc[0]
    st.caption(
        f"Highest test accuracy on this split: {best_row['Model']} "
        f"({best_row['Test Accuracy (%)']:.2f}%). This does not guarantee "
        "it will be best on every new dataset."
    )
with graph_tab:
    st.subheader("Explore the dataset — 8 colourful graphs")
    st.write(
        "Each graph includes two short interpretation points. These are prompts "
        "for exploring your actual results, not claims about what the data must show."
    )

    sns.set_theme(style="whitegrid", palette="Set2")
    plot_data = data.copy()
    plot_data["Stress Category"] = plot_data[TARGET].map(stress_label)
    class_order = [stress_label(value) for value in sorted(y.unique())]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    preferred_features = [
        "anxiety_level",
        "self_esteem",
        "sleep_quality",
        "study_load",
        "future_career_concerns",
        "academic_performance",
        "social_support",
        "peer_pressure",
        "headache",
        "bullying",
        "depression",
        "teacher_student_relationship"
    ]
    graph_features = [f for f in preferred_features if f in numeric_features]
    graph_features += [f for f in numeric_features if f not in graph_features]

    def show_insights(points):
        for point in points:
            st.markdown(f"- {point}")

    # Graph 1: Count plot
    st.markdown("### 1. How many records belong to each stress category?")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=plot_data,
        x="Stress Category",
        order=class_order,
        hue="Stress Category",
        palette="Set2",
        legend=False,
        ax=ax
    )
    ax.set_xlabel("Stress category")
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of stress categories")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    show_insights([
        "Compare the bar heights to see which stress category is most common in this dataset.",
        "If one class has far fewer records, use precision, recall and F1-score alongside accuracy."
    ])

    # Graph 2: Histograms by category
    if graph_features:
        feature = "anxiety_level" if "anxiety_level" in numeric_features else graph_features[0]
        st.markdown(f"### 2. Distribution of {friendly_name(feature).lower()}")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=plot_data,
            x=feature,
            hue="Stress Category",
            multiple="layer",
            kde=True,
            palette="Set2",
            ax=ax
        )
        ax.set_title(f"{friendly_name(feature)} across stress categories")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "Look for ranges where the category distributions overlap or differ.",
            "A visible overlap means this feature alone may not separate stress categories reliably."
        ])

    # Graph 3: Violin plot
    if len(graph_features) > 1:
        feature = "self_esteem" if "self_esteem" in numeric_features else graph_features[1]
        st.markdown(f"### 3. {friendly_name(feature)} across stress categories")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(
            data=plot_data,
            x="Stress Category",
            y=feature,
            hue="Stress Category",
            order=class_order,
            palette="Set3",
            legend=False,
            inner="quartile",
            ax=ax
        )
        ax.set_title(f"Distribution of {friendly_name(feature)}")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "The width of each violin shows where values are more concentrated.",
            "Compare medians and spreads, while remembering that group differences do not prove causation."
        ])

    # Graph 4: Bar chart of category means
    if len(graph_features) > 2:
        feature = "sleep_quality" if "sleep_quality" in numeric_features else graph_features[2]
        st.markdown(f"### 4. Average {friendly_name(feature).lower()} by stress category")
        means = plot_data.groupby("Stress Category", observed=False)[feature].mean().reindex(class_order)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(means.index, means.values, color=sns.color_palette("Set2", len(means)))
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylabel(f"Average {friendly_name(feature).lower()}")
        ax.set_xlabel("Stress category")
        ax.set_title(f"Average {friendly_name(feature)}")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "This chart compares the average feature value across the stress groups.",
            "Check the original scale before deciding whether a higher or lower average is favourable."
        ])

    # Graph 5: Scatter plot
    if len(graph_features) > 3:
        x_feature = "study_load" if "study_load" in numeric_features else graph_features[3]
        y_feature = "future_career_concerns" if "future_career_concerns" in numeric_features else graph_features[4]
        st.markdown(
            f"### 5. {friendly_name(x_feature)} vs {friendly_name(y_feature)}"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=plot_data,
            x=x_feature,
            y=y_feature,
            hue="Stress Category",
            hue_order=class_order,
            palette="Set1",
            alpha=0.75,
            ax=ax
        )
        ax.set_title("Comparing two student survey factors")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "Look for clusters of points and whether stress categories overlap.",
            "Overlapping colours suggest that these two features alone may not predict the category."
        ])

    # Graph 6: Stacked proportion chart
    if len(graph_features) > 4:
        feature = "academic_performance" if "academic_performance" in numeric_features else graph_features[5]
        st.markdown(f"### 6. Stress category mix across {friendly_name(feature).lower()} values")
        crosstab = pd.crosstab(
            plot_data[feature].round(2),
            plot_data["Stress Category"],
            normalize="index"
        ).reindex(columns=class_order, fill_value=0)
        # Limit overcrowded x-axis to the 15 most frequent feature values.
        if len(crosstab) > 15:
            frequent_values = plot_data[feature].round(2).value_counts().head(15).index
            crosstab = crosstab.loc[crosstab.index.isin(frequent_values)].sort_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        crosstab.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            colormap="Set3",
            width=0.85
        )
        ax.set_xlabel(friendly_name(feature))
        ax.set_ylabel("Proportion within feature value")
        ax.set_title("Proportion of stress categories")
        ax.legend(title="Stress category", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "Each bar shows the mix of stress categories for a feature value.",
            "Small groups can look unusual by chance, so check the number of records behind each bar."
        ])

    # Graph 7: Correlation heatmap
    st.markdown("### 7. Correlation heatmap")
    heatmap_columns = numeric_features[:15]
    if len(heatmap_columns) >= 2:
        correlation = plot_data[heatmap_columns].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            correlation,
            cmap="Spectral",
            center=0,
            annot=len(heatmap_columns) <= 8,
            fmt=".2f",
            linewidths=0.3,
            ax=ax
        )
        ax.set_title("Correlation between numeric features")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "Values near +1 indicate a strong positive linear association; values near -1 indicate a strong negative association.",
            "Correlation does not establish causation, and coded survey categories may need careful interpretation."
        ])
    else:
        st.info("At least two numeric features are needed for a correlation heatmap.")

    # Graph 8: Boxen plot (different from the other plot types)
    if len(graph_features) > 5:
        feature = "peer_pressure" if "peer_pressure" in numeric_features else graph_features[6]
        st.markdown(f"### 8. Detailed distribution of {friendly_name(feature).lower()}")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxenplot(
            data=plot_data,
            x="Stress Category",
            y=feature,
            hue="Stress Category",
            order=class_order,
            palette="muted",
            legend=False,
            ax=ax
        )
        ax.set_title(f"{friendly_name(feature)} by stress category")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        show_insights([
            "The boxen plot displays the spread of values in more detail than a simple average.",
            "Compare the distributions between categories rather than assuming one feature determines stress."
        ])

st.divider()
st.caption(
    "Privacy note: the report is created in your browser session from the values you enter. "
    "Educational use only; this model is not a medical diagnostic tool."
)
