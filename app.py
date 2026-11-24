import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Student Performance & Academic Risk", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction and Academic Risk Analysis")
st.caption("SEC C | Data Science & Machine Learning Project")

st.info("The included CSV is an illustrative synthetic dataset created for this academic demonstration. Replace it with authorized, anonymized student data for real-world use.")

uploaded = st.sidebar.file_uploader("Upload your student CSV (optional)", type=["csv"])
data_path = "student_performance.csv"
try:
    raw = pd.read_csv(uploaded) if uploaded else pd.read_csv(data_path)
except Exception as e:
    st.error(f"Could not read the dataset: {e}")
    st.stop()

required = ["attendance", "study_hours", "previous_marks", "assignment_marks", "internal_marks", "final_marks"]
missing_cols = [c for c in required if c not in raw.columns]
if missing_cols:
    st.error("Missing required columns: " + ", ".join(missing_cols))
    st.stop()

# Data cleaning: remove duplicates, coerce numeric values, impute missing values, cap outliers.
data = raw.copy().drop_duplicates()
for c in required:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data = data.dropna(subset=["final_marks"])
for c in required[:-1]:
    data[c] = data[c].fillna(data[c].median())
for c in required:
    data[c] = data[c].clip(0, 100) if c != "study_hours" else data[c].clip(0, 24)
data["pass"] = (data["final_marks"] >= 40).astype(int)

st.sidebar.markdown("---")
st.sidebar.write(f"**Rows after cleaning:** {len(data)}")
st.sidebar.write(f"**Duplicates removed:** {len(raw) - len(raw.drop_duplicates())}")
st.sidebar.write(f"**Missing values remaining:** {int(data[required].isna().sum().sum())}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview & EDA", "ML Predictions", "Student Risk Checker", "Methodology"])

with tab1:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(data))
    c2.metric("Average final marks", f"{data.final_marks.mean():.1f}/100")
    c3.metric("Pass rate", f"{data['pass'].mean()*100:.1f}%")
    c4.metric("Average attendance", f"{data.attendance.mean():.1f}%")
    st.dataframe(data.head(12), use_container_width=True)
    st.subheader("Academic factor relationships")
    corr = data[required].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax); fig.tight_layout(); st.pyplot(fig)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.hist(data.final_marks, bins=15, edgecolor="black")
        ax.set_xlabel("Final examination marks"); ax.set_ylabel("Number of students")
        ax.set_title("Distribution of final marks"); st.pyplot(fig)
    with right:
        fig, ax = plt.subplots()
        ax.scatter(data.attendance, data.final_marks, alpha=.6)
        ax.set_xlabel("Attendance (%)"); ax.set_ylabel("Final marks")
        ax.set_title("Attendance vs final marks"); st.pyplot(fig)

X_cols = required[:-1]
X = data[X_cols]
y_reg = data["final_marks"]
y_cls = data["pass"]
X_train, X_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=.2, random_state=42)
_, _, yc_train, yc_test = train_test_split(X, y_cls, test_size=.2, random_state=42, stratify=y_cls)

with tab2:
    st.subheader("1. Linear Regression — predict final marks")
    reg = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LinearRegression())])
    reg.fit(X_train, yr_train)
    pred = np.clip(reg.predict(X_test), 0, 100)
    a,b,c = st.columns(3)
    a.metric("MAE", f"{mean_absolute_error(yr_test, pred):.2f}")
    b.metric("RMSE", f"{np.sqrt(mean_squared_error(yr_test, pred)):.2f}")
    c.metric("R² score", f"{r2_score(yr_test, pred):.3f}")
    st.caption("Lower MAE/RMSE is better; R² closer to 1 indicates a better fit. Scores depend on the dataset.")

    st.subheader("2. Logistic Regression — pass/fail prediction")
    clf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    clf.fit(X_train, yc_train)
    cls_pred = clf.predict(X_test)
    accuracy = accuracy_score(yc_test, cls_pred)
    cv_scores = cross_val_score(clf, X, y_cls, cv=5, scoring="accuracy")
    a,b = st.columns(2)
    a.metric("Test accuracy", f"{accuracy*100:.1f}%")
    b.metric("5-fold CV accuracy", f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    st.text("Confusion matrix (actual rows × predicted columns)")
    st.dataframe(pd.DataFrame(confusion_matrix(yc_test, cls_pred), index=["Actual Fail","Actual Pass"], columns=["Predicted Fail","Predicted Pass"]))

    st.subheader("3. KNN — similar learning patterns")
    knn_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", KNeighborsClassifier())])
    search = GridSearchCV(knn_pipe, {"model__n_neighbors":[3,5,7,9,11]}, cv=5, scoring="accuracy")
    search.fit(X_train, yc_train)
    st.write(f"Best KNN setting: **k = {search.best_params_['model__n_neighbors']}** | Cross-validation accuracy: **{search.best_score_*100:.1f}%**")
    st.write("KNN predicts pass/fail using the labels of the most similar students in the training data.")

    st.subheader("4. K-Means — performance segmentation")
    cluster_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", KMeans(n_clusters=3, random_state=42, n_init=10))])
    labels = cluster_pipe.fit_predict(X)
    clustered = data.copy()
    clustered["cluster"] = labels
    means = clustered.groupby("cluster")["final_marks"].mean().sort_values()
    names = {means.index[0]:"At-Risk", means.index[1]:"Average", means.index[2]:"High Performing"}
    clustered["segment"] = clustered["cluster"].map(names)
    st.dataframe(clustered.groupby("segment")[required].mean().round(1), use_container_width=True)
    st.caption("Cluster names are assigned after sorting clusters by their average final marks; K-Means itself does not know these labels.")

    st.subheader("Important performance factors")
    importance = permutation_importance(reg, X_test, yr_test, n_repeats=10, random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.Series(importance.importances_mean, index=X_cols).sort_values()
    fig, ax = plt.subplots()
    imp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Decrease in model score when factor is shuffled")
    ax.set_title("Permutation feature importance")
    st.pyplot(fig)

with tab3:
    st.subheader("Enter a student's academic details")
    c1,c2,c3 = st.columns(3)
    with c1:
        att = st.slider("Attendance (%)", 0, 100, 75)
        hours = st.slider("Study hours per day", 0.0, 15.0, 3.0, .5)
    with c2:
        prev = st.slider("Previous marks", 0, 100, 60)
        assign = st.slider("Assignment marks", 0, 100, 65)
    with c3:
        internal = st.slider("Internal assessment marks", 0, 100, 60)
    row = pd.DataFrame([[att,hours,prev,assign,internal]], columns=X_cols)
    predicted_marks = float(np.clip(reg.predict(row)[0],0,100))
    pass_probability = float(clf.predict_proba(row)[0][1])
    knn_result = int(search.predict(row)[0])
    if pass_probability >= .75 and att >= 70:
        risk = "Low"
    elif pass_probability >= .45:
        risk = "Medium"
    else:
        risk = "High"
    segment = "High Performing" if predicted_marks >= 70 else ("Average" if predicted_marks >= 45 else "At-Risk")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Predicted final marks", f"{predicted_marks:.1f}/100")
    c2.metric("Pass probability", f"{pass_probability*100:.1f}%")
    c3.metric("Academic risk", risk)
    c4.metric("Performance segment", segment)
    st.write("KNN neighbour-based prediction:", "**Pass**" if knn_result else "**Fail**")
    st.warning("This is an educational prediction, not a definitive judgement about a student. Use it to support—not replace—teacher review and student support.")

with tab4:
    st.markdown("""
    **Workflow**
    1. Collect academic attributes from the CSV file.
    2. Preprocess: remove duplicate rows, convert values to numeric, impute missing feature values with medians, and cap impossible ranges.
    3. Explore the data using summary statistics, histograms, scatter plots and a correlation heatmap.
    4. Train Linear Regression for final marks and Logistic Regression for pass/fail.
    5. Use KNN to classify a student based on nearby examples and GridSearchCV to select k.
    6. Use K-Means (k=3) to group students by learning patterns, then interpret clusters using their average marks.
    7. Evaluate with a holdout test set and 5-fold cross-validation.

    **Pass criterion:** final marks >= 40 (editable in the code).

    **Academic risk rule:** Low = pass probability >= 75% and attendance >= 70%; Medium = pass probability >= 45%; otherwise High. This transparent rule is a project heuristic, not a universal academic standard.

    **Limitations:** The included dataset is synthetic and only demonstrates the workflow. Real results require consented, anonymized, representative institutional data. Correlation and feature importance do not prove causation.
    """)
