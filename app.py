import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

st.set_page_config(page_title="NovaRetail Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_excel("NR_dataset.xlsx")

    # Imputation justification: "label" is the only field containing NULL
    # values in the source data. "Stable" is the most neutral behavioral
    # classification (it does not imply growth or decline in either
    # direction), so imputing missing "label" values with "Stable"
    # introduces the least distortion to the overall distribution of the
    # other three categories (Growth, Promising, Decline).
    df["label"] = df["label"].fillna("Stable")

    return df

df = load_data()

CATEGORY_FIELDS = [
    "CustomerAgeGroup",
    "CustomerGender",
    "CustomerRegion",
    "RetailChannel",
    "ProductCategory",
]

LABEL_ORDER = ["Growth", "Promising", "Stable", "Decline"]
LABEL_COLORS = {
    "Growth": "#2ca02c",
    "Promising": "#1f77b4",
    "Stable": "#ff7f0e",
    "Decline": "#d62728",
}

st.title("NovaRetail: Interactive Dashboarding for Decision Making")

page = st.sidebar.radio(
    "Select Dashboard View",
    (
        "Population Distribution by Consumer Category",
        "Total Purchase Price by Consumer Category",
        "Average Purchase Price by Consumer Category",
        "Average Customer Satisfaction by Consumer Category",
        "Consumer Behavior by Consumer Category",
    ),
)

def gradient_colors_1_to_5(values):
    cmap = cm.get_cmap("RdYlGn")
    norm = mcolors.Normalize(vmin=1, vmax=5)
    return [cmap(norm(v)) for v in values]

if page == "Population Distribution by Consumer Category":
    st.header("Population Distribution by Consumer Category")
    field = st.selectbox("Select Consumer Category Field", CATEGORY_FIELDS)

    counts = df[field].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_xlabel(field)
    ax.set_ylabel("Population Count")
    ax.set_title(f"Population Distribution by {field}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

elif page == "Total Purchase Price by Consumer Category":
    st.header("Total Purchase Price by Consumer Category")
    field = st.selectbox("Select Consumer Category Field", CATEGORY_FIELDS)

    totals = df.groupby(field)["PurchaseAmount"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(totals.index.astype(str), totals.values, color="#4C72B0")
    ax.set_xlabel(field)
    ax.set_ylabel("Total Purchase Price ($)")
    ax.set_title(f"Total Purchase Price by {field}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

elif page == "Average Purchase Price by Consumer Category":
    st.header("Average Purchase Price by Consumer Category")
    field = st.selectbox("Select Consumer Category Field", CATEGORY_FIELDS)

    averages = df.groupby(field)["PurchaseAmount"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(averages.index.astype(str), averages.values, color="#4C72B0")
    ax.set_xlabel(field)
    ax.set_ylabel("Average Purchase Price ($)")
    ax.set_title(f"Average Purchase Price by {field}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

elif page == "Average Customer Satisfaction by Consumer Category":
    st.header("Average Customer Satisfaction by Consumer Category")
    field = st.selectbox("Select Consumer Category Field", CATEGORY_FIELDS)

    avg_satisfaction = df.groupby(field)["CustomerSatisfaction"].mean().sort_index()
    bar_colors = gradient_colors_1_to_5(avg_satisfaction.values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(avg_satisfaction.index.astype(str), avg_satisfaction.values, color=bar_colors)
    ax.set_xlabel(field)
    ax.set_ylabel("Average Customer Satisfaction (1-5)")
    ax.set_title(f"Average Customer Satisfaction by {field}")
    ax.set_ylim(0, 5)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

elif page == "Consumer Behavior by Consumer Category":
    st.header("Consumer Behavior by Consumer Category")
    field = st.selectbox("Select Consumer Category Field", CATEGORY_FIELDS)

    proportions = (
        df.groupby([field, "label"])
        .size()
        .unstack(fill_value=0)
    )
    proportions = proportions.reindex(columns=LABEL_ORDER, fill_value=0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0) * 100
    proportions = proportions.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(proportions))
    # Stack order from bottom to top: Decline, Stable, Promising, Growth
    for label in reversed(LABEL_ORDER):
        ax.bar(
            proportions.index.astype(str),
            proportions[label].values,
            bottom=bottom,
            color=LABEL_COLORS[label],
            label=label,
        )
        bottom += proportions[label].values

    ax.set_xlabel(field)
    ax.set_ylabel("Proportion of Population (%)")
    ax.set_title(f"Consumer Behavior by {field}")
    ax.set_ylim(0, 100)
    ax.legend(title="Behavior Category", loc="upper right")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
