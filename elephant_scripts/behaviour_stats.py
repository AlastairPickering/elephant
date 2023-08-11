"""
Behaviour Statistics Module.

Functions to analyse the relationship between the n=1 reduced acoustic feature
dimension (UMAP1) and our fixed and random effects of interest, behavioural
context, distress, age and sex. These include plotting boxplots to visualise
the relationships, testing for collinearity and zero variance as well as
specifying and running the mixed effects model.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson


def create_boxplots(umap_df):
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Function to add a mean line to each boxplot
    def add_mean_line(data, y, ax):
        mean_value = data[y].median()
        ax.axhline(
            y=mean_value, color="red", linestyle="dashed", label=f"Mean"
        )
        ax.legend(loc="upper left")

    # Plot 1
    filtered_df = umap_df[
        ~umap_df["Final_Category"].isin(["Unknown", "Unspecific"])
    ]
    ax1 = axes[0, 0]
    sns.boxplot(
        data=filtered_df,
        x="Final_Category",
        y="UMAP1",
        width=0.6,
        showfliers=True,
        ax=ax1,
    )
    add_mean_line(filtered_df, "UMAP1", ax1)  # Add mean line to the plot
    ax1.set_title(
        "a) Vocalisation~Behaviour", fontsize=16, loc="left"
    )  # Adjust title font size and alignment

    # Plot 2
    filtered_df_a = umap_df[~umap_df["Age"].isin(["Unknown"])]
    ax2 = axes[0, 1]
    sns.boxplot(
        data=filtered_df_a,
        x="Age",
        y="UMAP1",
        width=0.6,
        showfliers=True,
        ax=ax2,
    )
    add_mean_line(filtered_df_a, "UMAP1", ax2)  # Add mean line to the plot
    ax2.set_title(
        "b) Vocalisation~Age", fontsize=16, loc="left"
    )  # Adjust title font size and alignment

    # Plot 3
    filtered_df_s = umap_df[~umap_df["Sex"].isin(["Unknown"])]
    ax3 = axes[1, 0]
    sns.boxplot(
        data=filtered_df_s,
        x="Sex",
        y="UMAP1",
        width=0.6,
        showfliers=True,
        ax=ax3,
    )
    add_mean_line(filtered_df_s, "UMAP1", ax3)  # Add mean line to the plot
    ax3.set_title(
        "c) Vocalisation~Sex", fontsize=16, loc="left"
    )  # Adjust title font size and alignment

    # Plot 4
    filtered_dfd = umap_df[~umap_df["Distress"].isin(["un"])]
    ax4 = axes[1, 1]
    sns.boxplot(
        data=filtered_dfd,
        x="Distress",
        y="UMAP1",
        width=0.6,
        showfliers=True,
        ax=ax4,
    )
    add_mean_line(filtered_dfd, "UMAP1", ax4)  # Add mean line to the plot
    ax4.set_title(
        "d) Vocalisation~Distress", fontsize=16, loc="left"
    )  # Adjust title font size and alignment

    # Adjust x-axis label font size and labels
    for ax in axes.flatten():
        ax.tick_params(axis="x", labelsize=12)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45
        )  # Rotate x-axis labels

    plt.tight_layout()
    plt.show()


def check_collinearity(df, correlation_threshold=0.7):
    df_corr = df[["Age_Sex", "Final_Category", "Distress_factor"]].corr()
    collinear_vars = []

    for i in range(len(df_corr.columns)):
        for j in range(i + 1, len(df_corr.columns)):
            if abs(df_corr.iloc[i, j]) > correlation_threshold:
                collinear_vars.append((df_corr.columns[i], df_corr.columns[j]))

    if len(collinear_vars) > 0:
        print("Collinearity detected between the following variables:")
        for var1, var2 in collinear_vars:
            print(f"{var1} and {var2}")
    else:
        print("No collinearity detected among the independent variables.")


def preprocess_data(df):
    final_cat_df = df[
        (df["Final_Category"] != "Unknown")
        & (df["Final_Category"] != "Unspecific")
    ]

    numeric_vars = ["Age_Sex"]
    factorized_labels = {}

    for col in numeric_vars:
        factorized_labels[col] = pd.factorize(final_cat_df[col])[1]
        final_cat_df[col] = pd.factorize(final_cat_df[col])[0]

    return final_cat_df


def fit_mixed_effects_model(df):
    formula = 'UMAP1 ~ C(Final_Category, Treatment("Separation")) + C(Distress_factor)'
    behaviour_model = smf.mixedlm(formula, df, groups=df["Age_Sex"])
    behaviour_model_result = behaviour_model.fit()
    # Print the summary of the model
    print(behaviour_model_result.summary())

    return behaviour_model_result


def analyze_model_performance(df, behaviour_model_result):
    behaviour_model_performance = df[["UMAP1"]]
    behaviour_model_performance[
        "residuals"
    ] = behaviour_model_result.resid.values
    behaviour_model_performance["Final_Category"] = df.Final_Category
    behaviour_model_performance[
        "Predicted_UMAP"
    ] = behaviour_model_result.fittedvalues

    sns.lmplot(
        x="Predicted_UMAP", y="residuals", data=behaviour_model_performance
    )

    mae = mean_absolute_error(
        behaviour_model_performance["UMAP1"],
        behaviour_model_performance["Predicted_UMAP"],
    )
    print("Mean Absolute Error:", mae)

    rmse = mean_squared_error(
        behaviour_model_performance["UMAP1"],
        behaviour_model_performance["Predicted_UMAP"],
        squared=False,
    )
    print("Root Mean Squared Error:", rmse)

    r2 = r2_score(
        behaviour_model_performance["UMAP1"],
        behaviour_model_performance["Predicted_UMAP"],
    )
    print("R-squared Score:", r2)

    nrmse = rmse / (max(df.UMAP1) - min(df.UMAP1))
    print("Normalised RMSE:", nrmse)


# Normality test functions
def calculate_skewness(data):
    return data.skew()


def calculate_kurtosis(data):
    return data.kurtosis()


# Function to perform normality tests
def perform_normality_tests(data):
    skewness = calculate_skewness(data)
    kurtosis = calculate_kurtosis(data)

    # Shapiro-Wilk test
    shapiro_test_stat, shapiro_p_value = stats.shapiro(data)
    shapiro_result = (
        "Data is normally distributed."
        if shapiro_p_value > 0.05
        else "Data is not normally distributed."
    )

    # Anderson-Darling test
    (
        anderson_test_stat,
        anderson_critical_values,
        anderson_p_values,
    ) = stats.anderson(data, dist="norm")
    anderson_p_value = max(anderson_p_values)
    anderson_result = (
        "Data is normally distributed."
        if anderson_p_value > 0.05
        else "Data is not normally distributed."
    )

    # Jarque-Bera test
    jb_test_stat, jb_p_value = stats.jarque_bera(data)
    jb_result = (
        "Data is normally distributed."
        if jb_p_value > 0.05
        else "Data is not normally distributed."
    )

    # Durbin-Watson test
    dw_statistic = durbin_watson(data)
    dw_result = (
        "Data is normally distributed."
        if (1.5 <= dw_statistic <= 2.5)
        else "Data is not normally distributed."
    )

    return {
        "Skewness": {
            "Score": skewness,
            "Result": "Data is normally distributed."
            if (-1 < skewness < 1)
            else "Data is not normally distributed.",
        },
        "Kurtosis": {
            "Score": kurtosis,
            "Result": "Data is normally distributed."
            if (-1 < kurtosis < 1)
            else "Data is not normally distributed.",
        },
        "Shapiro-Wilk": {
            "Test Statistic": shapiro_test_stat,
            "p-value": shapiro_p_value,
            "Result": shapiro_result,
        },
        "Anderson-Darling": {
            "Test Statistic": anderson_test_stat,
            "p-value": anderson_p_value,
            "Result": anderson_result,
        },
        "Jarque-Bera": {
            "Test Statistic": jb_test_stat,
            "p-value": jb_p_value,
            "Result": jb_result,
        },
        "Durbin-Watson": {"Score": dw_statistic, "Result": dw_result},
    }


def inspect_residuals(df, behaviour_model_result):
    residuals = behaviour_model_result.resid

    # Scatterplot
    plt.scatter(df["UMAP1"], residuals)
    plt.xlabel("UMAP1")
    plt.ylabel("Residuals")
    plt.title("Scatterplot of UMAP1 vs. Residuals")
    plt.show()

    # Histogram
    bin_width = 3
    bins = np.arange(min(residuals), max(residuals) + bin_width, bin_width)

    plt.hist(residuals, bins=bins)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals")
    plt.show()

    # Q-Q plot
    stats.probplot(residuals, plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.show()


print("Functions for Statistical Analysis successfully loaded")
