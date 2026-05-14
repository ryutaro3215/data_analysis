import pandas as pd
import statsmodels.api as sm


CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERIC_COLS = ['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']


def load_data():
    math_df = pd.read_csv("./student/student-mat.csv", sep=";")
    return pd.get_dummies(math_df, columns=CATEGORICAL_COLS, drop_first=True)


def run_ols(y, x):
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const.astype(float))
    return model.fit()


def extract_significant(results, alpha=0.05):
    table = results.summary().tables[1]
    df = pd.DataFrame(table.data[1:], columns=table.data[0])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    significant = df[df['P>|t|'] <= alpha].copy()
    return pd.DataFrame(significant).sort_values(by='coef', ascending=False)


def main():
    math_df_encoded = load_data()
    y = math_df_encoded["G3"]

    models = {
        "G3 and all attributes": math_df_encoded.drop(columns=["G3"]),
        "G3 and all attributes without G2": math_df_encoded.drop(columns=["G2", "G3"]),
        "G3 and all attributes without G1 and G2": math_df_encoded.drop(columns=["G1", "G2", "G3"]),
    }

    for label, x in models.items():
        results = run_ols(y, x)
        print(f"-----------------{label}-----------------")
        print(results.summary().tables[0])
        print(extract_significant(results))
        print()


if __name__ == "__main__":
    main()
