import pandas as pd
import statsmodels.api as sm
from main import load_data, run_ols, extract_significant, NUMERIC_COLS


def get_insignificant_cols(results, alpha=0.05):
    """Return column names with p-value > alpha (excluding const)."""
    table = results.summary().tables[1]
    df = pd.DataFrame(table.data[1:], columns=table.data[0])
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    insignificant = df[df['P>|t|'] > alpha]
    return [c.strip() for c in insignificant.iloc[:, 0] if c.strip() != 'const']


def run_ols_no_const(y, x):
    model = sm.OLS(y, x.astype(float))
    return model.fit()


def iterative_ols(y, x, alpha=0.05):
    """Repeatedly remove insignificant variables and refit until all are significant."""
    step = 0
    use_const = True
    while True:
        step += 1
        results = run_ols(y, x) if use_const else run_ols_no_const(y, x)
        insignificant = get_insignificant_cols(results, alpha)

        print(f"=== Step {step} ({x.shape[1]} variables, const={'yes' if use_const else 'no'}) ===")
        print(results.summary().tables[0])
        print(extract_significant(results, alpha))
        print()

        if not insignificant:
            print("All remaining variables are significant.\n")
            break

        print(f"Removing {len(insignificant)} insignificant variables: {insignificant}\n")
        x = x.drop(columns=[c for c in insignificant if c in x.columns])
        use_const = False

    return results


def main():
    math_df_encoded = load_data()
    y = math_df_encoded["G3"]
    x = math_df_encoded.drop(columns=["G1", "G2", "G3"])

    print("Without G1 and G2 — iterative removal of insignificant variables (no const)\n")
    iterative_ols(y, x)


if __name__ == "__main__":
    main()
