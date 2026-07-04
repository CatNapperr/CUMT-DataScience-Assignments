import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset():
    """Load California Housing dataset.

    Tries sklearn fetch first (data only). Falls back to local housing.csv if available.
    """
    try:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing(as_frame=True)
        df = data.frame.copy()
        return df
    except Exception:
        # Fallback: user-provided CSV, expected to be housing.csv in current directory
        df = pd.read_csv("housing.csv")
        return df


def one_hot_encode(df, column):
    """Manual one-hot encoding for a single categorical column."""
    categories = sorted(df[column].dropna().unique())
    for cat in categories:
        df[f"{column}__{cat}"] = (df[column] == cat).astype(float)
    df = df.drop(columns=[column])
    return df


def standardize_features(X):
    """Standardize features to mean 0 and std 1 (manual)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    # Avoid division by zero
    std_safe = np.where(std == 0, 1.0, std)
    X_std = (X - mean) / std_safe
    return X_std, mean, std_safe


def add_bias(X):
    """Add bias term (column of ones)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_linear_regression_normal_eq(X, y, l2_lambda=0.0):
    """Closed-form solution for linear regression (optional L2 regularization)."""
    Xb = add_bias(X)
    n_features = Xb.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0.0  # Do not regularize bias
    A = Xb.T @ Xb + l2_lambda * I
    b = Xb.T @ y
    w = np.linalg.pinv(A) @ b
    return w


def predict(X, w):
    Xb = add_bias(X)
    return Xb @ w


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def train_test_split(X, y, test_size=0.3, seed=42, shuffle=True):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    split = int(n * (1.0 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def k_fold_indices(n, k=5, seed=42, shuffle=True):
    """Return list of (train_idx, val_idx) for K-fold CV."""
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def cross_validate(X, y, k=5, seed=42, shuffle=True, l2_lambda=0.0):
    """Manual K-fold cross validation for linear regression."""
    r2_list = []
    mse_list = []
    for train_idx, val_idx in k_fold_indices(len(y), k=k, seed=seed, shuffle=shuffle):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        w = fit_linear_regression_normal_eq(X_train, y_train, l2_lambda=l2_lambda)
        y_pred = predict(X_val, w)
        r2_list.append(r2_score(y_val, y_pred))
        mse_list.append(mse(y_val, y_pred))
    return np.array(r2_list), np.array(mse_list)


def bootstrap_evaluate(X, y, n_bootstrap=100, seed=42):
    """Bootstrap evaluation using out-of-bag (OOB) samples."""
    n = len(y)
    rng = np.random.default_rng(seed)
    r2_list = []
    mse_list = []

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[sample_idx] = False
        oob_idx = np.where(oob_mask)[0]

        if oob_idx.size == 0:
            continue

        X_train, y_train = X[sample_idx], y[sample_idx]
        X_oob, y_oob = X[oob_idx], y[oob_idx]

        w = fit_linear_regression_normal_eq(X_train, y_train)
        y_pred = predict(X_oob, w)
        r2_list.append(r2_score(y_oob, y_pred))
        mse_list.append(mse(y_oob, y_pred))

    return np.array(r2_list), np.array(mse_list)


def learning_curve(X, y, train_sizes, k=5, seed=42, shuffle=True):
    """Compute training and validation errors for different train sizes."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)

    train_mse_means = []
    val_mse_means = []

    for frac in train_sizes:
        size = max(2, int(n * frac))
        subset_idx = indices[:size]
        X_sub = X[subset_idx]
        y_sub = y[subset_idx]

        # For each size, run K-fold CV on the subset
        train_mse_folds = []
        val_mse_folds = []
        for train_idx, val_idx in k_fold_indices(len(y_sub), k=k, seed=seed, shuffle=shuffle):
            X_train, y_train = X_sub[train_idx], y_sub[train_idx]
            X_val, y_val = X_sub[val_idx], y_sub[val_idx]
            w = fit_linear_regression_normal_eq(X_train, y_train)
            y_train_pred = predict(X_train, w)
            y_val_pred = predict(X_val, w)
            train_mse_folds.append(mse(y_train, y_train_pred))
            val_mse_folds.append(mse(y_val, y_val_pred))

        train_mse_means.append(np.mean(train_mse_folds))
        val_mse_means.append(np.mean(val_mse_folds))

    return np.array(train_mse_means), np.array(val_mse_means)


def main():
    # 1) Data loading
    df = load_dataset()

    # Drop rows with missing total_bedrooms if present
    if "total_bedrooms" in df.columns:
        df = df.dropna(subset=["total_bedrooms"])

    # If ocean_proximity exists, apply manual one-hot encoding
    if "ocean_proximity" in df.columns:
        df = one_hot_encode(df, "ocean_proximity")

    # Separate features and target
    if "MedHouseVal" in df.columns:
        target_col = "MedHouseVal"
    elif "median_house_value" in df.columns:
        target_col = "median_house_value"
    else:
        raise ValueError("Target column not found.")

    X = df.drop(columns=[target_col]).to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)

    # Standardize features
    X, X_mean, X_std = standardize_features(X)

    # 2) Hold-out evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42, shuffle=True)
    w = fit_linear_regression_normal_eq(X_train, y_train)
    y_test_pred = predict(X_test, w)
    holdout_r2 = r2_score(y_test, y_test_pred)
    holdout_mse = mse(y_test, y_test_pred)

    print("Hold-out evaluation:")
    print(f"  R2  = {holdout_r2:.4f}")
    print(f"  MSE = {holdout_mse:.4f}")

    # 3) K-fold cross validation
    for k in [5, 10]:
        r2_list, mse_list = cross_validate(X, y, k=k, seed=42, shuffle=True)
        print(f"\n{k}-fold CV results:")
        print(f"  R2  mean = {r2_list.mean():.4f}, std = {r2_list.std():.4f}")
        print(f"  MSE mean = {mse_list.mean():.4f}, std = {mse_list.std():.4f}")

    # 4) Bootstrap evaluation (OOB)
    boot_r2, boot_mse = bootstrap_evaluate(X, y, n_bootstrap=100, seed=42)
    print("\nBootstrap (OOB) results:")
    print(f"  R2  mean = {boot_r2.mean():.4f}, std = {boot_r2.std():.4f}")
    print(f"  MSE mean = {boot_mse.mean():.4f}, std = {boot_mse.std():.4f}")

    # 5) Learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_mse, val_mse = learning_curve(X, y, train_sizes, k=5, seed=42, shuffle=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes * 100, train_mse, marker="o", label="Train MSE")
    plt.plot(train_sizes * 100, val_mse, marker="s", label="Validation MSE")
    plt.xlabel("Training set size (%)")
    plt.ylabel("MSE")
    plt.title("Learning Curve (Linear Regression)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Basic bias/variance diagnosis
    if val_mse[-1] > train_mse[-1] * 1.5:
        conclusion = "High variance (overfitting)"
    elif train_mse[-1] > val_mse[-1] * 0.9:
        conclusion = "High bias (underfitting)"
    else:
        conclusion = "Balanced bias/variance"
    print(f"\nLearning-curve conclusion: {conclusion}")

    # 6) Optional: Ridge regression impact
    lambdas = np.logspace(-4, 3, 10)
    ridge_val_mse = []
    for lam in lambdas:
        _, mse_list = cross_validate(X, y, k=5, seed=42, shuffle=True, l2_lambda=lam)
        ridge_val_mse.append(mse_list.mean())

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas, ridge_val_mse, marker="o")
    plt.xlabel("Lambda (L2)")
    plt.ylabel("Validation MSE")
    plt.title("Ridge Regression: Lambda vs Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
