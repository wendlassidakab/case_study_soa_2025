import os
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor


def imputation(region: str, path_data: str, filename: str | None = None) -> pd.DataFrame:
    """
    Impute les variables physiques (Height, Length, Volume, Surface, Drainage, Distance)
    puis entraîne un modèle supervisé pour prédire 3 cibles de perte (Qm) manquantes.

    Étapes :
      1) Lecture des données (CSV) pour une région donnée
      2) Nettoyage / mise à NA des valeurs aberrantes (règles fixes)
      3) Création des indicateurs *_missing pour les variables imputées
      4) Imputation des variables numériques via IterativeImputer (estimateur RF)
      5) Pour chaque target : sélection du meilleur modèle (MAE en CV=5)
         puis prédiction des valeurs manquantes
      6) Export du fichier final en CSV

    Paramètres
    ----------
    region : str
        Nom/identifiant de la région (utilisé pour construire le chemin et nommer le fichier de sortie).
    path_data : str
        Répertoire contenant les fichiers CSV.
    filename : str | None
        Nom du fichier. Si None, on suppose que le fichier s’appelle f"{region}.csv".

    Retour
    ------
    pd.DataFrame
        DataFrame final avec imputation + prédiction des targets.
    """

    # -----------------------------
    # 1) Lecture
    # -----------------------------
    if filename is None:
        filename = f"{region}.csv"

    filepath = os.path.join(path_data, filename)
    df = pd.read_csv(filepath)

    # -----------------------------
    # 2) Nettoyage (outliers -> NaN)
    # -----------------------------
    # Petite fonction utilitaire : applique un masque si la colonne existe
    def set_nan_where(col: str, condition):
        if col in df.columns:
            df.loc[condition, col] = np.nan

    # Height
    set_nan_where("Height..m.", (df.get("Height..m.", pd.Series(index=df.index)) == 0))
    set_nan_where("Height..m.", (df.get("Height..m.", pd.Series(index=df.index)) > 320))

    # Length (corrige le nom : Length..km. dans tes vars_to_impute)
    # Certains fichiers ont "Length..km." / d'autres "Length..km" : on harmonise.
    if "Length..km." not in df.columns and "Length..km" in df.columns:
        df.rename(columns={"Length..km": "Length..km."}, inplace=True)

    set_nan_where("Length..km.", (df.get("Length..km.", pd.Series(index=df.index)) == 0))
    set_nan_where("Length..km.", (df.get("Length..km.", pd.Series(index=df.index)) > 2.5))

    # Volume
    set_nan_where("Volume..m3.", (df.get("Volume..m3.", pd.Series(index=df.index)) == 0))
    set_nan_where("Volume..m3.", (df.get("Volume..m3.", pd.Series(index=df.index)) > 1.8e11))

    # Surface
    set_nan_where("Surface..km2.", (df.get("Surface..km2.", pd.Series(index=df.index)) == 0))
    set_nan_where("Surface..km2.", (df.get("Surface..km2.", pd.Series(index=df.index)) > 8500))

    # Drainage
    set_nan_where("Drainage..km2.", (df.get("Drainage..km2.", pd.Series(index=df.index)) == 0))
    set_nan_where("Drainage..km2.", (df.get("Drainage..km2.", pd.Series(index=df.index)) > 1.8e6))

    # -----------------------------
    # 3) Variables à imputer + indicateurs de manquants
    # -----------------------------
    vars_to_impute = [
        "Height..m.",
        "Length..km.",
        "Volume..m3.",
        "Drainage..km2.",
        "Surface..km2.",
        "Distance.to.Nearest.City..km."
    ]

    # On ne crée l'indicateur que si la colonne existe
    for v in vars_to_impute:
        if v in df.columns:
            df[v + "_missing"] = df[v].isna().astype(int)

    # Variables à exclure de l'imputation (ne pas les utiliser comme features d'imputation)
    vars_excluded = [
        "Probability of Failure",
        "Loss given failure – prop (Qm)",
        "Loss given failure – liab (Qm)",
        "Loss given failure – BI (Qm)",
        "Height_clean",
        "Length_clean",
        "Length_missing",
        "Volume_clean",
        "Volume_missing",
        "Volume_log",
        "n_modifications",
        "last_modification_year"
    ]

    # -----------------------------
    # 4) Imputation numérique (IterativeImputer)
    # -----------------------------
    # On prend toutes les colonnes sauf exclues, puis uniquement numériques.
    imputation_features = [c for c in df.columns if c not in vars_excluded]
    df_imp_base = df[imputation_features].select_dtypes(include=["number"])

    # On retire les colonnes entièrement NA (IterativeImputer ne les gère pas bien)
    all_na_cols = df_imp_base.columns[df_imp_base.isna().all()]
    df_imp_base2 = df_imp_base.drop(columns=all_na_cols)

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        max_iter=10,
        random_state=42
    )

    df_imp_array = imputer.fit_transform(df_imp_base2)

    df_imp = pd.DataFrame(
        df_imp_array,
        columns=df_imp_base2.columns,
        index=df.index
    )

    # Réinjecte les valeurs imputées dans une copie
    df_imputed = df.copy()
    for v in vars_to_impute:
        if v in df_imputed.columns and v in df_imp.columns:
            df_imputed[v] = df_imp[v]

    # enlever du DF final les colonnes 100% NA de la matrice d'imputation
    df_imputed = df_imputed.drop(columns=list(all_na_cols), errors="ignore")

    # -----------------------------
    # 5) Modélisation supervisée : targets
    # -----------------------------
    features = [
        "Regulated.Dam", "Primary.Purpose", "Primary.Type",
        "Height..m.", "Length..km.", "Year.Completed",
        "Surface..km2.", "Drainage..km2.", "Spillway",
        "Distance.to.Nearest.City..km.", "Hazard", "Assessment",
        "n_modifications"
    ]

    # Filtre features existantes (robustesse si colonnes absentes)
    features = [f for f in features if f in df_imputed.columns]

    targets = [
        "Loss.given.failure...prop..Qm.",
        "Loss.given.failure...liab..Qm.",
        "Loss.given.failure...BI..Qm."
    ]
    targets = [t for t in targets if t in df_imputed.columns]

    if len(features) == 0:
        raise ValueError("Aucune feature valide trouvée dans le DataFrame pour entraîner les modèles.")

    # Prétraitement : one-hot sur colonnes catégorielles, num en passthrough
    X_all = df_imputed[features]
    cat_cols = X_all.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    # Pool de modèles candidats
    models = {
        "Linear": LinearRegression(),
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        "XGB": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )
    }

    # Pour garder une trace des performances
    cv_summary = {}

    for target in targets:
        # Train/predict split pour ce target
        df_train = df_imputed[df_imputed[target].notna()].copy()
        df_pred = df_imputed[df_imputed[target].isna()].copy()

        # S'il n'y a rien à prédire ou pas assez de train, on passe
        if df_pred.shape[0] == 0:
            cv_summary[target] = {"status": "nothing_to_predict"}
            continue
        if df_train.shape[0] < 20:
            cv_summary[target] = {"status": "insufficient_training_data", "n_train": int(df_train.shape[0])}
            continue

        X_train = df_train[features]
        y_train = df_train[target]
        X_pred = df_pred[features]

        # Sélection du meilleur modèle via MAE en CV
        results = {}
        for name, model in models.items():
            pipe = Pipeline([("prep", preprocess), ("model", model)])

            mae = -cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=5,
                scoring="neg_mean_absolute_error"
            ).mean()

            results[name] = float(mae)

        best_model_name = min(results, key=results.get)
        best_model = models[best_model_name]

        final_pipe = Pipeline([("prep", preprocess), ("model", best_model)])
        final_pipe.fit(X_train, y_train)

        # Prédiction
        df_imputed.loc[df_imputed[target].isna(), target] = final_pipe.predict(X_pred)

        cv_summary[target] = {
            "status": "ok",
            "best_model": best_model_name,
            "mae_cv5": results[best_model_name],
            "all_models_mae_cv5": results,
            "n_train": int(df_train.shape[0]),
            "n_pred": int(df_pred.shape[0])
        }

    # -----------------------------
    # 6) Export
    # -----------------------------
    out_name = f"{os.path.splitext(filename)[0]}_imputed.csv"
    out_path = os.path.join(path_data, out_name)
    df_imputed.to_csv(out_path, index=False)

    
    pd.DataFrame(cv_summary).T.to_csv(os.path.join(path_data, f"{region}_cv_summary.csv"))

    return df_imputed
