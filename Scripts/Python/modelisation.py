# -*- coding: utf-8 -*-
"""
==============================================================================
 Conception et évaluation financière d'un régime national d'assurance
 contre les défaillances de barrages en terre — Tarrodan (SRCSC 2025, SOA)

 Partie actuarielle : tarification, capital économique et soutenabilité.
 Auteur : Wendlassida KABORE
==============================================================================
Ce script prolonge le travail de nettoyage / exploration / imputation déjà
réalisé (notebooks « Nettoyage et exploration », « Imputation »,
« Analyse et ingénierie des données ») et construit la solution actuarielle :

  1. Constitution du portefeuille national (3 régions) à partir des données
     imputées.
  2. Annualisation de la probabilité de défaillance et prime pure par barrage.
  3. Segmentation tarifaire en classes de risque.
  4. Modèle fréquence–sévérité et simulation de Monte-Carlo de la perte agrégée
     annuelle (VaR / TVaR, capital économique).
  5. Tarification commerciale (marge de risque par coût du capital + frais).
  6. Projection stochastique pluriannuelle du fonds (inflation, rendement).
  7. Analyses de sensibilité et impact d'un programme de prévention.

Toutes les pertes sont exprimées en Qm = millions de Qums (Q), la monnaie de
Tarrodan.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl

np.random.seed(2025)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "..", "..", "Données")
FIG  = os.path.join(_HERE, "..", "..", "figures")
TAB  = os.path.join(_HERE, "..", "..", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
COL = {"Navaldia": "#1f77b4", "Lyndrassia": "#2ca02c", "Flumevale": "#d62728"}
R = {}   # dictionnaire de résultats exporté pour le rapport


def qfmt(x):
    return float(np.round(x, 4))


# ===========================================================================
# 1. PORTEFEUILLE NATIONAL
# ===========================================================================
regs = ["Navaldia", "Lyndrassia", "Flumevale"]
df = pd.concat(
    [pd.read_csv(f"{DATA}/df_{r}_imputed.csv").assign(Region=r) for r in regs],
    ignore_index=True,
)

df["LGF_prop"] = df["Loss.given.failure...prop..Qm."]
df["LGF_liab"] = df["Loss.given.failure...liab..Qm."]
df["LGF_BI"] = df["Loss.given.failure...BI..Qm."]
df["LGF_total"] = df[["LGF_prop", "LGF_liab", "LGF_BI"]].sum(axis=1)
df["p10"] = df["Probability.of.Failure"].clip(1e-6, 0.999)

# Annualisation : la PoF fournie est une probabilité sur 10 ans . On suppose un
# risque constant dans le temps -> taux annuel équivalent.
df["p_annual"] = 1 - (1 - df["p10"]) ** (1 / 10)

# Prime pure annuelle (espérance de perte annuelle) par barrage
df["EAL"] = df["p_annual"] * df["LGF_total"]
df["EL10"] = df["p10"] * df["LGF_total"]

N = len(df)
R["n_dams"] = int(N)
R["n_by_region"] = df["Region"].value_counts().to_dict()
R["mean_p10"] = qfmt(df["p10"].mean())
R["mean_p_annual"] = qfmt(df["p_annual"].mean())
R["mean_LGF_total"] = qfmt(df["LGF_total"].mean())
R["exposure_total_Qm"] = qfmt(df["LGF_total"].sum())
R["EAL_total_Qm"] = qfmt(df["EAL"].sum())
R["EL10_total_Qm"] = qfmt(df["EL10"].sum())
R["pure_premium_mean_per_dam_Qm"] = qfmt(df["EAL"].mean())

print(f"Portefeuille national : {N} barrages en terre")
print(f"Exposition totale (somme LGF) : {df['LGF_total'].sum():,.0f} Qm")
print(f"Perte attendue 10 ans : {df['EL10'].sum():,.0f} Qm")
print(f"Perte annuelle attendue (prime pure totale) : {df['EAL'].sum():,.0f} Qm")

# ----- Tableaux de synthèse par segments -----
def seg(col):
    g = df.groupby(col).agg(
        n=("ID", "size"),
        pof10=("p10", "mean"),
        pof_ann=("p_annual", "mean"),
        lgf_mean=("LGF_total", "mean"),
        exposure=("LGF_total", "sum"),
        EAL=("EAL", "sum"),
    )
    g["pure_prem_moy"] = g["EAL"] / g["n"]
    g["part_EAL_%"] = 100 * g["EAL"] / g["EAL"].sum()
    return g.round(3)


for c, name in [("Region", "region"), ("Hazard", "hazard"),
                ("Regulated.Dam", "regule"), ("Assessment", "assessment"),
                ("Primary.Purpose", "usage")]:
    t = seg(c)
    t.to_csv(f"{TAB}/segment_{name}.csv")
R["seg_region"] = seg("Region").reset_index().to_dict(orient="records")
R["seg_hazard"] = seg("Hazard").reset_index().to_dict(orient="records")
R["seg_regule"] = seg("Regulated.Dam").reset_index().to_dict(orient="records")

# ===========================================================================
# 2. SEGMENTATION TARIFAIRE EN CLASSES DE RISQUE
# ===========================================================================
# Score de risque = combinaison du niveau de danger (Hazard), de l'état
# (Assessment) et de la probabilité annuelle. On construit 5 classes (A..E)
# par quantiles du score de risque.
haz_w = {"Low": 1.0, "Undetermined": 1.15, "Significant": 1.35, "High": 1.25}
ass_w = {"Satisfactory": 0.95, "Fair": 1.05, "Poor": 1.20,
         "Unsatisfactory": 1.30, "Not Available": 1.10, "Not Rated": 1.10}
df["haz_w"] = df["Hazard"].map(haz_w).fillna(1.10)
df["ass_w"] = df["Assessment"].map(ass_w).fillna(1.10)
df["risk_score"] = df["p_annual"] * df["haz_w"] * df["ass_w"]

df["classe"] = pd.qcut(df["risk_score"], 5, labels=["A", "B", "C", "D", "E"])
cls = df.groupby("classe", observed=True).agg(
    n=("ID", "size"),
    pof_ann=("p_annual", "mean"),
    lgf_mean=("LGF_total", "mean"),
    pure_prem=("EAL", "mean"),
    EAL=("EAL", "sum"),
)
cls["part_n_%"] = 100 * cls["n"] / cls["n"].sum()
cls["part_EAL_%"] = 100 * cls["EAL"] / cls["EAL"].sum()
cls = cls.round(3)
cls.to_csv(f"{TAB}/classes_risque.csv")
R["classes"] = cls.reset_index().astype({"classe": str}).to_dict(orient="records")
print("\nClasses de risque:\n", cls)

# ===========================================================================
# 3. SIMULATION MONTE-CARLO DE LA PERTE AGRÉGÉE ANNUELLE
# ===========================================================================
# Modèle fréquence-sévérité indépendant par barrage :
#   - fréquence : Bernoulli(p_annual_i)   (au plus une défaillance / an / barrage)
#   - sévérité  : LGF_total_i (déterministe au niveau du barrage)
# On simule la perte agrégée du portefeuille national.
p = df["p_annual"].to_numpy(dtype=np.float64)
sev = df["LGF_total"].to_numpy(dtype=np.float64)

NSIM = 100000
BATCH = 2000
agg = np.empty(NSIM, dtype=np.float64)
done = 0
while done < NSIM:
    b = min(BATCH, NSIM - done)
    u = np.random.random((b, N))
    fail = u < p          # broadcast p sur chaque ligne
    agg[done:done + b] = fail @ sev
    done += b

mean_agg = agg.mean()
sd_agg = agg.std()


def var_tvar(a, q):
    v = np.quantile(a, q)
    t = a[a >= v].mean()
    return v, t


var995, tvar995 = var_tvar(agg, 0.995)
var990, tvar990 = var_tvar(agg, 0.990)
var950, _ = var_tvar(agg, 0.95)

R["mc"] = {
    "nsim": NSIM,
    "mean": qfmt(mean_agg),
    "sd": qfmt(sd_agg),
    "cv": qfmt(sd_agg / mean_agg),
    "var95": qfmt(var950),
    "var99": qfmt(var990),
    "tvar99": qfmt(tvar990),
    "var995": qfmt(var995),
    "tvar995": qfmt(tvar995),
    "min": qfmt(agg.min()),
    "max": qfmt(agg.max()),
}
# Capital économique = TVaR 99.5% - prime pure (espérance)
capital = tvar995 - mean_agg
R["capital_economique_Qm"] = qfmt(capital)
print(f"\nMonte-Carlo (n={NSIM}) perte agrégée annuelle:")
print(f"  Moyenne {mean_agg:,.0f} | écart-type {sd_agg:,.0f} | CV {sd_agg/mean_agg:.3f}")
print(f"  VaR99.5 {var995:,.0f} | TVaR99.5 {tvar995:,.0f}")
print(f"  Capital économique (TVaR99.5 - E) {capital:,.0f} Qm")

# Figure : distribution de la perte agrégée
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(agg / 1000, bins=80, color="#4C72B0", alpha=0.85)
ax.axvline(mean_agg / 1000, color="black", ls="--", lw=1.5, label=f"Moyenne {mean_agg/1000:,.0f}")
ax.axvline(var995 / 1000, color="#d62728", ls="--", lw=1.5, label=f"VaR 99,5% {var995/1000:,.0f}")
ax.axvline(tvar995 / 1000, color="#8B0000", ls="-", lw=1.5, label=f"TVaR 99,5% {tvar995/1000:,.0f}")
ax.set_xlabel("Perte agrégée annuelle (milliards de Qums)")
ax.set_ylabel("Fréquence")
ax.set_title("Distribution simulée de la perte agrégée annuelle du portefeuille national")
ax.legend()
fig.savefig(f"{FIG}/01_distribution_perte_agregee.png")
plt.close(fig)

# ===========================================================================
# 4. TARIFICATION COMMERCIALE
# ===========================================================================
COC = 0.06            # coût du capital (marge de risque)
EXPENSE = 0.10        # frais (gestion, acquisition, prévention) en % de la prime brute
PREV_FUND = 0.05      # contribution au fonds de prévention en % de la prime brute

risk_margin_total = COC * capital                        # marge de risque agrégée
# Allocation de la marge de risque proportionnellement à l'écart-type
# contributif (approx : sev_i * sqrt(p(1-p))_i)
contrib = sev * np.sqrt(p * (1 - p))
alloc = contrib / contrib.sum()
df["risk_margin"] = risk_margin_total * alloc
df["prime_tech"] = df["EAL"] + df["risk_margin"]         # prime technique (pure+marge)
df["prime_brute"] = df["prime_tech"] / (1 - EXPENSE - PREV_FUND)

prime_brute_tot = df["prime_brute"].sum()
R["tarification"] = {
    "coc": COC, "expense": EXPENSE, "prev_fund": PREV_FUND,
    "risk_margin_total_Qm": qfmt(risk_margin_total),
    "prime_tech_total_Qm": qfmt(df["prime_tech"].sum()),
    "prime_brute_total_Qm": qfmt(prime_brute_tot),
    "chargement_global_%": qfmt(100 * (prime_brute_tot / df["EAL"].sum() - 1)),
    "prime_brute_moy_par_barrage_Qm": qfmt(df["prime_brute"].mean()),
}
print(f"\nPrime brute totale : {prime_brute_tot:,.0f} Qm "
      f"(chargement global {100*(prime_brute_tot/df['EAL'].sum()-1):.1f}%)")

# Prime moyenne par classe
clp = df.groupby("classe", observed=True).agg(
    pure=("EAL", "mean"), tech=("prime_tech", "mean"), brute=("prime_brute", "mean"),
).round(3)
clp.to_csv(f"{TAB}/prime_par_classe.csv")
R["prime_par_classe"] = clp.reset_index().astype({"classe": str}).to_dict(orient="records")

# Prime par région
rgp = df.groupby("Region").agg(
    n=("ID", "size"), pure=("EAL", "sum"), brute=("prime_brute", "sum"),
).round(1)
rgp.to_csv(f"{TAB}/prime_par_region.csv")
R["prime_par_region"] = rgp.reset_index().to_dict(orient="records")

# ===========================================================================
# 5. HYPOTHÈSES ÉCONOMIQUES (données fournies)
# ===========================================================================
wb = openpyxl.load_workbook(f"{DATA}/srcsc-2025-economic-data-summary.xlsx", data_only=True)
ws = wb["Inflation-Interest"]
erows = [r for r in ws.iter_rows(values_only=True) if isinstance(r[0], int)]
infl = np.array([r[1] for r in erows])
spot10 = np.array([r[4] for r in erows])
INFL = float(np.mean(infl[-20:]))           # inflation tendancielle ~2.5%
IYIELD = float(np.mean(spot10[-10:]))       # rendement ~ taux 10 ans récents ~3%
R["eco"] = {"inflation": qfmt(INFL), "rendement_invest": qfmt(IYIELD),
            "infl_hist_moy": qfmt(infl.mean()), "spot10_hist_moy": qfmt(spot10.mean())}
print(f"\nInflation retenue {INFL:.3%} | rendement d'investissement {IYIELD:.3%}")

# Figure : historique économique
fig, ax = plt.subplots(figsize=(8, 4))
yrs = [r[0] for r in erows]
ax.plot(yrs, infl, label="Inflation", color="#d62728")
ax.plot(yrs, spot10, label="Taux sans risque 10 ans", color="#1f77b4")
ax.axhline(INFL, color="#d62728", ls=":", lw=1)
ax.axhline(IYIELD, color="#1f77b4", ls=":", lw=1)
ax.set_title("Historique de l'inflation et des taux à Tarrodan (1962-2024)")
ax.set_xlabel("Année"); ax.set_ylabel("Taux"); ax.legend()
fig.savefig(f"{FIG}/02_historique_economique.png")
plt.close(fig)

# ===========================================================================
# 6. PROJECTION STOCHASTIQUE DU FONDS (20 ans)
# ===========================================================================
HORIZON = 20
NPATH = 5000
prime0 = prime_brute_tot
# Sinistres annuels : on rééchantillonne dans la distribution MC (agg) et on
# indexe sur l'inflation cumulée.
claims_base = np.random.choice(agg, size=(NPATH, HORIZON), replace=True)

infl_factor = (1 + INFL) ** np.arange(HORIZON)            # indexation des coûts
prime_factor = (1 + INFL) ** np.arange(HORIZON)           # primes indexées aussi

fund = np.zeros((NPATH, HORIZON + 1))
fund[:, 0] = capital                                       # dotation initiale
ruin = np.zeros(NPATH, dtype=bool)
exp_ratio = EXPENSE
for t in range(HORIZON):
    opening = fund[:, t]
    invest = opening * IYIELD
    prem = prime0 * prime_factor[t] * (1 - exp_ratio)      # prime nette de frais
    clm = claims_base[:, t] * infl_factor[t]
    closing = opening + invest + prem - clm
    fund[:, t + 1] = closing
    ruin |= closing < 0

prob_ruin = ruin.mean()
terminal = fund[:, -1]
R["projection"] = {
    "horizon": HORIZON, "npath": NPATH,
    "capital_initial_Qm": qfmt(capital),
    "prime_brute_an0_Qm": qfmt(prime0),
    "prob_ruine_%": qfmt(100 * prob_ruin),
    "fonds_terminal_median_Qm": qfmt(np.median(terminal)),
    "fonds_terminal_p05_Qm": qfmt(np.quantile(terminal, 0.05)),
    "fonds_terminal_p95_Qm": qfmt(np.quantile(terminal, 0.95)),
}
print(f"\nProjection {HORIZON} ans : prob. de ruine {100*prob_ruin:.2f}% | "
      f"fonds terminal médian {np.median(terminal):,.0f} Qm")

# Figure : éventail des trajectoires du fonds
fig, ax = plt.subplots(figsize=(8, 4.5))
qs = np.quantile(fund, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0) / 1000
xs = np.arange(HORIZON + 1)
ax.fill_between(xs, qs[0], qs[4], color="#4C72B0", alpha=0.15, label="5-95 %")
ax.fill_between(xs, qs[1], qs[3], color="#4C72B0", alpha=0.30, label="25-75 %")
ax.plot(xs, qs[2], color="#1f3a93", lw=2, label="Médiane")
ax.axhline(0, color="red", ls="--", lw=1)
ax.set_title(f"Projection stochastique du fonds d'assurance sur {HORIZON} ans")
ax.set_xlabel("Année"); ax.set_ylabel("Solde du fonds (milliards de Qums)")
ax.legend()
fig.savefig(f"{FIG}/03_projection_fonds.png")
plt.close(fig)

# ===========================================================================
# 7. ANALYSES DE SENSIBILITÉ
# ===========================================================================
def project_prob_ruin(prime_mult=1.0, ret=IYIELD, infl=INFL, cap=capital,
                      pof_mult=1.0, npath=3000):
    cl = np.random.choice(agg, size=(npath, HORIZON), replace=True) * pof_mult
    iff = (1 + infl) ** np.arange(HORIZON)
    pff = (1 + infl) ** np.arange(HORIZON)
    f = np.full(npath, cap, dtype=float)
    r = np.zeros(npath, dtype=bool)
    for t in range(HORIZON):
        f = f + f * ret + prime0 * prime_mult * pff[t] * (1 - exp_ratio) - cl[:, t] * iff[t]
        r |= f < 0
    return 100 * r.mean(), np.median(f)

sens = []
for label, kw in [
    ("Base", {}),
    ("Prime -10%", {"prime_mult": 0.9}),
    ("Prime +10%", {"prime_mult": 1.1}),
    ("Rendement 1,5%", {"ret": 0.015}),
    ("Rendement 4,5%", {"ret": 0.045}),
    ("Inflation 4%", {"infl": 0.04}),
    ("Sinistralité +20%", {"pof_mult": 1.2}),
    ("Prévention -15% PoF", {"pof_mult": 0.85}),
    ("Capital initial /2", {"cap": capital / 2}),
]:
    pr, med = project_prob_ruin(**kw)
    sens.append({"scenario": label, "prob_ruine_%": round(pr, 2),
                 "fonds_terminal_median_Qm": round(med, 1)})
R["sensibilite"] = sens
pd.DataFrame(sens).to_csv(f"{TAB}/sensibilite.csv", index=False)
print("\nSensibilité:")
for s in sens:
    print(f"  {s['scenario']:<22} ruine {s['prob_ruine_%']:>6}%  | fonds médian {s['fonds_terminal_median_Qm']:>12,.0f}")

# ===========================================================================
# 8. IMPACT D'UN PROGRAMME DE PRÉVENTION
# ===========================================================================
# Hypothèse fondée sur les données : les barrages régulés et bien évalués
# présentent une PoF plus faible. Un programme national (inspections, plans
# d'urgence, réhabilitation des ouvrages anciens / mal notés) est modélisé en
# réduisant la PoF des ouvrages à fort enjeu et mal notés.
df["p_annual_prev"] = df["p_annual"].copy()
mask_high = df["Hazard"].isin(["High", "Significant"])
mask_bad = df["Assessment"].isin(["Poor", "Unsatisfactory", "Not Rated", "Not Available"])
df.loc[mask_high, "p_annual_prev"] *= 0.80     # -20% sur ouvrages à fort enjeu
df.loc[mask_bad, "p_annual_prev"] *= 0.90      # -10% via remise à niveau / inspection
df["EAL_prev"] = df["p_annual_prev"] * df["LGF_total"]
reduction = 1 - df["EAL_prev"].sum() / df["EAL"].sum()
R["prevention"] = {
    "EAL_avant_Qm": qfmt(df["EAL"].sum()),
    "EAL_apres_Qm": qfmt(df["EAL_prev"].sum()),
    "reduction_%": qfmt(100 * reduction),
    "economie_annuelle_Qm": qfmt(df["EAL"].sum() - df["EAL_prev"].sum()),
}
print(f"\nPrévention : EAL {df['EAL'].sum():,.0f} -> {df['EAL_prev'].sum():,.0f} Qm "
      f"({100*reduction:.1f}% de réduction)")

# Figure : EAL par région avant/après prévention
fig, ax = plt.subplots(figsize=(7.5, 4))
gr = df.groupby("Region").agg(av=("EAL", "sum"), ap=("EAL_prev", "sum")) / 1000
x = np.arange(len(gr)); w = 0.38
ax.bar(x - w/2, gr["av"], w, label="Sans prévention", color="#d62728", alpha=0.85)
ax.bar(x + w/2, gr["ap"], w, label="Avec prévention", color="#2ca02c", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(gr.index)
ax.set_ylabel("Prime pure annuelle (milliards de Qums)")
ax.set_title("Effet du programme de prévention sur la prime pure par région")
ax.legend()
fig.savefig(f"{FIG}/04_prevention.png")
plt.close(fig)

# Figure : prime pure par classe de risque
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(clp.index.astype(str), clp["brute"], color="#4C72B0", alpha=0.6, label="Prime brute")
ax.bar(clp.index.astype(str), clp["pure"], color="#1f3a93", label="Prime pure")
ax.set_xlabel("Classe de risque"); ax.set_ylabel("Prime moyenne par barrage (Qm)")
ax.set_title("Prime pure et prime brute moyennes par classe de risque")
ax.legend()
fig.savefig(f"{FIG}/05_prime_par_classe.png")
plt.close(fig)

# Figure : contribution à l'EAL par région et par hazard (barres empilées)
piv = (df.pivot_table(index="Region", columns="Hazard", values="EAL",
                      aggfunc="sum") / 1000).fillna(0)
fig, ax = plt.subplots(figsize=(7.5, 4))
bottom = np.zeros(len(piv))
for h in ["Low", "High", "Significant", "Undetermined"]:
    if h in piv.columns:
        ax.bar(piv.index, piv[h], bottom=bottom, label=h)
        bottom += piv[h].to_numpy()
ax.set_ylabel("Prime pure annuelle (milliards de Qums)")
ax.set_title("Décomposition de la prime pure par région et niveau de danger")
ax.legend(title="Hazard")
fig.savefig(f"{FIG}/06_decomposition_eal.png")
plt.close(fig)

# ===========================================================================
# EXPORT DES RÉSULTATS
# ===========================================================================
import os
out_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "resultats.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)

df_out = df[["ID", "Region", "Hazard", "Assessment", "Regulated.Dam",
             "p10", "p_annual", "LGF_total", "EAL", "classe",
             "risk_margin", "prime_tech", "prime_brute", "EAL_prev"]]
df_out.to_csv(os.path.join(out_dir, "portefeuille_tarife.csv"), index=False)
print(f"\n=== Termine. Resultats -> {out_dir} ===")
