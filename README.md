# Régime national d'assurance contre les défaillances de barrages en terre — Tarrodan
### SOA Student Research Case Study Challenge 2025

> **Auteur :** Wendlassida KABORE  
> **Langue :** Français  
> **Outils :** Python 3 · R · Jupyter Notebook  

---

## Table des matières
1. [Contexte](#1-contexte)  
2. [Objectifs](#2-objectifs)  
3. [Structure du projet](#3-structure-du-projet)  
4. [Données](#4-données)  
5. [Méthodologie](#5-méthodologie)  
6. [Résultats clés](#6-résultats-clés)  
7. [Figures produites](#7-figures-produites)  
8. [Installation et exécution](#8-installation-et-exécution)  
9. [Sorties générées](#9-sorties-générées)  

---

## 1. Contexte

Ce projet s'inscrit dans le cadre du **2025 Student Research Case Study Challenge** organisé par la *Society of Actuaries* (SOA). Il porte sur un pays fictif, **Tarrodan**, divisé en trois régions (*Navaldia*, *Lyndrassia*, *Flumevale*), où de nombreuses communautés vivent à proximité de barrages en terre essentiels pour l'approvisionnement en eau, la prévention des inondations et le développement économique.

À la suite de plusieurs événements majeurs liés à des ruptures de barrages observés à l'échelle mondiale, l'étude propose de concevoir et d'évaluer financièrement un **régime national d'assurance** destiné à indemniser les pertes causées par ces défaillances.

---

## 2. Objectifs

- **Constituer le portefeuille national** des barrages en terre à partir des données brutes fournies par la SOA.
- **Annualiser la probabilité de défaillance** et calculer la prime pure par barrage.
- **Segmenter** le portefeuille en classes de risque homogènes (A → E).
- **Simuler la perte agrégée annuelle** par Monte-Carlo et en déduire le capital économique (VaR / TVaR).
- **Tarifier** commercialement (prime brute) en intégrant marge de risque, frais de gestion et fonds de prévention.
- **Projeter sur 20 ans** la soutenabilité du fonds sous différents scénarios économiques.
- **Quantifier l'impact** d'un programme national de prévention sur la sinistralité.

---

## 3. Structure du projet

```
case_study_soa_2025/
│
├── Données/                              # Données sources et données traitées
│   ├── srcsc-2025-dam-data-for-students.csv      # Données brutes fournies par la SOA
│   ├── srcsc-2025-economic-data-summary.xlsx     # Historique économique (inflation, taux)
│   ├── data_cleaned.csv                          # Données nettoyées (après étape 1)
│   ├── data_analyse.csv                          # Données enrichies (après étape 3)
│   ├── df_{Navaldia|Lyndrassia|Flumevale}.csv    # Données brutes par région
│   ├── df_{Navaldia|Lyndrassia|Flumevale}_imputed.csv  # Données imputées par région
│   ├── {région}_cv_summary.csv                   # Résumé de la validation croisée d'imputation
│   └── statistiques.xlsx                         # Statistiques descriptives exportées
│
├── Scripts/
│   ├── Notebook/                         # Notebooks Jupyter (exploration interactive)
│   │   ├── Nettoyage et exploration.ipynb
│   │   ├── Imputation.ipynb
│   │   ├── Analyse et ingenierie donnees.ipynb
│   │   ├── Analyse_actuarielle_tarification.ipynb
│   │   └── modelisation.ipynb
│   ├── Python/                           # Scripts Python autonomes
│   │   ├── imputation_donnees.py         # Imputation des données manquantes
│   │   ├── Exécution imputation.py       # Script de lancement de l'imputation
│   │   ├── modelisation.py               # Script principal — modélisation actuarielle
│   │   └── output/                       # Sorties générées par modelisation.py
│   │       ├── resultats.json
│   │       └── portefeuille_tarife.csv
│   └── R/
│       └── donnees_par_region.R          # Analyse par région sous R
│
├── figures/                              # Graphiques produits par modelisation.py
│   ├── 01_distribution_perte_agregee.png
│   ├── 02_historique_economique.png
│   ├── 03_projection_fonds.png
│   ├── 04_prevention.png
│   ├── 05_prime_par_classe.png
│   └── 06_decomposition_eal.png
│
├── tables/                               # Tableaux CSV produits par modelisation.py
│   ├── classes_risque.csv
│   ├── prime_par_classe.csv
│   ├── prime_par_region.csv
│   ├── sensibilite.csv
│   └── segment_{hazard|region|regule|assessment|usage}.csv
│
├── Rapport_final_SOA_2025.pdf            # Rapport actuariel final
├── Rapport_final_SOA_2025.docx           # Version Word du rapport
├── 2025-case-study-challenge.pdf         # Énoncé officiel SOA
└── README.md
```

---

## 4. Données

| Fichier | Description |
|---|---|
| `srcsc-2025-dam-data-for-students.csv` | 19 368 barrages en terre — région, niveau de danger (*Hazard*), évaluation structurelle (*Assessment*), statut de régulation, probabilité de défaillance sur 10 ans (*p10*) et exposition financière (*LGF*) |
| `srcsc-2025-economic-data-summary.xlsx` | Séries annuelles d'inflation et de taux d'intérêt de Tarrodan (1962–2024), onglet *Inflation-Interest* |

**Variables clés :**

| Variable | Description |
|---|---|
| `p10` | Probabilité de défaillance sur 10 ans — annualisée : `p_annual = 1 − (1−p10)^(1/10)` |
| `LGF` | *Loss Given Failure* — perte financière en cas de rupture (en millions de Qums, Qm) |
| `Hazard` | Niveau de danger aval : *Low / High / Significant / Undetermined* |
| `Assessment` | État structurel : *Satisfactory / Fair / Poor / Unsatisfactory / …* |
| `Regulated.Dam` | Indicateur de régulation du barrage |

---

## 5. Méthodologie

### Étape 1 — Nettoyage et exploration
Traitement des valeurs manquantes et aberrantes, analyse univariée et bivariée, comparaison inter-régionale.

### Étape 2 — Imputation
Les valeurs manquantes de `p10` et `LGF` sont imputées par **régression ridge** avec validation croisée 5-fold, stratifiée par région et par niveau de danger pour préserver l'hétérogénéité géographique. Les résumés de validation croisée sont exportés dans `Données/{région}_cv_summary.csv`.

### Étape 3 — Ingénierie des données
Création des variables dérivées : annualisation de `p_annual`, calcul de l'EAL (*Expected Annual Loss* = `p_annual × LGF`), segmentation tarifaire, statistiques par segment.

### Étape 4 — Modélisation actuarielle (`modelisation.py`)

| Bloc | Méthode |
|---|---|
| **Prime pure** | `EAL = p_annual × LGF_total` par barrage, puis agrégation nationale |
| **Classes de risque** | Segmentation en quintiles d'EAL → 5 classes A (risque faible) à E (risque élevé) |
| **Monte-Carlo** | 100 000 simulations de la perte agrégée annuelle (loi de Poisson composée) |
| **Capital économique** | TVaR 99,5 % − E[perte] |
| **Tarification** | Prime technique + marge de risque (coût du capital 6 %) + frais (10 %) + fonds de prévention (5 %) |
| **Hypothèses économiques** | Inflation tendancielle 2,48 % ; rendement 2,78 % (moyennes des 10–20 dernières années) |
| **Projection 20 ans** | 5 000 trajectoires stochastiques du fonds avec inflation et rendement |
| **Sensibilité** | 9 scénarios : primes ±10 %, sinistralité +20 %, capital initial /2, rendement 1,5 %/4,5 %, inflation 4 %, prévention −15 % PoF |
| **Prévention** | Réduction ciblée de la PoF : −20 % sur ouvrages *High/Significant*, −10 % sur ouvrages mal notés |

---

## 6. Résultats clés

### Portefeuille national

| Indicateur | Valeur |
|---|---|
| Nombre de barrages | **19 368** |
| Exposition totale (Σ LGF) | **7 594 040 Qm** |
| Perte attendue sur 10 ans | **731 410 Qm** |
| Prime pure annuelle totale (EAL) | **76 800 Qm** |

### Segmentation par classe de risque

| Classe | Nb barrages | PoF annuelle | LGF moyen (Qm) | Prime pure moy. (Qm) | Part EAL |
|:---:|---:|---:|---:|---:|---:|
| A | 3 881 | 0,6 % | 304 | 1,84 | 9,3 % |
| B | 3 870 | 0,8 % | 331 | 2,68 | 13,5 % |
| C | 3 871 | 1,0 % | 378 | 3,60 | 18,2 % |
| D | 3 881 | 1,1 % | 380 | 4,15 | 21,0 % |
| E | 3 865 | 1,4 % | 568 | 7,57 | **38,1 %** |

### Prime pure par région

| Région | Nb barrages | Prime pure (Qm) | Prime brute (Qm) |
|---|---:|---:|---:|
| Navaldia | 8 374 | 32 003 | 38 314 |
| Lyndrassia | 7 920 | 27 852 | 33 347 |
| Flumevale | 3 074 | 16 946 | 20 296 |

### Capital économique et tarification

| Indicateur | Valeur |
|---|---|
| Monte-Carlo — moyenne | 76 784 Qm |
| Monte-Carlo — écart-type | 7 480 Qm (CV = 9,7 %) |
| VaR 99,5 % | 96 964 Qm |
| TVaR 99,5 % | 99 512 Qm |
| **Capital économique** (TVaR − E) | **22 728 Qm** |
| **Prime brute totale** | **91 957 Qm** (chargement global +19,7 %) |

### Projection du fonds sur 20 ans

| Indicateur | Valeur |
|---|---|
| Probabilité de ruine | **0,42 %** |
| Fonds terminal médian | 236 189 Qm |
| Fonds terminal P5 / P95 | 142 935 / 325 140 Qm |

### Analyses de sensibilité

| Scénario | Prob. ruine | Fonds médian (Qm) |
|---|---:|---:|
| Base | 0,37 % | 233 886 |
| Prime −10 % | **82,83 %** | −34 387 |
| Prime +10 % | 0,00 % | 508 013 |
| Rendement 1,5 % | 0,43 % | 204 635 |
| Rendement 4,5 % | 0,23 % | 282 685 |
| Inflation 4 % | 0,47 % | 263 382 |
| Sinistralité +20 % | **100 %** | −268 571 |
| Prévention −15 % PoF | 0,00 % | 613 307 |
| Capital initial / 2 | 4,33 % | 215 224 |

> Les scénarios *Prime −10 %* et *Sinistralité +20 %* sont les plus critiques : ils conduisent à une ruine quasi-certaine, soulignant la fragilité du régime face à une sous-tarification ou à un choc de sinistralité majeur.

### Programme de prévention

Une politique ciblée (inspection renforcée et réhabilitation des ouvrages à fort enjeu et mal notés) réduit la PoF de −20 % sur les barrages *High/Significant* et de −10 % sur les barrages mal évalués :

| | Avant | Après |
|---|---:|---:|
| EAL national (Qm) | 76 800 | 63 476 |
| **Économie annuelle** | | **13 324 Qm (−17,3 %)** |

---

## 7. Figures produites

| Fichier | Description |
|---|---|
| `01_distribution_perte_agregee.png` | Distribution Monte-Carlo de la perte agrégée annuelle avec VaR et TVaR à 99,5 % |
| `02_historique_economique.png` | Séries historiques inflation et taux sans risque 10 ans (Tarrodan, 1962–2024) |
| `03_projection_fonds.png` | Éventail stochastique des trajectoires du fonds sur 20 ans (P5 / P25 / médiane / P75 / P95) |
| `04_prevention.png` | Effet du programme de prévention sur la prime pure par région |
| `05_prime_par_classe.png` | Prime pure et brute moyennes par classe de risque (A–E) |
| `06_decomposition_eal.png` | Décomposition de la prime pure par région et niveau de danger |

---

## 8. Installation et exécution

### Prérequis Python

```bash
pip install numpy pandas matplotlib openpyxl scipy
```

Python ≥ 3.9 recommandé.

### Pipeline complet (ordre à respecter)

```
1. Nettoyage       → Scripts/Notebook/Nettoyage et exploration.ipynb
2. Imputation      → python "Scripts/Python/imputation_donnees.py"
                      (ou Scripts/Notebook/Imputation.ipynb)
3. Ingénierie      → Scripts/Notebook/Analyse et ingenierie donnees.ipynb
4. Modélisation    → python "Scripts/Python/modelisation.py"
```

### Exécution standalone du script de modélisation

Le script peut être lancé depuis **n'importe quel répertoire** grâce à la résolution automatique des chemins via `__file__` :

```bash
# Depuis le dossier du projet
python Scripts/Python/modelisation.py

# Depuis n'importe où
python "C:/chemin/vers/case_study_soa_2025/Scripts/Python/modelisation.py"
```

Les données sont lues dans `Données/`, les figures sont sauvegardées dans `figures/`, les tableaux dans `tables/` et les sorties JSON/CSV dans `Scripts/Python/output/`.

---

## 9. Sorties générées

Après exécution de `modelisation.py` :

| Fichier | Description |
|---|---|
| `Scripts/Python/output/resultats.json` | Tous les indicateurs actuariels au format JSON (portefeuille, tarification, capital économique, projection, sensibilité, prévention) |
| `Scripts/Python/output/portefeuille_tarife.csv` | Tableau barrage par barrage avec classe, prime pure, prime brute et EAL avec/sans prévention |
| `tables/classes_risque.csv` | Statistiques par classe de risque |
| `tables/prime_par_classe.csv` | Prime moyenne par classe |
| `tables/prime_par_region.csv` | Prime agrégée par région |
| `tables/sensibilite.csv` | Résultats des 9 scénarios de sensibilité |
| `tables/segment_*.csv` | Segmentation par hazard, région, régulation, assessment, usage |
| `figures/0{1-6}_*.png` | 6 graphiques (voir §7) |
