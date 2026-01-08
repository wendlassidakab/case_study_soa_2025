from imputation_donnees import imputation

path_data = "../../Données/"

imputation("Flumevale", path_data=path_data, filename="df_Flumevale.csv")
imputation("Lyndrassia", path_data=path_data, filename="df_Lyndrassia.csv")
imputation("Navaldia", path_data=path_data, filename="df_Navaldia.csv")

