file <- "data_analyse.csv" # Adapter selon le nom du fichier

df <- read.csv(paste0("../../Données/", file))

regions <- unique(df$Region)

for (reg in regions){
    df_reg <- df[df$Region == reg, ]

    write.csv(df_reg,
              file = paste0("../../Données/df_", reg, ".csv"),
              row.names = FALSE)
}
