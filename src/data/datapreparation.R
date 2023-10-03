# Retrieve gene expression and meta data from GDC Data Portal using TCGAbiolinks

library(TCGAbiolinks)
library(SummarizedExperiment)

query <- GDCquery(project = c("TCGA-LGG", "TCGA-GBM"),
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification", 
                  workflow.type = "STAR - Counts",
                  sample.type = "Primary Tumor")

GDCdownload(query, method = "api", files.per.chunk = 10)

exp_data <- GDCprepare(query)
exp_matrix <- as.data.frame(assay(exp_data, "tpm_unstrand")) # normalized gene expression matrix

meta_data <- as.data.frame(colData(exp_data)) # meta data matrix

# Data preparation
  ## Meta data
meta_matrix <- meta_data[c('patient', 'paper_Supervised.DNA.Methylation.Cluster')] 
colnames(meta_matrix)[2] <- "subtypes"

dim(meta_matrix) # 673  2

meta_matrix <- meta_matrix[!duplicated(meta_matrix$patient),] # remove duplicates (2)

table(is.na(meta_matrix$subtypes)) # remove NAs (39)
meta_matrix <- meta_matrix[!is.na(meta_matrix$subtypes),] # 632 samples

meta_matrix[meta_matrix$subtypes %in% 'PA-like', ]$subtypes <- 'LGm6-GBM' # WHO recomendation

rownames(meta_matrix) <- meta_matrix$patient
meta_matrix$patient <- NULL

dim(meta_matrix) # 632  1

  ## Gene Expression data
dim(exp_matrix) # 60660  673

colnames(exp_matrix) <- substr(colnames(exp_matrix), 1, 12) # match patient ID with meta_matrix
colnames(exp_matrix) <- gsub("\\.", "-", colnames(exp_matrix))
exp_matrix <- exp_matrix[colnames(exp_matrix) %in% rownames(meta_matrix)]

all(colnames(exp_matrix) %in% rownames(meta_matrix)) #TRUE
all(colnames(exp_matrix) == rownames(meta_matrix)) #TRUE

    ### Filter only protein coding genes
wget("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz" ) #always check the latest version at https://www.gencodegenes.org/human/
gencode <- readGFF("./gencode.v44.annotation.gtf.gz") 
gencode <- subset(gencode, type %in% "gene")
gencode.coding <- gencode[gencode$gene_type %in% "protein_coding",] # only protein coding genes IDs

for(i in 1:nrow(gencode.coding)) {
  gencode.coding$gene_id[i] <- strsplit(gencode.coding$gene_id[i], split = "\\.")[[1]][1]
}

exp_matrix$gene_id <- rownames(exp_matrix) # avoid duplicates in rownames
for(i in 1:nrow(exp_matrix)) {
  exp_matrix$gene_id[i] <- strsplit(exp_matrix$gene_id[i], split = "\\.")[[1]][1]
}

exp_matrix = exp_matrix[!duplicated(exp_matrix$gene_id),]
rownames(exp_matrix) <- exp_matrix$gene_id
exp_matrix$gene_id <- NULL

exp_matrix <- exp_matrix[rownames(exp_matrix) %in% gencode.coding$gene_id,]

dim(exp_matrix) # 19920  632

# Save objects as csv
save(exp_matrix, file = "./data.csv")
save(meta_matrix, file = "./meta.csv")

# Join both expression data and subtype information in a single dataframe
dataset <- as.data.frame(t(exp_matrix)) # transpose

all(rownames(dataset) %in% rownames(meta_matrix)) #TRUE
all(rownames(dataset) == rownames(meta_matrix)) #TRUE

dataset$subtype <- meta_matrix$subtypes # add new column w/ patient's subtype

save(dataset, file = "./dataset.csv") # save as csv
