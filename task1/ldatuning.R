library("tm")
library("ldatuning")

name = "categories/Chinese_processed.txt"
f = file(name)
doc_text=readLines(f)
close(f)

corpus = Corpus(VectorSource(doc_text))
dtm <- TermDocumentMatrix(corpus)
result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 5, to = 50, by = 2),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 3L,
  verbose = TRUE
)
