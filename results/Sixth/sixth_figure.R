library(dplyr)
library(ggplot2)

GLVQ_RVFL <- read.csv(file="int_lvq_acc.csv", sep='\t', header=FALSE)
KGLVQ <- read.csv(file="kglvq_acc.csv", sep='\t', header=FALSE)
compData <- cbind(GLVQ_RVFL, KGLVQ)
colnames(compData) <- c("GLVQ", "KGLVQ")

mean(compData[, "GLVQ"])
mean(compData[, "KGLVQ"])

t.test(compData$GLVQ, compData$KGLVQ, paired = TRUE)

shapiro.test(compData$GLVQ)
shapiro.test(compData$KGLVQ)
wilcox.test(compData$GLVQ, compData$KGLVQ, paired = TRUE)

cor(compData$GLVQ, compData$KGLVQ)

png("sixth_figure.png", width=1000, height=900, res=300)

ggplot(compData, aes(x=KGLVQ, y=GLVQ)) +
  geom_point(size = 1, color = "blue") +
  geom_abline(slope=1, intercept=0) +
  xlim(0.3, 1.0) +
  ylim(0.3, 1.0) +
  xlab("Accuracy of KGLVQ Classifier") +
  ylab("Accuracy of the proposed approach") +
  theme_bw() +
  theme(axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        axis.title.x=element_text(size=10),
        axis.title.y=element_text(size=10))
#        text=element_text(family="Arial", size=10))

dev.off()

