library(dplyr)
library(ggplot2)

intLMS <- read.csv(file="int_lms_acc.csv", sep='\t', header=FALSE)
intLVQ <- read.csv(file="int_lvq1_acc.csv", sep='\t', header=FALSE)
compData <- cbind(intLMS, intLVQ)
colnames(compData) <- c("LMS", "LVQ")

mean(compData[, "LMS"]) 
mean(compData[, "LVQ"])

t.test(compData$LMS, compData$LVQ, paired = TRUE)

shapiro.test(compData$LMS)
shapiro.test(compData$LVQ)
wilcox.test(compData$LMS, compData$LVQ, paired = TRUE)

cor(compData$LMS, compData$LVQ)

png("fourth_figure.png", width=1000, height=900, res=300)

ggplot(compData, aes(x=LMS, y=LVQ)) +
  geom_point(size = 1, color = "blue") +
  geom_abline(slope=1, intercept=0) +
  xlim(0.1, 1.0) +
  ylim(0.1, 1.0) +
  xlab("Accuracy of intRVFL with RLS") +
  ylab("Accuracy of intRVFL with GLVQ") +
  theme_bw() +
  theme(axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        axis.title.x=element_text(size=10),
        axis.title.y=element_text(size=10))
#        text=element_text(family="Arial", size=10))

dev.off()

