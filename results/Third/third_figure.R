library(dplyr)
library(ggplot2)

convLMS <- read.csv(file="conv_lms_acc.csv", sep='\t', header=FALSE)
intLMS <- read.csv(file="int_lms_acc.csv", sep='\t', header=FALSE)
compData <- cbind(convLMS, intLMS)
colnames(compData) <- c("conv", "int")

mean(compData[, "conv"])
mean(compData[, "int"])

t.test(compData$conv, compData$int, paired = TRUE)

shapiro.test(compData$conv)
shapiro.test(compData$int)
wilcox.test(compData$conv, compData$int, paired = TRUE)

cor(compData$conv, compData$int)

png("third_figure.png", width=1000, height=900, res=300)

ggplot(compData, aes(x=conv, y=int)) +
  geom_point(size = 1, color = "blue") +
  geom_abline(slope=1, intercept=0) +
  xlim(0.1, 1.0) +
  ylim(0.1, 1.0) +
  xlab("Accuracy of conventional RVFL") +
  ylab("Accuracy of intRVFL") +
  theme_bw() +
  theme(axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        axis.title.x=element_text(size=10),
        axis.title.y=element_text(size=10))
#        text=element_text(family="Arial", size=10))

dev.off()

