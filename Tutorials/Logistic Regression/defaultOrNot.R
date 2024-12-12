rm(list=ls())
library(ISLR)
library(caret)
data <- ISLR::Default
summary(data)

#Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.7,0.3))
train <- data[sample, ]
test <- data[!sample, ] 

model <- glm(default~student+balance+income, family="binomial", data=train)
options(scipen=999)
summary(model)

#Example taken from the training set
myexample <- data.frame(balance = 2205, income = 14271, student = "Yes")
predict(model, myexample, type = "response")

predictions <- predict(model, test, type = "response")
test$default <- ifelse(test$default=="Yes", TRUE, FALSE)
predictions <- ifelse(predictions >= 0.5, TRUE, FALSE)
confusionMatrix(factor(predictions), factor(test$default), positive = "TRUE")

simplerModel <- glm(default~balance, family = "binomial", data = train)
summary(simplerModel)
xs <- train$balance
ys <- predict(simplerModel, train, type = "response")
line <- data.frame(xs, ys)
line <- line[order(line$xs),]
plot(line, col="blue", type = "l")
points(xs[train$default == "Yes"], rep(1,length(xs[train$default == "Yes"]) ), col="red")
points(xs[train$default == "No"], rep(0, length(xs[train$default == "No"])), col="green")