library(grf)


df = read.csv("data/lalonde.csv")

covs <- c("age", "educ", "black", "hispanic", "married", "nodegree", "re75")

X <- df[, covs]
Y <- df$re78 
D <- df$treat

cf <- causal_forest(X, Y, D, W.hat = 0.5)
ate <- average_treatment_effect(cf)
varimp <- variable_importance(cf)
ranked.vars <- order(varimp, decreasing = TRUE)
colnames(X)[ranked.vars[1:5]]
best_linear_projection(cf, X[ranked.vars[1:5]])
ate


cfp <- causal_forest(X, Y, D)
atep <- average_treatment_effect(cfp)
varimp <- variable_importance(cfp)
ranked.vars <- order(varimp, decreasing = TRUE)
colnames(X)[ranked.vars[1:5]]
best_linear_projection(cfp, X[ranked.vars[1:5]])
atep


png("figs/grfr.png")
hist(e.hat <- cfp$W.hat)
dev.off()

test_calibration(cfp)
