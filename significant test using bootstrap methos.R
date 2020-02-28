library(forecast)
library(tseries)
library(boot)
setwd("~/Desktop/IAQF/bmg_data")
stocks <-read.csv('30 stocks log daily return(new).csv', header =T)
rf <- read.csv('F-F_Research_Data_Factors_daily.csv',header =T, stringsAsFactors =F)
ret <- read.csv('new_data.csv', head = T, stringsAsFactors =F)
index = which(rf[,1]>19800729)
index = which()
MKT <- as.vector(rf[,2][index])
SMB <- as.vector(rf[,3][index])
HML <- as.vector(rf[,4][index])
ffm = cbind(MKT,SMB,HML)
riskfree <- rf[,5][index]

logret_rep <- ret[,2] - riskfree
logret_dem <- ret[,3] -riskfree

rf_new <- rep(0.001, 25)
riskfree <- c(riskfree, rf_new)
ret$rm_rf <- ret$SPX.Index - riskfree

c_c <- function(data=ret, indices) {
  d <- ret[indices, ]
  model_rep <- lm(logret_rep[1:9941] ~ ffm, data=d)
  model_dem <- lm(logret_dem[1:9941] ~ ffm , data=d)
  return(coef(model_rep)[4] - coef(model_dem)[4])
}
result <- boot(data=ret, statistic=c_c, R=1000);
result;
#intercept
4.05432e-06/0.0001019589
#mkt-rf
-0.0002989607/9.557233e-05
#smb
0.0004785483/0.0001757919
#hml
-6.969441e-05/0.0001825584
c_c1 <- function(data=ret, indices) {
  d <- ret[indices, ]
  model_rep <- lm(logret_rep ~ rm_rf, data=d)
  model_dem <- lm(logret_dem ~ rm_rf , data=d)
  return(coef(model_rep)[1] - coef(model_dem)[1])
}
result1 <- boot(data=ret, statistic=c_c1, R=1000);
result1;
-6.386094e-06/9.874519e-05



