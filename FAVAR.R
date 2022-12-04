library(FAVAR)
library(readxl)
library(boot)
library(tsDyn)
library(vars)
library(repr)
library(flexmix)
library(ggplot2)
library(flexmix)

observations = read.csv('observations.csv')
knownvariables = read.csv('knownvariables.csv')
#X = data.matrix(observations[2:465], rownames.force = NA)
#Y = data.matrix(knownvariables[2:7], rownames.force = NA)
#fit <- FAVAR(Y, X, slowcode = slowcode,fctmethod = 'BBE',
#             factorprior = list(b0 = 0, vb0 = NULL, c0 = 0.01, d0 = 0.01),
#             varprior = list(b0 = 0,vb0 = 10, nu0 = 0, s0 = 0),
#             nrep = 15, nburn = 5, K = 10, plag = 1)

#summary(fit)

data <- cbind(knownvariables[2:7],observations[2:465])
data_s <- scale(data, center = TRUE, scale = TRUE)
df_cr <- data.frame(0,0,0)
names(df_cr) <- c("N","AIC", "BIC")

for (r in 7:25)
{  
pc_all <- prcomp(data_s, center=FALSE, scale.=FALSE, rank. = r) 
C <- pc_all$x 
reg <- lm(C ~  data_s[,1:6])
F_hat <- C - data.matrix(data_s[,1:6])%*%reg$coefficients[2:7,] # cleaning and saving F_hat
colnames(F_hat) <- paste("Latent", colnames(F_hat), sep = "_")
data_var <- data.frame(F_hat, data_s[,1:6])
X = data.matrix(data_var, rownames.force = NA)
AIC = 0  
BIC = 0 
for (i in 2:465)
{
  Y = data.matrix(observations[i], rownames.force = NA)
  reg2 = lm(Y~X)
  AIC = AIC + AIC(reg2)
  BIC = BIC + BIC(reg2)
}
AICavg =  AIC/464
BICavg = BIC/464
df_cr[nrow(df_cr) + 1,] = c(r-5,AICavg,BICavg,)
}
df_cr[1,] = c(1,-1280.382,-1228.971)

ggplot()+geom_line(data = df_cr,aes(x = N,y = AIC,colour = "AIC"),size=0.5)+
  geom_line(data = df_cr,aes(x = N,y = BIC,colour = "BIC"),size=0.5)+
  scale_colour_manual("",values = c("AIC" = "green","BIC"="red"))+
  xlab("K")+ylab("Information Criteria")


r = 10

pc_all <- prcomp(data_s, center=FALSE, scale.=FALSE, rank. = r) 
C <- pc_all$x 
reg <- lm(C ~  data_s[,1:6])
F_hat <- C - data.matrix(data_s[,1:6])%*%reg$coefficients[2:7,] # cleaning and saving F_hat
colnames(F_hat) <- paste("Latent", colnames(F_hat), sep = "_")
data_var <- data.frame(F_hat, data_s[,1:6])
write.csv(data_var, "RF_FAVAR.csv")
X = data.matrix(data_var, rownames.force = NA)
Y = data.matrix(observations[2:465], rownames.force = NA)
reg_fin = lm(Y~X)
resid = residuals(reg_fin)
fitted_FARVA = fitted.values(reg_fin)
#write.csv(fitted_FARVA, "fitted_FARVA.csv", row.names=FALSE)
