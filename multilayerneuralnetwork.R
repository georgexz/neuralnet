library(stats)
library(ggplot2)
library(plot3D)
numdata = 100

numnodeseachlayer <- c(1, 20, 30, 20, 10, 15, 15, 1)

x <- matrix(runif(numdata*numnodeseachlayer[1], -10, 10), nrow = numnodeseachlayer[1])
#y <- matrix(-.04*x[1,]^3 + x[2,]^2 -x[3,] + 2*x[4,]-x[5,]*x[6,] - x[7,]*x[8,], nrow = 1)
#y <- matrix(x[1,] + 0.5*x[2,] + 0.3 * x[3,] + (x[4,]) + (x[5,]) - 2*x[6,] -(x[7,]) -(x[8,]), nrow = 1)
#y <- matrix(x[1,]-x[1,]^2 + x[2,] - x[2,]^2 + x[1,]*x[2,], nrow = 1)
#y <- matrix(-x[1,]^5 + x[1,]^3-6*x[1,]^2 + 4*x[1,], nrow = 1)
y <- matrix(sin(x[1,]), nrow = numnodeseachlayer[length(numnodeseachlayer)])

constantslist <- c()
for (i in 1:(length(numnodeseachlayer) - 1) ) {
  constantslist[[i]] <- matrix(runif(numnodeseachlayer[i]*numnodeseachlayer[i+1], -1, 1), nrow = numnodeseachlayer[i+1], ncol = numnodeseachlayer[i])
}

hiddenfunc <- function(...) {
  #return(1/(1 + exp(-1*input)))
  return(log(1+exp(...)))
}
derivhiddenfunc <- function(...) {
  #return((1/(1 + exp(-1*input)))*(1-(1/(1 + exp(-1*input)))))
  return(1/(1 + exp(-1*...)))
}
hiddenlayer <- function(vecinput, func, constants) {
  vecoutput <- constants %*% vecinput
  funccalc <- func(vecoutput)
  return(funccalc)
}

outputlayer <- function(vecinput, constants) {
  vecoutput <- constants %*% vecinput
  return(vecoutput)
}
#NN

nny <- matrix(nrow=numnodeseachlayer[length(numnodeseachlayer)], ncol = numdata)
alpha = 0.000005
#initialize gradients
grad <- c()
for(i in 1: (length(numnodeseachlayer) - 1) ) {
  grad[[i]] <- matrix(nrow = numnodeseachlayer[i + 1], ncol = numnodeseachlayer[i])
}

#initialize momentum
#v1 <- matrix(runif(numhiddennodes1*numinput, 0, 1), nrow = numhiddennodes1, ncol = numinput)
#v2 <- matrix(runif(numhiddennodes1*numhiddennodes2, 0, 1), nrow = numhiddennodes2, ncol = numhiddennodes1)
#v3 <- matrix(runif(numhiddennodes2*numoutput, 0, 1), nrow = numoutput, ncol = numhiddennodes2)

#calculate layers input
layers <- c()
for( i in 1:(length(constantslist) - 1) ) {
  if (i == 1)
    layers[[i]] <- hiddenlayer(x, hiddenfunc, constantslist[[i]])
  else
    layers[[i]] <- hiddenlayer(layers[[i-1]], hiddenfunc, constantslist[[i]])
}

nny <- outputlayer(layers[[length(layers)]], constantslist[[length(constantslist)]])

#MSE
MSE <- 0.5*sum((nny-y)^2)
count <- 0
prevMSE <- MSE + 2
#recalculate with new constants
while(MSE>0.03 & abs(prevMSE-MSE) > 0.00000001) {
  count <- count + 1
  #calculate derivative of layers
  derivlayers <- c() 
  for(i in 1: length(layers)) {
    if (i ==1)
      derivlayers[[i]] <- derivhiddenfunc(constantslist[[i]] %*% x)
    else 
      derivlayers[[i]] <- derivhiddenfunc(constantslist[[i]] %*% layers[[i-1]])
  }

  #repnnyminusy <- t(matrix(rep(nny-y, numhiddennodes1),nrow=numdata/numinput ))
  #repconstants3 <- matrix(rep(constants3, numdata/numinput),ncol=numdata/numinput )
  deltas <- c()
  for(i in length(constantslist):1) {
    if(i == length(constantslist))
      deltas[[i]] <- nny - y
    else
      deltas[[i]] <- (t(constantslist[[i+1]]) %*% deltas[[i+1]])* derivlayers[[i]]
  }

  #gradient calculation
  for(i in 1: length(constantslist)) {
    if (i == 1)
      grad[[i]] <- deltas[[i]] %*% t(x)
    else
      grad[[i]] <- deltas[[i]] %*%t(layers[[i-1]])
  }
  
  
  #momentum variables
  #v1 <- 0.5*v1 + grad1
  #v2 <- 0.5*v2 + grad2
  #v3 <- 0.5*v3 + grad3
  #replace bottom grad with momentum variables
  #update constants
  for(i in 1: length(constantslist)) {
    constantslist[[i]] <- constantslist[[i]] - alpha * grad[[i]]
  }

  
  #update nny and layers
  for( i in 1:(length(constantslist) - 1) ) {
    if (i == 1)
      layers[[i]] <- hiddenlayer(x, hiddenfunc, constantslist[[i]])
    else
      layers[[i]] <- hiddenlayer(layers[[i-1]], hiddenfunc, constantslist[[i]])
  }
  nny <- outputlayer(layers[[length(layers)]], constantslist[[length(constantslist)]])
  
  prevMSE <- MSE
  #MSE
  MSE <- 0.5*sum((y-nny)^2)
}
plot(x, y, col = "blue", pch = 20)
points(x, nny, col = "red", pch = 20)
#plotting and other testing
scatter3D(as.matrix(t(x[1,])),as.matrix(t(x[2,])),as.matrix(y))
a_list = list(x, y, nny)


# as.matrix(t(x[1,]))
# p1 <- plot(x[1,], y)
# p <- plot(x[1,], nny)
# p2 <- plot(x[2,], y)
# p3 <- plot(x[3,], y)
# p4 <- plot(x[4,], y)
# p5 <- plot(x[5,], y)
# p6 <- plot(x[6,], y)
# p7 <- plot(x[7,], y)
# p8 <- plot(x[8,], y)
