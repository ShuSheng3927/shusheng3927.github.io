# NOTATION
# n: number of patients in trial.
# prior.i: prior number of successes on treatment A.
# prior.j: prior number of failures on treatment A.
# prior.k: prior number of successes on treatment B.
# prior.l: prior number of failures on treatment B.
# V: value function representing the maximum expected total reward (i.e. number of successes) in the rest of the trial after t patients have been treated.
# t: number of patients that have been treated.
# p: the degree of randomisation.
# Y: minimum number of observations required on each treatment arm (i.e. the degree of constraining).
# i: observed number of successes on treatment A.
# j: observed number of failures on treatment A.
# k: observed number of successes on treatment B.
# l: observed number of failures on treatment B.

CRDP <- function(n, prior.i, prior.j, prior.k, prior.l, p, Y){
  
  V      <- array(0, dim = c(n+1, n+1, n+1))
  Action <- array(0, dim = c(n+1, n+1, n+1, n+1 ))
  t      <- n+4
  
  for (i in 1:(t-3)){
    for (j in 1:(t-i-2)){
      for (k in 1:(t-i-j-1)){
        
        l <- (t-i-j-k)
        
        if (i+j < Y) V[i,j,k] <- -10*n
        if (k+l < Y) V[i,j,k] <- -10*n
        
      }
    }
  }
  
  for(t in (n+3):4){
    for (i in 1:(t-3)){
      for (j in 1:(t-i-2)){
        for (k in 1:(t-i-j-1)){
          
          l <- (t-i-j-k)
          
          expected.prob.c <- (i - 1 + prior.i) / (i - 1 + prior.i + j - 1 + prior.j)
          expected.prob.n <- (k - 1 + prior.k) / (k - 1 + prior.k + l - 1 + prior.l)
          
          Vcontrol <- expected.prob.c*(1 + V[i+1, j, k]) + (1 - expected.prob.c)*(0 + V[i, j+1, k])
          Vnovel   <- expected.prob.n*(1 + V[i, j, k+1]) + (1 - expected.prob.n)*(0 + V[i, j, k])
          
          if (p*Vcontrol + (1-p)*Vnovel > (1-p)*Vcontrol + p*Vnovel) {
            Action[n-(t-4), i, j, k] <- 0
            # Action 1 is optimal for the next patient.
          }
          if (p*Vcontrol + (1-p)*Vnovel < (1-p)*Vcontrol + p*Vnovel) {
            Action[n-(t-4), i, j, k] <- 1
            # Action 2 is optimal for the next patient.
          } else {
            if (p*Vcontrol + (1-p)*Vnovel == (1-p)*Vcontrol + p*Vnovel){
              Action[n-(t-4), i ,j, k] <- 2
              # Either Action 1 or 2 is optimal for the next patient.
            }
          }
          V[i,j,k] <- max( p*Vcontrol + (1-p)*Vnovel, (1-p)*Vcontrol + p*Vnovel )
        }
      }
    }
  }
  
  return(Action)
  # Returns the optimal action for the next patient.
  
}


truth <- function(theta_A, theta_B,group){
  if (group == 0){
    # group 0 is A
    outcome <- rbinom(1,1,theta_A) 
  }else{
    # group 1 is B
    outcome <- rbinom(1,1,theta_B) 
  }
  
  # outcome 0 is FAILURE, 1 is SUCCESS
  return(outcome)
}




n = 100
prior.i = 1
prior.j = 1
prior.k = 1
prior.l = 1


theta_A <- 0.6
theta_B <- 0.7



test_CRDP <- c()
delta_CRDP <- c()
PBA_CRDP <- c()

test_DP <- c()
delta_DP <- c()
PBA_DP <- c()

test_RCT <- c()
delta_RCT <- c()
PBA_RCT <- c()



iter <- 1000

for (j in 1:iter){
  print(j)
  result_CRDP <- CRDP(n, prior.i, prior.j, prior.k, prior.l,p = 0.9,15)
  result_DP <- CRDP(n, prior.i, prior.j, prior.k, prior.l,p = 1,0)
  result_RCT <- rbinom((n+1),1,0.5)
  
  
  
  
  patients_CRDP <- c()
  outcome_CRDP <- c()
  s_A_CRDP <- 1
  f_A_CRDP <- 1
  s_B_CRDP <- 1
  f_B_CRDP <- 1
  
  patients_DP <- c()
  outcome_DP <- c()
  s_A_DP <- 1
  f_A_DP <- 1
  s_B_DP <- 1
  f_B_DP <- 1
  
  patients_RCT <- c()
  outcome_RCT <- c()
  s_A_RCT <- 1
  f_A_RCT <- 1
  s_B_RCT <- 1
  f_B_RCT <- 1
  
  
  
  for (i in 2:(n+1)){
    action_CRDP <- result_CRDP[(n+2-i),s_A_CRDP, f_A_CRDP,s_B_CRDP]
    
    if (action_CRDP == 2){
      group <- rbinom(1,1,0.5)
    }else{
      group <- action_CRDP
    }
    
    patients_CRDP <- c(patients_CRDP, group)
    outcome <- truth(theta_A,theta_B,group)
    outcome_CRDP <- c(outcome_CRDP, outcome)
    if (group == 0){
      if (outcome == 0){
        f_A_CRDP <- f_A_CRDP + 1
      }else{
        s_A_CRDP <- s_A_CRDP + 1
      }
    }else{
      if (outcome == 0){
        f_B_CRDP <- f_B_CRDP + 1
      }else{
        s_B_CRDP <- s_B_CRDP + 1
      }
    }
    
    
    
    action_DP <- result_DP[(n+2-i),s_A_DP, f_A_DP,s_B_DP]
    
    if (action_DP == 2){
      group <- rbinom(1,1,0.5)
    }else{
      group <- action_DP
    }
    
    patients_DP <- c(patients_DP, group)
    outcome <- truth(theta_A,theta_B,group)
    outcome_DP <- c(outcome_DP, outcome)
    if (group == 0){
      if (outcome == 0){
        f_A_DP <- f_A_DP + 1
      }else{
        s_A_DP <- s_A_DP + 1
      }
    }else{
      if (outcome == 0){
        f_B_DP <- f_B_DP + 1
      }else{
        s_B_DP <- s_B_DP + 1
      }
    }
    
    
    
    
    
    action_RCT <- result_RCT[i]
    
    if (action_RCT == 2){
      group <- rbinom(1,1,0.5)
    }else{
      group <- action_RCT
    }
    
    patients_RCT <- c(patients_RCT, group)
    outcome <- truth(theta_A,theta_B,group)
    outcome_RCT <- c(outcome_RCT, outcome)
    if (group == 0){
      if (outcome == 0){
        f_A_RCT <- f_A_RCT + 1
      }else{
        s_A_RCT <- s_A_RCT + 1
      }
    }else{
      if (outcome == 0){
        f_B_RCT <- f_B_RCT + 1
      }else{
        s_B_RCT <- s_B_RCT + 1
      }
    }
  }
  
  
  
  
  
  
  s_A_CRDP <- s_A_CRDP - 1
  f_A_CRDP <- f_A_CRDP - 1
  s_B_CRDP <- s_B_CRDP - 1
  f_B_CRDP <- f_B_CRDP - 1
  
  p_CRDP <- fisher.test(matrix(c(s_A_CRDP,s_B_CRDP,f_A_CRDP,f_B_CRDP),nrow=2))$p
  if (p_CRDP < 0.1){
    test_CRDP <- c(test_CRDP, 1)
  }else{
    test_CRDP <- c(test_CRDP, 0)
  }
  
  delta_CRDP <- c(delta_CRDP, s_A_CRDP / (s_A_CRDP + f_A_CRDP) - s_B_CRDP / (s_B_CRDP + f_B_CRDP))
  
  if (which.max(c(theta_A, theta_B)) == 1){
    PBA_CRDP <- c(PBA_CRDP, (s_A_CRDP + f_A_CRDP) / n)
  } else{
    PBA_CRDP <- c(PBA_CRDP, (s_B_CRDP + f_B_CRDP) / n)
  }
  
  
  s_A_DP <- s_A_DP - 1
  f_A_DP <- f_A_DP - 1
  s_B_DP <- s_B_DP - 1
  f_B_DP <- f_B_DP - 1
  
  p_DP <- fisher.test(matrix(c(s_A_DP,s_B_DP,f_A_DP,f_B_DP),nrow=2))$p
  if (p_DP < 0.1){
    test_DP <- c(test_DP, 1)
  }else{
    test_DP <- c(test_DP, 0)
  }
  
  delta_DP <- c(delta_DP, s_A_DP / (s_A_DP + f_A_DP) - s_B_DP / (s_B_DP + f_B_DP))
  
  if (which.max(c(theta_A, theta_B)) == 1){
    PBA_DP <- c(PBA_DP, (s_A_DP + f_A_DP) / n)
  } else{
    PBA_DP <- c(PBA_DP, (s_B_DP + f_B_DP) / n)
  }
  
  
  s_A_RCT <- s_A_RCT - 1
  f_A_RCT <- f_A_RCT - 1
  s_B_RCT <- s_B_RCT - 1
  f_B_RCT <- f_B_RCT - 1
  
  p_RCT <- fisher.test(matrix(c(s_A_RCT,s_B_RCT,f_A_RCT,f_B_RCT),nrow=2))$p
  if (p_RCT < 0.1){
    test_RCT <- c(test_RCT, 1)
  }else{
    test_RCT <- c(test_RCT, 0)
  }
  
  delta_RCT <- c(delta_RCT, s_A_RCT / (s_A_RCT + f_A_RCT) - s_B_RCT / (s_B_RCT + f_B_RCT))
  
  if (which.max(c(theta_A, theta_B)) == 1){
    PBA_RCT <- c(PBA_RCT, (s_A_RCT + f_A_RCT) / n)
  } else{
    PBA_RCT <- c(PBA_RCT, (s_B_RCT + f_B_RCT) / n)
  }     
}







mean(test_RCT)
mean(test_DP)
mean(test_CRDP)

delta_true <- theta_A - theta_B
mean((delta_RCT - delta_true)^2)
mean((delta_DP - delta_true)^2)
mean((delta_CRDP - delta_true)^2)

mean(PBA_RCT)
mean(PBA_DP)
mean(PBA_CRDP)

