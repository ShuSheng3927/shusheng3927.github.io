# 0 means group A 
# 1 means group B 


# true success prob of A and B
prob_A <- 0.2
prob_B <- 0.65
prob <- c(prob_A, prob_B)

# total number of patients
n <- 12


# RPW(u, alpha, beta)
u <- 10
alpha <- 0
beta <- 1


n_A_result <- c()
n_B_result <- c()

iter <- 100000

for (j in 1:iter){
  
  #current urn contents
  urn_A <- u
  urn_B <- u
  urn <- c(urn_A, urn_B)
  
  
  arm_result <- rep(NA, n)
  outcome_result <- rep(NA, n)
  
  for (i in 1:n){
    arm <- rbinom(1,1,urn[2]/sum(urn)) + 1
    outcome <- rbinom(1,1,prob[arm])
    arm_result[i] <- arm
    outcome_result[i] <- outcome
    if ((arm + outcome) %% 2 == 0){
      urn <- urn + c(beta,alpha)
    }else{
      urn <- urn + c(alpha,beta)
    }
  }
  
  #result <- rbind(arm_result, outcome_result)
  n_A <- sum(arm_result == 1)
  n_B <- sum(arm_result == 2)
  
  n_A_result <- c(n_A_result, n_A)
  n_B_result <- c(n_B_result, n_B)
}


print(prob[1])
print(mean(n_A_result))
print(sd(n_A_result))


print(prob[2])
print(mean(n_B_result))
print(sd(n_B_result))

print(mean(n_B_result > 10))
