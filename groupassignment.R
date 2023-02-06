if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr,
               tidyr,
               ggplot2,
               reshape2,
               latex2exp,
               contextual
)

df <- read.csv('./zozo_Context_80items.csv')

df <- df %>% rename(
    reward = click,
    arm = position
  )


# reads in full csv of Yahoo dataset
dfYahoo <- read.csv('./yahoo_day1_10arms.csv')[,-c(1,2)]
# selects the two relevant columns from Yahoo dataset; arm shown to user and reward observed
dfYahoo_for_sim <- dfYahoo %>% select(arm, reward)
dfYahoo_for_sim$index <- 1:nrow(dfYahoo_for_sim)

# grab first 100 observations, create dataframe to practice for policy_greedy() function
df_practice <- dfYahoo_for_sim[1:100,]
# set value of epsilon parameter
eps = 0.
#####
# policy_greedy : picks an arm, based on the greedy algorithm for multi - armed bandits
#
# Arguments :
# df: data.frame with two columns: arm, reward
# eps : float, percentage of times a random arm needs to be picked
# n_arms: integer, the total number of arms available. Standard value is 10.
#
# Output :
# chosen_arm ; integer, index of the arm chosen
####
policy_greedy <- function(df, eps, n_arms=10){
  # draws a random float between 0 and 1
  random_uniform_variable <- runif(1, min=0, max=1)
  # in epsilon % of cases, pick a random arm
  if (random_uniform_variable < eps){
    # sample uniform random arm between 1-n_arms
    chosen_arm <- sample(1:n_arms,1 )
    # in (1-epsilon)% of cases, pick the arm that has the highest average reward
  }else{
    # create dataframe; per arm it shows
    # - succes_size: total reward for arm
    # - sample_size: total observations for arm
    # - succes_rate: total reward/total observations per arm
    df_reward_overview <- df %>%
      drop_na() %>%
      group_by(arm)%>%
      summarise(reward_size = sum(reward, na.rm=TRUE),
                sample_size = n(),
                succes_rate = reward_size/sample_size
      )
    # pick the arm with the highest success rate
    chosen_arm <- which.max(df_reward_overview$succes_rate)
  }
  # return the chosen arm
  return(chosen_arm)
}
# get the chosen arm from the function
chosen_arm_practice <- policy_greedy(df_practice, eps)

print(paste0('The chosen arm is: ', chosen_arm_practice))