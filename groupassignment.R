# Packes required for subsequent analysis. P_load ensures these will be installed and loaded. 
#devtools::install_github('Nth-iteration-labs/contextual')
#if(!require(ggnormalviolin)) devtools::install_github("wjschne/ggnormalviolin")
library(contextual)
library(ggnormalviolin)
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr,
               tidyr,
               ggplot2,
               reshape2,
               latex2exp,
               devtools,
               BiocManager)

# set the seed
set.seed(1234)

## OfflineReplayEvaluatorBandit: simulates a bandit based on provided data
#
# Arguments:
#
#   data: dataframe that contains variables with (1) the reward and (2) the arms
#
#   formula: should consist of variable names in the dataframe. General structure is: 
#       reward variable name ~ arm variable name
#
#   randomize: whether or not the bandit should receive the data in random order,
#              or as ordered in the dataframe.
#

# in our case, create a bandit for the data with 20 arms, 
#   formula = click ~ item_id, 
#   no randomization 
get_all_bandits <- function(df){
  bandit_OfflineReplay_item <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
  bandit_OfflineReplay_position <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_position, data = df, randomize = FALSE)
  
  # bandit contextual
  
  return(list(bandit_OfflineReplay_item = bandit_OfflineReplay_item, bandit_OfflineReplay_position = bandit_OfflineReplay_position))
}


get_policies <- function(epsilons, alphas){
  policies_all <- vector("list", length = 3)
  policies_greedy <- vector("list", length = length(epsilons))
  policies_UCB <- vector("list", length = length(alphas))
  policies_TS <- vector("list", length = 1)
  
  TS  <- ThompsonSamplingPolicy$new()
  
  for (idx in 1:length(epsilons)){
    policies_greedy[[idx]] <- EpsilonGreedyPolicy$new(epsilon = epsilons[idx])
  }
  
  for (idx in 1:length(alphas)){
    policies_UCB[[idx]] <- UCB2Policy$new(alpha = alphas[idx])
  }
  
  policies_TS[[1]] <- TS
  
  policies_all[[1]] <- policies_greedy
  policies_all[[2]] <- policies_UCB
  policies_all[[3]] <- policies_TS
  
  
  return(list(policies_TS = policies_TS, policies_greedy = policies_greedy, policies_UCB = policies_UCB, policies_all = policies_all))
}


get_agents <- function(policies, bandit_zozo){
  TS  <- policies$policies_TS[[1]] #[['policies_TS']] #ThompsonSamplingPolicy$new()
  UCBpolicies <- policies$policies_UCB
  EGpolicies <- policies$policies_greedy
  
  EG00 <- EGpolicies[[1]]
  EG01 <- EGpolicies[[2]]
  EG001 <- EGpolicies[[3]]
  
  UCB_01 <- UCBpolicies[[1]]
  UCB_05 <- UCBpolicies[[2]]
  
  
  
  agent_TS_zozo <-  Agent$new(TS, # add policy
                              bandit_zozo) # add bandit
  
  agent_ES_00eps_zozo <-  Agent$new(EG00, # add policy
                                    bandit_zozo, "e = 0, greedy") # add bandit
  agent_ES_01eps_zozo <-  Agent$new(EG01, # add policy
                                    bandit_zozo, "e = 0.1") # add bandit
  agent_ES_001eps_zozo <-  Agent$new(EG001, # add policy
                                     bandit_zozo, "e = 0.01") # add bandit
  
  UCB_01_agent <- Agent$new(UCB_01, bandit_zozo) # add our bandit
  
  UCB_05_agent <- Agent$new(UCB_05, bandit_zozo)
  #UCB_01_agent <- Agent$new(UCB_01, # add our UCB1 policy
  #                          bandit_zozo) # add our bandit
  
  # aggregate agents 
  agents_ES <- list(agent_ES_00eps_zozo, agent_ES_01eps_zozo, agent_ES_001eps_zozo)
  agents_UCB <- list(UCB_01_agent, UCB_05_agent)
  agents_all <- list(agent_ES_00eps_zozo, agent_ES_01eps_zozo, agent_ES_001eps_zozo, agent_TS_zozo, UCB_01_agent, UCB_05_agent)
  agents_UCBvTS <- list(UCB_01_agent, UCB_05_agent, agent_TS_zozo)
  
  return(list(agents_all = agents_all, agents_ES = agents_ES, agents_UCB = agents_UCB, agents_TS = agent_TS_zozo, agents_UCBvTS = agents_UCBvTS))
}



get_simulators <- function(agents, size_sim, n_sim){
  
  
  simulator_TS <- Simulator$new(agents$agents_TS, # set our agent
                                horizon= size_sim, # set the sizeof each simulation
                                do_parallel = TRUE, # run in parallel for speed
                                simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_ES00 <- Simulator$new(agents$agents_ES[[1]], # set our agent
                                horizon= size_sim, # set the sizeof each simulation
                                do_parallel = TRUE, # run in parallel for speed
                                simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_ES01 <- Simulator$new(agents$agents_ES[[2]], # set our agent
                                  horizon= size_sim, # set the sizeof each simulation
                                  do_parallel = TRUE, # run in parallel for speed
                                  simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_ES001 <- Simulator$new(agents$agents_ES[[3]], # set our agent
                                  horizon= size_sim, # set the sizeof each simulation
                                  do_parallel = TRUE, # run in parallel for speed
                                  simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_UCB01 <- Simulator$new(agents$agents_UCB[[1]], # set our agent
                                   horizon= size_sim, # set the sizeof each simulation
                                   do_parallel = TRUE, # run in parallel for speed
                                   simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_UCB05 <- Simulator$new(agents$agents_UCB[[2]], # set our agent
                                 horizon= size_sim, # set the sizeof each simulation
                                 do_parallel = TRUE, # run in parallel for speed
                                 simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_UCB <- Simulator$new(agents$agents_UCB, # set our agent
                                 horizon= size_sim, # set the sizeof each simulation
                                 do_parallel = TRUE, # run in parallel for speed
                                 simulations = n_sim, # simulate it n_sim times
  )
  
  simulator_UCBvTS <- Simulator$new(agents$agents_UCBvTS, # set our agent
                             horizon= size_sim, # set the sizeof each simulation
                             do_parallel = TRUE, # run in parallel for speed
                             simulations = n_sim, # simulate it n_sim times
  )
  
  return(list(simulator_TS = simulator_TS, simulator_ES00 = simulator_ES00, simulator_ES01 = simulator_ES01, 
              simulator_ES001 = simulator_ES001, simulator_UCB01 = simulator_UCB01, simulator_UCB05 = simulator_UCB05, 
              simulator_UCB = simulator_UCB, simulator_UCBvTS = simulator_UCBvTS))
}


run_simulatorsES <- function(simulators){
  history_ES00 <- simulators$simulator_ES00$run()
  history_ES01 <- simulators$simulator_ES01$run()
  history_ES001 <- simulators$simulator_ES001$run()
  
  return(list(history_ES00 = history_ES00, history_ES01 = history_ES01, history_ES001 = history_ES001))
}


run_simulatorsUCB <- function(simulators){
  history_UCB01 <- simulators$simulator_UCB01$run()
  history_UCB05 <- simulators$simulator_UCB05$run()
  history_UCB <- simulators$simulator_UCB$run()
  
  return(list(history_UCB01 = history_UCB01, history_UCB05 = history_UCB05, history_UCB = history_UCB))
}

run_simulatorsTS <- function(simulators){
  history_TS <- simulators$simulator_TS$run()
  
  return(list(history_TS = history_TS))
}

run_simulatorsUCVBvTS <- function(simulators){
  history_UCBvTS <- simulators$simulator_UCBvTS$run()
  
  return(list(history_UCBvTS = history_UCBvTS))
}


get_results <- function(history){
  df <- history$data %>%
    select(t, sim, choice, reward, agent) 
  return(df)
}


get_maxObsSims <- function(dfResults){
  dfResults_max_t <- dfResults %>%
    group_by(sim) %>% # group by per agent
    summarize(max_t = max(t)) # get max t
  return(dfResults_max_t)
}

show_results <- function(df_results){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  max_obs=9000
  df_history_agg <- df_results %>%
    group_by(sim)%>% # group by simulation
    mutate(cumulative_reward = cumsum(reward))%>% # calculate, per sim, cumulative reward over time
    group_by(t) %>% # group by timestep 
    summarise(avg_cumulative_reward = mean(cumulative_reward), # average cumulative reward
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>% # SE + Confidence interval
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <= max_obs)
  
  
  # define the legend of the plot
  legend <- c("Avg." = "orange", "95% CI" = "gray") # set legend
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward))+ 
    geom_line(size=1.5,aes(color="Avg."))+ # add line 
    geom_ribbon(aes(ymin=ifelse(cumulative_reward_lower_CI<0, 0,cumulative_reward_lower_CI),
                    # add confidence interval
                    ymax=cumulative_reward_upper_CI,
                    color = "95% CI"
    ), # 
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color='Metric')+ # add titles
    scale_color_manual(values=legend)+ # add legend
    theme_bw()+ # set the theme
    theme(text = element_text(size=16)) # enlarge text
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}


show_results_multipleagents <- function(df_results){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  max_obs=2500
  
  # data.frame aggregated for two versions: 20 and 40 arms
  df_history_agg <- df_results %>%
    group_by(agent, sim)%>% # group by number of arms, the sim
    mutate(cumulative_reward = cumsum(reward))%>% # calculate cumulative sum
    group_by(agent, t) %>% # group by number of arms, the t
    summarise(avg_cumulative_reward = mean(cumulative_reward),# calc cumulative reward, se, CI
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>%
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <=max_obs)
  
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward, color =agent))+
    geom_line(size=1.5)+
    geom_ribbon(aes(ymin=cumulative_reward_lower_CI , 
                    ymax=cumulative_reward_upper_CI,
                    fill = agent,
    ),
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color ='c', fill='c')+
    theme_bw()+
    theme(text = element_text(size=16))

  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}


show_results_multipleagents2 <- function(df_results){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  max_obs=2500

  # data.frame aggregated for two agents: UCB and epsilon greedy
  df_history_agg <- df_results %>%
    group_by(agent, sim)%>% # group by number of arms, the sim
    mutate(cumulative_reward = cumsum(reward))%>% # calculate cumulative sum
    group_by(agent, t) %>% # group by number of arms, the t
    summarise(avg_cumulative_reward = mean(cumulative_reward),# calc cumulative reward, se, CI
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>%
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <=max_obs)
  
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward, color =agent))+
    geom_line(size=1.5)+
    geom_ribbon(aes(ymin=cumulative_reward_lower_CI ,
                    ymax=cumulative_reward_upper_CI,
                    fill = agent,
    ),
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color ='Policy', fill='Policy')+ # set titles
    theme_bw()+ # set theme
    theme(text = element_text(size=16))+ # enlarge text
    scale_color_manual(values = c("red", "blue", "green"))
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}

# read in dataset
df <- read.csv('./zozo_Context_80items.csv')

df <- df %>% rename(
  reward = click,
  arm_item = item_id,
  arm_position = position
)


# generate a 10 simulations, each of size 100.000,
size_sim=100000
n_sim=10

epsilons <- c(0.0, 0.01, 0.001) # paramaters epsilon greedy 
alphas <- c(0.1, 0.5) # parameters UCB

bandits_zozo <- get_all_bandits(df)
bandit_OfflineReplay_item <- bandits_zozo$bandit_OfflineReplay_item
bandit_OfflineReplay_pos <- bandits_zozo$bandit_OfflineReplay_position


policies <- get_policies(epsilons, alphas)



## test

## define epsilon greedy policy
#eps_greedy <- EpsilonGreedyPolicy$new(epsilon = 0.1)
## define agent
#eps_greedy_agent <- Agent$new(eps_greedy, bandit_OfflineReplay_item)
## simulate
#simulatorepsgreedy <- Simulator$new(eps_greedy_agent, # set our agents
#                           horizon= size_sim, # set the sizeof each simulation
#                           do_parallel = TRUE, # run in parallel for speed
#                           simulations = n_sim, # simulate it n_sim times
#)
## run the simulator object
#history_eps_greedy <- simulatorepsgreedy$run()


### 
agents_items <- get_agents(policies, bandit_OfflineReplay_item)

simulators_items <- get_simulators(agents_items, size_sim, n_sim)

# histories

ES_sims_items <- run_simulatorsES(simulators_items)

UCB_sims_items <- run_simulatorsUCB(simulators_items)
UCBvTS_sims_items <- run_simulatorsUCVBvTS(simulators_items)
TS_sims_items <- run_simulatorsTS(simulators_items)

dfResults_UCB01_items <- get_results(UCB_sims_items$history_UCB01)
dfResults_UCBvTS__items <- get_results(UCBvTS_sims_items$history_UCBvTS)
max_obs_UCBvTS__items <- get_maxObsSims(dfResults_UCBvTS__items)

dfResults_ES01_items <- get_results(ES_sims_items$history_ES01)
max_obs_UCBvTS__items <- get_maxObsSims(dfResults_UCBvTS__items)


figures_ES01_items <- show_results(dfResults_ES01_items)

figures_ES01_items$fig2
#figures <- show_results(dfResults_UCBvTS_sims)
figures_items <- show_results_multipleagents(dfResults_UCBvTS__items)

#figures2 <- show_results_multipleagents2(dfResults_UCBvTS_sims)

#####
agents_pos <- get_agents(policies, bandit_OfflineReplay_pos)

simulators_pos <- get_simulators(agents_pos, size_sim, n_sim)

# statistche user features niet over tijd veranderen device website gender 
# histories
UCB_sims_pos <- run_simulatorsUCB(simulators_pos)
UCBvTS_sims_pos <- run_simulatorsUCVBvTS(simulators_pos)
TS_sims_pos <- run_simulatorsTS(simulators_pos)

dfResults_UCB01_pos <- get_results(UCB_sims_pos$history_UCB01)
dfResults_UCBvTS__pos <- get_results(UCBvTS_sims_pos$history_UCBvTS)
max_obs_UCBvTS__pos <- get_maxObsSims(dfResults_UCBvTS__pos)

figures_UCB01_pos <- show_results(dfResults_UCB01_pos)
#figures <- show_results(dfResults_UCBvTS_sims)
figures_pos <- show_results_multipleagents(dfResults_UCBvTS__pos)

#figures2 <- show_results_multipleagents2(dfResults_UCBvTS_sims)

figures_pos$fig2
plot(dfResults_UCB01, type = "average", regret = FALSE, lwd = 1, legend_position = "bottomright")
plot(TS_sims$history_TS, type = "optimal", lwd = 1, legend_position = "bottomright")



max_obs = 2500


## werkt niet voor #arms te groot?
# dataframe with arm choices for UCB
df_arm_choices_UCB_01 <- dfResults_UCB01%>%
  complete(t, sim,choice,fill = list(n = 0)) %>%
  group_by(t, sim,choice, .drop=FALSE) %>%
  mutate(n = sum(!is.na(reward))) %>%
  group_by(t, choice, .drop=FALSE)%>%
  summarise(sum_n = sum(n)) %>%
  group_by(choice, .drop=FALSE)%>%
  mutate(cum_sum_n = cumsum(sum_n),
         avg_cum_sum_n = cum_sum_n / (80 *t)) %>%
  filter(t <=max_obs)

ggplot(data = df_arm_choices_UCB_01, aes(x=t, y=avg_cum_sum_n, color = as.factor(choice)))+
  geom_line(size=1.5)+
  xlim(500, 2500)+
  theme_bw()+
  scale_color_brewer(palette="Spectral")+
  labs(color = 'Arm', x = 'Timestep', y='Average % of arms chosen ', main ='Arm Choices UCB')+
  theme(text = element_text(size=16))

