# Packes required for subsequent analysis. P_load ensures these will be installed and loaded. 
#devtools::install_github('Nth-iteration-labs/contextual')
#if(!require(ggnormalviolin)) devtools::install_github("wjschne/ggnormalviolin")
library(contextual)
library(ggnormalviolin)
library(factoextra)
library(fastDummies)
library(FactoMineR)
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
  max_obs=1000
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


show_results_multipleagents <- function(df_results,max_obs=900){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  
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

classicPCA <- function(mX,k){
  #perform pca
  pca <- prcomp(mX)
  pca_rotation <- pca$rotation
  
  #print largest loadings
  print(sort(abs(pca_rotation[,1])))
  print(sort(abs(pca_rotation[,2])))
  print(sort(abs(pca_rotation[,3])))
  
  #compute principal components
  mX_pca <- as.matrix(mX)%*%pca$rotation[,1:k]
  
  #create scree plot
  print(fviz_eig(pca))
  return(mX_pca)
}

# read in dataset
df <- read.csv('./zozo_Context_80items.csv')

df <- df %>% rename(
  reward = click,
  arm_item = item_id,
  arm_position = position
)

x = df$arm_item
y = df$arm_position
z = rep(0, length(x))
for (i in 1:length(x)){
  z[i] = x[i]*3-2 + y[i]-1
}
df$arm_item_position <- as.integer(z)

# Create dummies
arm_dummy <- dummy_cols(df$arm_position)[2:4]
colnames(arm_dummy) <- c("arm_1", "arm_2", "arm_3")
feature_0_dummy <- dummy_cols(df$user_feature_0)[,2:5]
colnames(feature_0_dummy) <- c("f00", "f01", "f02", "f03")
feature_1_dummy <- dummy_cols(df$user_feature_1)[,2:7]
colnames(feature_1_dummy) <- c("f10", "f11", "f12", "f13", "f14", "f15")
feature_2_dummy <- dummy_cols(df$user_feature_2)[,2:11]
colnames(feature_2_dummy) <- c("f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29")
feature_3_dummy <- dummy_cols(df$user_feature_3)[,2:11]
colnames(feature_3_dummy) <- c("f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39")

dummy_matrix <- data.frame(arm_dummy, feature_0_dummy, feature_1_dummy, feature_2_dummy, feature_3_dummy)
mX_pca <- classicPCA(dummy_matrix[,4:33],9)
colnames(mX_pca) <- c("pc1","pc2","pc3","pc4","pc5","pc6","pc7","pc8","pc9")
df <- cbind(df, dummy_matrix, mX_pca)

dummy_matrix_test <- dummy_matrix[1:10000,]

dist_matrix <- dist(dummy_matrix_test, method = "binary")

#perform hierarchical clustering
hc<-hclust(dist_matrix, method="complete") 
plot(hc, hang=-1) 
rect.hclust(hc, k=4, border="red") 
clust.vec.5<-cutree(hc, k=4) 
plot(density(dist_matrix))

#plots clusters selected
fviz_cluster(list(data=dummy_matrix_test, cluster=clust.vec.5))


# begin modellen
# generate a 10 simulations, each of size 100.000,
size_sim=100000
n_sim=10

# hyperparameters
alphas <- c(0.1, 0.5) # parameters UCB/linucb


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





#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

size_sim=200000
n_sim=9
alpha= 0.1
bandit_item <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item_pc <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item |pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)
bandit_item_pos <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position, data = df, randomize = FALSE)
bandit_item_pos_pc <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position |pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)

UCBitem <- LinUCBDisjointPolicy$new(alpha=alpha)
UCBitem_pc <- LinUCBDisjointPolicy$new(alpha=alpha)
UCBitempos <- LinUCBDisjointPolicy$new(alpha=alpha)
UCBitempos_pc <- LinUCBDisjointPolicy$new(alpha=alpha)

agent_item <- Agent$new(UCBitem, bandit_item, name="itemUCB")
agent_item_pc <- Agent$new(UCBitem_pc, bandit_item_pc, name="itempcUCB")
agent_itempos <- Agent$new(UCBitempos, bandit_item_pos, name="itemposUCB")
agent_itempos_pc <- Agent$new(UCBitempos_pc, bandit_item_pos_pc, name="itempospcUCB")

simulator <- Simulator$new(list(agent_item,agent_item_pc,agent_itempos,agent_itempos_pc), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1730)






size_sim=400000
n_sim=12
alpha= 0.1
bandit_item <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item_pc <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)
bandit_item_pos <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position, data = df, randomize = FALSE)
bandit_item_pos_pc <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)

par<- 0.01
TSitem <- ContextualLinTSPolicy$new(v=par)
TSitem_pc <- ContextualLinTSPolicy$new(v=par)
TSitempos <- ContextualLinTSPolicy$new(v=par)
TSitempos_pc <- ContextualLinTSPolicy$new(v=par)

agent_item <- Agent$new(TSitem, bandit_item, name="itemTS")
agent_item_pc <- Agent$new(TSitem_pc, bandit_item_pc, name="itempcTS")
agent_itempos <- Agent$new(TSitempos, bandit_item_pos, name="itemposTS")
agent_itempos_pc <- Agent$new(TSitempos_pc, bandit_item_pos_pc, name="itempospcTS")

simulator <- Simulator$new(list(agent_item_pc,agent_itempos_pc), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1800)



size_sim=200000
n_sim=9
alpha= 0.1
bandit_UCB <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize = FALSE)
bandit_TS <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)
bandit_UCB2 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize = FALSE)
bandit_TS2 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item_position | pc1+pc2+pc3+pc4+pc5+pc6+pc7+pc8+pc9, data = df, randomize=FALSE)

par <- 0.01
UCB <- LinUCBDisjointPolicy$new(alpha=0.1)
TS <- ContextualLinTSPolicy$new(v=par)
UCB2 <- LinUCBDisjointPolicy$new(alpha=0.1)
TS2 <- ContextualLinTSPolicy$new(v=par)

agent_UCB <- Agent$new(UCB, bandit_UCB, name="item UCB")
agent_TS <- Agent$new(TS, bandit_TS, name="item TS")
agent_UCB2 <- Agent$new(UCB2, bandit_UCB2, name="itempos UCB")
agent_TS2 <- Agent$new(TS2, bandit_TS2, name="itempos TS")

simulator <- Simulator$new(list(agent_UCB,agent_TS,agent_UCB2,agent_TS2), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,2413)

1/(sum(df$reward)/length(df$reward))

#BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB

#
valpha=c(0.05, 0.1,0.5,1,5)
bandit_item05 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item1 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item5 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item10 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_item50 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
help("OfflineReplayEvaluatorBandit")
UCBitem05 <- LinUCBDisjointPolicy$new(alpha=valpha[1])
UCBitem1 <- LinUCBDisjointPolicy$new(alpha=valpha[2])
UCBitem5 <- LinUCBDisjointPolicy$new(alpha=valpha[3])
UCBitem10 <- LinUCBDisjointPolicy$new(alpha=valpha[4])
UCBitem50 <- LinUCBDisjointPolicy$new(alpha=valpha[5])

agentitem05 <- Agent$new(UCBitem05, bandit_item05, name='itemUCB 0.05')
agentitem1 <- Agent$new(UCBitem1, bandit_item1, name='itemUCB 0.1')
agentitem5 <- Agent$new(UCBitem5, bandit_item5, name='itemUCB 0.5')
agentitem10 <- Agent$new(UCBitem10, bandit_item10, name='itemUCB 1')
agentitem50 <- Agent$new(UCBitem50, bandit_item50, name='itemUCB 5')

size_sim=100000
n_sim=9

simulator <- Simulator$new(list(agentitem05,agentitem1,agentitem5,agentitem10,agentitem50), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,950)
help(Simulator)




vN <- c(20000,50000,100000,200000)
n_sim = 6
alpha = 0.1
bandit_UCB <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
bandit_TS <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize=FALSE)

UCB_item <- LinUCBDisjointPolicy$new(alpha=alpha)
TS_item <- ThompsonSamplingPolicy$new()

agent_UCB <- Agent$new(UCB_item, bandit_UCB, name="itemUCB")
agent_TS <- Agent$new(TS_item, bandit_TS, name="itemTS")

simulator <- Simulator$new(list(agent_UCB,agent_TS), # set our agents
                           horizon= vN[4], # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,2410)
help(Simulator)






#CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

df_pos1 <- df[which(df$arm_position == 1)]
df_pos2 <- df[which(df$arm_position == 2)]
df_pos3 <- df[which(df$arm_position == 3)]

size_sim=100000
n_sim=8
alpha= 0.1
bandit_pos1 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos1, randomize = FALSE)
bandit_pos2 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos2, randomize=FALSE)
bandit_pos3 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos3, randomize = FALSE)

UCBpos1 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCBpos2 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCBpos3 <- LinUCBDisjointPolicy$new(alpha=alpha)

agent_pos1 <- Agent$new(UCBpos1, bandit_pos1, name="UCB pos1")
agent_pos2 <- Agent$new(UCBpos2, bandit_pos2, name="UCB pos2")
agent_pos3 <- Agent$new(UCBpos3, bandit_pos3, name="UCB pos3")

simulator <- Simulator$new(list(agent_pos1,agent_pos2,agent_pos3), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,875)




size_sim=100000
n_sim=8
alpha= 0.1
bandit_pos1 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos1, randomize = FALSE)
bandit_pos2 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos2, randomize=FALSE)
bandit_pos3 <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df_pos3, randomize = FALSE)

TSpos1 <- ThompsonSamplingPolicy$new(alpha=alpha)
TSpos2 <- ThompsonSamplingPolicy$new(alpha=alpha)
TSpos3 <- ThompsonSamplingPolicy$new(alpha=alpha)

agent_pos1 <- Agent$new(TSpos1, bandit_pos1, name="TS pos1")
agent_pos2 <- Agent$new(TSpos2, bandit_pos2, name="TS pos2")
agent_pos3 <- Agent$new(TSpos3, bandit_pos3, name="TS pos3")

simulator <- Simulator$new(list(agent_pos3), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1191)





bandit_UCB <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
alpha=0.1
UCBitem <- LinUCBDisjointPolicy$new(alpha=alpha)
agent_UCB <- Agent$new(UCBitem, bandit_UCB, name='UCB item')

size_sim=100000
n_sim=6
simulator <- Simulator$new(list(agent_UCB), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,835)






bandit_TS <- OfflineReplayEvaluatorBandit$new(formula = reward ~ arm_item, data = df, randomize = FALSE)
TSitem <- ThompsonSamplingPolicy$new()
agent_TS <- Agent$new(TSitem, bandit_TS, name='TS item')

size_sim=100000
n_sim=6
simulator <- Simulator$new(list(agent_TS), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE, 
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1175)

