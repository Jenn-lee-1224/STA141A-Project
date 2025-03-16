---
title: "ECN 141A Project"
author: "Hoang Trinh Nguyen"
date: "2025-03-06"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(include = FALSE)
```

```{r include=FALSE}
install.packages("ROCR", repos = "[https://cloud.r-project.org](https://cloud.r-project.org)")
```

```{r include=FALSE}
library(tidyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(knitr)
library(readr)
library(caret) 
library(xgboost)
library(pROC)
library(ROCR)
```


```{r include=FALSE}
session <- list()
for (i in 1:18) {
  session[[i]] <- readRDS(paste0("C:/Users/trinh/OneDrive/Documents/STA 141A/session", i, ".rds"))
}
```
PREDICTING DECISION OUTCOMES FROM NEURAL DATA

ABSTRACT:
This project explores the neural activity of mice performing a decision-making task involving visual stimuli with varying contrast levels. The data, collected from 18 sessions across four mice, includes spike trains from neurons in the visual cortex recorded during the onset of stimuli and feedback provided for each trial. The project's goal is to predict the feedback of each trial by conducting regression models with neural data and stimuli characteristics. 

1 - INTRODUCTION:
The study of Steinmetz et al. (2019) examines the neural activity in the visual cortex of mice as they engaged in a task requiring them to make decisions based on visual stimuli. The stimuli consisted of varying contrast levels presented on two screens, with the animals required to choose one side based on the relative contrast between the left and right stimuli. In this project, we analyze data from 18 sessions of experiments on four mice. Each session contains records of neural spike trains and experimental parameters, including the contrast levels of the visual stimuli, the feedback type, and the number of neurons. This project conducts models to predict the feedback of each trial based on the neural data and the contrast conditions. 

2 - EXPLORATORY DATA ANALYSIS:
  a. Data Structure:
```{r include=FALSE}
#Checking the structure of a session (session 1)
names(session[[1]])
dim(session[[1]]$spks[[1]]) 
length(session[[1]]$brain_area)
session[[1]]$spks[[1]][6,3] 
session[[1]]$brain_area[6]

#Getting unique mouse names and brain areas
unique_mice <- unique(sapply(session, function(x) x$mouse_name[1]))
print(unique_mice)
unique_brain_areas <- unique(unlist(lapply(session, function(x) unique(x$brain_area))))
print(unique_brain_areas)
```
The dataset contains recordings of neural activity, where each trial includes left and right stimulus contrasts, a binary feedback outcome, and detailed spike data from 734 brain regions across 40 time points. Metadata such as mouse name, experimental date, and brain area labels provide context for each recording. Data from four mice—Cori, Forssmann, Hench, and Lederberg—is included, with neural activity observed across a set of brain areas. 
Five variables are available for each trial, namely 
- `feedback_type`: type of the feedback, 1 for success and -1 for failure
- `contrast_left`: contrast of the left stimulus
- `contrast_right`: contrast of the right stimulus
- `time`: centers of the time bins for `spks`  
- `spks`: numbers of spikes of neurons in the visual cortex in time bins defined in `time`
- `brain_area`: area of the brain where each neuron lives

  b. Data Summary: 
```{r include=FALSE}
session_summary <- function(session_data) {
  num_trials <- length(session_data$feedback_type)
  num_neurons <- length(unique(session_data$brain_area))
  num_brain_areas <- length(unique(session_data$brain_area))

  feedback_counts <- table(session_data$feedback_type)
  contrast_left_summary <- summary(session_data$contrast_left)
  contrast_right_summary <- summary(session_data$contrast_right)

  return(list(
    num_trials = num_trials,
    num_neurons = num_neurons,
    num_brain_areas = num_brain_areas,
    feedback_counts = feedback_counts,
    contrast_left_summary = contrast_left_summary,
    contrast_right_summary = contrast_right_summary
  ))
}

session_summaries <- lapply(session, session_summary)
session_summaries[[1]]

n.session=length(session)
n_success = 0
n_trial = 0
for(i in 1:n.session){
    tmp = session[[i]];
    n_trial = n_trial + length(tmp$feedback_type);
    n_success = n_success + sum(tmp$feedback_type == 1);
}
n_success/n_trial

```
  
```{r echo=FALSE}
meta <- tibble(
  session=rep('name',n.session),
  mouse_name = rep('name',n.session),
  date_exp =rep('dt',n.session),
  n_brain_area = rep(0,n.session),
  n_neurons = rep(0,n.session),
  n_trials = rep(0,n.session),
  success_rate = rep(0,n.session),
  contrast_left = rep(0,n.session),
  contrast_right = rep(0,n.session),
  ratio_right_contrast= rep(0,n.session),
  ratio_left_contrast= rep(0,n.session)
)
for(i in 1:n.session){
  tmp = session[[i]];
  meta[, 1]= 1:18;
  meta[i,2]=tmp$mouse_name;
  meta[i,3]=tmp$date_exp;
  meta[i,4]=length(unique(tmp$brain_area)); 
  meta[i,5]=dim(tmp$spks[[1]])[1];
  meta[i,6]=length(tmp$feedback_type); 
  meta[i,7]=mean(tmp$feedback_type+1)/2; 
  meta[i,8]=mean(tmp$contrast_left);
  meta[i,9]=mean(tmp$contrast_right);
  meta[i,10]=mean(tmp$contrast_left)/mean(tmp$contrast_right);
  meta[i,11]=mean(tmp$contrast_right)/(mean(tmp$contrast_left))
}
colnames(meta) <- c("Session", "Mouse Name", "Experiment Date", "Number of Brain Areas", "Number of Neurons", "Number of Trials", "Success Rate", "Avg. Left Contrast Level", "Avg. Right Contrast Level", "Ratio Left Vs Right", "Ratio Right Vs Left")
kable(meta, format = "html", 
      table.attr = "class='table table-striped'", 
      digits = 2,
      caption = "Summary of Experimental Data by Session", 
      footnote = "Note: Success rate is calculated as the proportion of positive feedback trials.") 
```
The table above summarizes the experiment report across 18 sessions of the four mice named Cori, Forssmann, Hench, and Lederberg alonging with the date of experiment. Afterward is the number of brain areas, neurons, and trials in each session. Whereas, the number of neurons is the length of the brain_area variable, which is in the range between 474 and 1769. The table also includes the success rate across sessions, which is in the range between 0.61 and 0.83. Following that is the average left and right contrast levels. The last columns are the ratios of contrast levels in the left versus right and right versus left. Note that the date of variables contrast_left and contrast_right represents four scenarios as follows:
When left contrast > right contrast, success (1) if turning the wheel to the right and failure (-1) otherwise.
When right contrast > left contrast, success (1) if turning the wheel to the left and failure (-1) otherwise.
When both left and right contrasts are zero, success (1) if holding the wheel still and failure (-1) otherwise.
When left and right contrasts are equal but non-zero, left or right will be randomly chosen (50%) as the correct choice.

  c. Data Visualization:
Alonging with the variations in number of trials across sessions is the variations in neural activities. Even though there is a similarity among the mice's dominant pattern of neural activity, their firing rates and feedback are different. Yet, their success rates are similar and the overall success rate is 71%. On the other hand, Lederberg has the least variation in firing rate so it's firing rate is likely to be the most consistent, whereas Cori has the highest mean of firing rates.   
```{r include=FALSE}
calculate_mean_firing_rates <- function(session_data) {
  num_trials <- length(session_data$spks)
  num_neurons <- length(session_data$brain_area)
  time_bins <- session_data$time
  firing_rates <- matrix(NA, nrow = num_neurons, ncol = num_trials)
  for (trial_idx in 1:num_trials) {
  trial_spikes <- session_data$spks[[trial_idx]]
  firing_rates[, trial_idx] <- rowMeans(trial_spikes)
  }
  return(firing_rates)
}

all_trials_data <- data.frame()
for (i in 1:18) {
  session_data <- data.frame(
    contrast_left = session[[i]]$contrast_left,
    contrast_right = session[[i]]$contrast_right,
    feedback = factor(session[[i]]$feedback_type),
    mouse_name = session[[i]]$mouse_name[1]
  )
  all_trials_data <- rbind(all_trials_data, session_data)
}

all_sessions_data <- data.frame()
for (i in 1:18) {
  session_data <- session[[i]] 
  firing_rates_session <- calculate_mean_firing_rates(session_data) 
  session_data <- data.frame(
    contrast_left = session[[i]]$contrast_left,
    contrast_right = session[[i]]$contrast_right,
    feedback = factor(session[[i]]$feedback_type),
    firing_rate = c(firing_rates_session),
    brain_area = rep(session[[i]]$brain_area, ncol(firing_rates_session)),
    mouse_name = session[[i]]$mouse_name[1],
    session_num = i
  )
  all_sessions_data <- rbind(all_sessions_data, session_data)
}

```
    (i) Neural activity across sessions:
      **The changes in numbers of trials and neurons across sessions:
```{r echo=FALSE}
num_trials_per_session <- sapply(session_summaries, function(x) x$num_trials)
plot(1:18, num_trials_per_session, type = "b", xlab = "Session", ylab = "Number of Trials", main = "Trials per Session")

num_neurons_per_session <- sapply(session_summaries, function(x) x$num_neurons)
plot(1:18, num_neurons_per_session, type = "b", xlab = "Session", ylab = "Number of Neurons", main = "Neurons per Session")
```

      **Neural activities in session 1 as an example:
```{r echo=FALSE}
    #   Example for the first session
    session_spikes <- session[[1]]$spks

    #   Calculate mean firing rate for each neuron in each trial
    mean_firing_rates_across_trials <- sapply(session_spikes, rowMeans)

    #   Average Firing Rate Across All Neurons per Trial
    average_firing_rate_per_trial <- colMeans(mean_firing_rates_across_trials)

    #   Plot the average firing rate
    plot(
        average_firing_rate_per_trial,
        type = "l",
        main = "Average Firing Rate Across Trials - Session 1",
        xlab = "Trial",
        ylab = "Average Firing Rate"
    )
```
      
      ***Neural activity across sessions grouped by mouse:
```{r echo=FALSE}
# Boxplot of firing rate by brain area, grouped by mouse
ggplot(all_sessions_data, aes(x = brain_area, y = firing_rate, fill = mouse_name)) +
  geom_boxplot() +
  facet_wrap(~mouse_name) + #facets allow you to see each mouse as its own plot.
  ggtitle("Firing Rate by Brain Area, Grouped by Mouse")

# Violin plot of firing rate by mouse
ggplot(all_sessions_data, aes(x = mouse_name, y = firing_rate, fill = mouse_name)) +
  geom_violin() +
  ggtitle("Firing Rate Distribution by Mouse")

# Density plot of firing rate by mouse
ggplot(all_sessions_data, aes(x = firing_rate, fill = mouse_name)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~mouse_name) +
  ggtitle("Firing Rate Density by Mouse")
```
```{r echo=FALSE}
#   Mean Firing Rates Across Mice (Boxplots)
    mouse_mean_firing_rates <- lapply(session, function(s) {
        data.frame(
            mouse = s$mouse_name[1],
            mean_firing_rate = mean(unlist(s$spks))
        )
    })
    mouse_mean_firing_rates_df <- do.call(rbind, mouse_mean_firing_rates)

    #   Boxplot using ggplot2
    ggplot(mouse_mean_firing_rates_df, aes(x = mouse, y = mean_firing_rate)) +
        geom_boxplot() +
        labs(
            title = "Mean Firing Rates Across Mice",
            x = "Mouse",
            y = "Mean Firing Rate"
        ) +
        theme_minimal()
```

      **Stimuli and feedback across sessions:
```{r echo=FALSE}
# "Stimulus" Category
all_sessions_data$stimulus <- ifelse(
    all_sessions_data$contrast_left > 0 | all_sessions_data$contrast_right > 0,
    "Stimulus Present",
    "No Stimulus"
)

# Histogram of Stimulus Presentation
ggplot(all_sessions_data, aes(x = factor(session_num), fill = stimulus)) +
  geom_bar() +
  labs(
    title = "Stimulus Presentation Across Sessions",
    x = "Session Number",
    y = "Count",
    fill = "Stimulus"
  ) +
  theme_minimal()

# Histogram of Feedback Type
ggplot(all_sessions_data, aes(x = factor(session_num), fill = feedback)) +
  geom_bar() +
  labs(
    title = "Feedback Type Across Sessions",
    x = "Session Number",
    y = "Count",
    fill = "Feedback"
  ) +
  theme_minimal()
```
```{r echo=FALSE}

# Visualize Stimuli and Feedback, grouped by mouse
ggplot(all_trials_data, aes(x = contrast_left, y = contrast_right, color = feedback)) +
  geom_point() +
  facet_wrap(~mouse_name) +
  ggtitle("Stimuli and Feedback grouped by Mouse") +
  xlab("Left Contrast") +
  ylab("Right Contrast")
```

      (ii) Homogeneity and heterogeneity across sessions and mice. 
```{r echo=FALSE}
# Calculate mean firing rate for each trial in each session
session_trial_firing <- lapply(session, function(s) {
  trial_firing <- sapply(s$spks, mean)
  data.frame(
    mouse = s$mouse_name[1],
    session = s$date_exp[1], # Or use a session number
    trial_firing = trial_firing
  )
})
session_trial_firing_df <- do.call(rbind, session_trial_firing)

# Calculate mean firing rate variability (standard deviation) per session/mouse
session_variability <- session_trial_firing_df %>%
  group_by(mouse, session) %>%
  summarize(firing_rate_sd = sd(trial_firing))

# Boxplot of firing rate variability by mouse
ggplot(session_variability, aes(x = mouse, y = firing_rate_sd)) +
  geom_boxplot() +
  labs(
    title = "Firing Rate Variability Across Mice",
    x = "Mouse",
    y = "Standard Deviation of Trial Firing Rate"
  ) +
  theme_minimal()

# Boxplot of firing rate variability by session
ggplot(session_variability, aes(x = session, y = firing_rate_sd)) +
  geom_boxplot() +
  labs(
    title = "Firing Rate Variability Across Sessions",
    x = "Session",
    y = "Standard Deviation of Trial Firing Rate"
  ) +
  theme_minimal()
```

```{r echo=FALSE}
# Function to perform PCA on a session and add mouse name
perform_pca_with_mouse <- function(session_data) {
  firing_rates_session <- calculate_mean_firing_rates(session_data)
  constant_neurons <- apply(firing_rates_session, 1, function(x) {
    var(x) == 0
  })
  firing_rates_session <- firing_rates_session[!constant_neurons, ]
  firing_rates_transposed <- t(firing_rates_session)

  pca_result <- prcomp(firing_rates_transposed, scale. = TRUE)
  pcs <- pca_result$x
  variance_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)

  mouse_name <- session_data$mouse_name[1]
  pca_df <- data.frame(pcs, mouse_name = mouse_name)
  variance_df <- data.frame(variance_explained = variance_explained, PC = paste("PC", 1:length(variance_explained)), mouse_name = mouse_name)

  return(list(pca_df = pca_df, variance_df = variance_df))
}

# Apply PCA to all sessions and combine results
all_pca_data <- data.frame()
all_variance_data <- data.frame()
max_pcs <- 0

# First pass to find the maximum number of PCs
for (i in 1:18) {
  pca_results <- perform_pca_with_mouse(session[[i]])
  num_pcs <- ncol(pca_results$pca_df) - 1 # Subtract 1 for mouse_name column
  max_pcs <- max(max_pcs, num_pcs)
}

# Second pass to align columns and combine data
for (i in 1:18) {
  pca_results <- perform_pca_with_mouse(session[[i]])
  pca_df <- pca_results$pca_df
  num_pcs <- ncol(pca_df) - 1

  # Pad with NA if necessary
  if (num_pcs < max_pcs) {
    for (j in (num_pcs + 1):max_pcs) {
      pca_df[[paste0("PC", j)]] <- NA
    }
  }

  all_pca_data <- rbind(all_pca_data, pca_df)
  all_variance_data <- rbind(all_variance_data, pca_results$variance_df)
}

# Visualize PC1 vs PC2, grouped by mouse, all in one plot
ggplot(all_pca_data, aes(x = PC1, y = PC2, color = mouse_name)) +
  geom_point(alpha=0.5) +
  ggtitle("PC1 vs PC2, Grouped by Mouse")
```


3 - DATA INTEGRATION:
We use Benchmark Method 1 to simplify neural activity data. We address heterogeneity caused by varying neuron populations across session by averaging over their activities. In detail, we sum the spikes for each neuron within a 0.4-second trial window then average those sum across all neurons. Although this method is straightforward, it lose information of individual neuron contributions, leading to a potential limit of nuanced neural pattern detection.
```{r echo=TRUE}
for (i in 1:18) {
  n.trials <- length(session[[i]]$feedback_type)
  avg.spikes.all <- numeric(n.trials)

  for (j in 1:n.trials) {
    spks.trial <- session[[i]]$spks[[j]]
    total.spikes <- apply(spks.trial, 1, sum)
    avg.spikes.all[j] <- mean(total.spikes)
  }
  session[[i]]$avg.spks <- avg.spikes.all
}

trials.all <- tibble() 

for (i in 1:length(session)) {
  n.trials <- length(session[[i]]$feedback_type)
  tmp <- session[[i]]

  trials <- tibble(
    mouse_name = rep(tmp$mouse_name[1], n.trials),
    avg.spks = tmp$avg.spks,
    contrast_left = tmp$contrast_left,
    contrast_right = tmp$contrast_right,
    feedback_type = factor(tmp$feedback_type),
    session_ID = factor(rep(i, n.trials))
  )

  trials.all <- bind_rows(trials.all, trials)
}
summary(trials.all)
```


4 - PREDICTIVE MODEL:
```{r echo=FALSE}
fit.mod <- glm(feedback_type ~ contrast_left + contrast_right + avg.spks + factor(session_ID),
  data = trials.all,
  family = "binomial"
)

summary(fit.mod)
```
This logistic regression model reveals that average spike counts (avg.spks) are a strong positive predictor of feedback type, while higher right contrast (contrast_right) significantly decreases the likelihood of success. Notably, session-specific effects, captured by the factor(session_ID) coefficients, demonstrate substantial variability across sessions, indicating that factors beyond stimulus contrasts and spike rates influence feedback. Conversely, left contrast (contrast_left) shows no significant predictive power. The model, while explaining a portion of the data's variability, highlights the importance of neural activity and right contrast in predicting feedback, while emphasizing the need to account for session-specific differences.

```{r echo=FALSE}
set.seed(101)
sample_indices <- sample.int(n = nrow(trials.all), size = floor(0.8 * nrow(trials.all)), replace = FALSE)
train_data <- trials.all[sample_indices, ]
test_data <- trials.all[-sample_indices, ]

# Convert to vectors and remove names attribute
test_data$contrast_left <- unname(as.vector(as.numeric(test_data$contrast_left)))
test_data$contrast_right <- unname(as.vector(as.numeric(test_data$contrast_right)))
train_data$contrast_left <- unname(as.vector(as.numeric(train_data$contrast_left)))
train_data$contrast_right <- unname(as.vector(as.numeric(train_data$contrast_right)))

# Retrain the model
fit.mod <- glm(feedback_type ~ contrast_left + contrast_right + avg.spks + factor(session_ID),
    data = train_data,
    family = "binomial"
)

# Predict probabilities 
probabilities <- predict(fit.mod, newdata = test_data, type = "response")

# Confusion Matrix
predicted_classes <- factor(ifelse(probabilities > 0.5, 1, -1), levels = c("-1", "1"))
conf_matrix <- confusionMatrix(predicted_classes, test_data$feedback_type)
print("Confusion Matrix:")
print(conf_matrix)

# Misclassification Rate
misclassification_rate <- mean(predicted_classes != test_data$feedback_type)
print(paste("Misclassification Rate:", misclassification_rate))

# AUROC (using probabilities)
roc_obj <- roc(test_data$feedback_type, probabilities)
auc_value <- auc(roc_obj)
print(paste("AUROC:", auc_value))

# Plot the ROC curve
plot(roc_obj, main = paste("ROC Curve, AUROC =", round(auc_value, 4)))
```
The model's performance on the test data, as measured by the misclassification rate, is approximately 28.4%. This indicates that the model incorrectly predicts the outcome (feedback type) for about 28.4% of the trials in the test set.
The model demonstrates a bias towards predicting the majority class (1), resulting in high specificity (95.85%) but very low sensitivity (11.91%). This imbalance is reflected in the confusion matrix, where the model correctly identifies most negative cases but struggles significantly with positive cases. While the overall accuracy (71.58%) might seem reasonable, the low Kappa value (0.1009) indicates poor agreement beyond chance, and the highly significant McNemar's Test P-value (< 2e-16) highlights a systematic bias in the model's errors. Essentially, the model is much more reliable at identifying the negative class than the positive class, suggesting a need for adjustments to address the class imbalance and improve the prediction of positive instances.


5 - PREDICTION PERFORMANCE ON THE TEST SETS:

```{r include=FALSE}
test1 <- readRDS("C:/Users/trinh/AppData/Local/Temp/716f2e62-7801-47bc-a1d1-28f9efab6c20_test.zip.c20/test1.rds")
test2 <- readRDS("C:/Users/trinh/AppData/Local/Temp/b4a3d0e3-33d8-4e41-a273-1cdb7a23b976_test.zip.976/test2.rds")
```

```{r include=FALSE}
test1$feedback_type <- factor(test1$feedback_type, levels = c("-1", "1"))
test2$feedback_type <- factor(test2$feedback_type, levels = c("-1", "1"))
test1$session_ID <- factor(rep(1, length(test1$feedback_type)))
test2$session_ID <- factor(rep(18, length(test2$feedback_type)))

avg_spikes_test1 <- sapply(test1$spks, function(trial_spks) mean(apply(trial_spks, 1, sum)))
test1$avg.spks <- avg_spikes_test1

avg_spikes_test2 <- sapply(test2$spks, function(trial_spks) mean(apply(trial_spks, 1, sum)))
test2$avg.spks <- avg_spikes_test2
```

```{r include=FALSE}
test1$contrast_left <- as.numeric(test1$contrast_left)
test1$contrast_right <- as.numeric(test1$contrast_right)
test2$contrast_left <- as.numeric(test2$contrast_left)
test2$contrast_right <- as.numeric(test2$contrast_right)
test1$contrast_left <- unname(test1$contrast_left)
test1$contrast_right <- unname(test1$contrast_right)
test2$contrast_left <- unname(test2$contrast_left)
test2$contrast_right <- unname(test2$contrast_right)
```

```{r echo=FALSE}
probabilities_test1 <- predict(fit.mod, newdata = test1, type = "response")
predicted_classes_test1 <- factor(ifelse(probabilities_test1 > 0.5, 1, -1), levels = c("-1", "1"))

conf_matrix_test1 <- confusionMatrix(predicted_classes_test1, test1$feedback_type)
print("Test1 Confusion Matrix:")
print(conf_matrix_test1)

misclassification_rate_test1 <- mean(predicted_classes_test1 != test1$feedback_type)
print(paste("Test1 Misclassification Rate:", misclassification_rate_test1))

roc_obj_test1 <- roc(test1$feedback_type, probabilities_test1)
auc_value_test1 <- auc(roc_obj_test1)
print(paste("Test1 AUROC:", auc_value_test1))

plot(roc_obj_test1, main = paste("Test1 ROC Curve, AUROC =", round(auc_value_test1, 4)))

probabilities_test2 <- predict(fit.mod, newdata = test2, type = "response")
predicted_classes_test2 <- factor(ifelse(probabilities_test2 > 0.5, 1, -1), levels = c("-1", "1"))

conf_matrix_test2 <- confusionMatrix(predicted_classes_test2, test2$feedback_type)
print("Test2 Confusion Matrix:")
print(conf_matrix_test2)

misclassification_rate_test2 <- mean(predicted_classes_test2 != test2$feedback_type)
print(paste("Test2 Misclassification Rate:", misclassification_rate_test2))

roc_obj_test2 <- roc(test2$feedback_type, probabilities_test2)
auc_value_test2 <- auc(roc_obj_test2)
print(paste("Test2 AUROC:", auc_value_test2))

plot(roc_obj_test2, main = paste("Test2 ROC Curve, AUROC =", round(auc_value_test2, 4)))
```
The model is essentially predicting the majority class (1) for every single instance in test2. It shows a strong bias.  


6 - DISCUSSION:
The logistic regression model showed mixed results, revealing a significant bias towards predicting the majority class. While achieving a reasonable accuracy on the general test data, the model struggled to identify the minority class. It faces a critical failure in practical application due to systematic bias. Performance on a specific test dataset was particularly concerning, as the model consistently predicted only the majority class, demonstrating complete inability to identify the minority class. These findings suggests the need to address class imbalance and potential overfitting through techniques like resampling, alternative models, and robust feature engineering. 

# Reference {-}

Steinmetz, N.A., Zatka-Haas, P., Carandini, M. et al. Distributed coding of choice, action and engagement across the mouse brain. Nature 576, 266–273 (2019). https://doi.org/10.1038/s41586-019-1787-x

# Acknowledge {-}

The author would like to acknowledge the assistance of an AI language model from Google AI, which provided guidance during the data analysis and model evaluation phases of this project.

# Appendix: R Code

