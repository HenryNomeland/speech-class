library(tidyverse)
library(hrbrthemes)
library(ggtext)
setwd(file.path(getwd(), "Documents", "senior-thesis", "speech-class"))

df <- read.csv(file.path(".", "data", "model-results.csv"))

df |> 
  filter(data %in% c("gmu uk", "timit nyc")) |> 
  filter(model %in% c("rforest", "ridgereg", "svc")) |> 
  ggplot(aes(y = standardization, x = accuracy, fill = model)) + 
  geom_bar(position = "dodge", stat = "identity") +
  facet_grid(data~.,scales="free_y") +
  coord_cartesian(xlim = c(0.5, 1)) +
  theme_minimal() +
  scale_fill_discrete(labels = c("Random Forest", "Ridge Reg.", "SVC")) +
  theme(
    plot.title = element_text(family = "Times", size = 14, face = "bold"),
    plot.subtitle = element_text(family = "Times", face = "italic", size = 12),
    strip.text.y = element_text(family = "Times", size = 12, color="black"),
    text = element_text(family = "Times", color="black", size = 12),
    axis.title.x = element_text(margin = margin(t = 2), color="black"),
    legend.position = "bottom"
  ) +
  labs(x="", y="",
       title="Figure 1: Comparison of \nStandardization Methods",
       subtitle="z-score normalization vs min-max scaling",
       caption="", fill="", strip="") +
  scale_fill_grey()

ggsave(file.path(".", "plots", "standardization-comparison.pdf"), 
       width = 14, height = 14, units = "cm")

df |> 
  filter(model %in% c("rforest", "svc", "ridgereg")) |> 
  filter(data %in% c("timit nyc", "timit newengland", "timit northern",
                     "timit northmidland", "timit southmidland", 
                     "timit southern", "timit western")) |> 
  ggplot(aes(y = data, x = accuracy, fill = model)) + 
  geom_bar(position = "dodge", stat = "identity") +
  coord_cartesian(xlim = c(0.5, 1)) 

ggsave(file.path(".", "plots", "timit-comparison.png"), 
       width = 20, height = 22, units = "cm")

