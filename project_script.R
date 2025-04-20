library(quantmod)
library(dplyr)
library(tidyr)
library(lubridate)
library(tidyverse)
library(ggplot2)
library(forecast)
library(rstan)
library(posterior)
library(tidybayes)

## the tech stacks we will be working with
tickers <- c("META", "AAPL", "AMZN", "NFLX", "GOOGL", "XLK")

## retrieving data from yahoo finance for this current decade
getSymbols(tickers, src = "yahoo", from = "2020-01-01", to = Sys.Date())

# extract close prices and work with them accordingly
prices <- do.call(merge, lapply(tickers, function(tk) Ad(get(tk))))
colnames(prices) <- tickers

# data wrangling, removing na's and converting it into the dataframe; obtaining
# log_return by using using lag function to obtain yesterday's price and 
# calculating the appropriate log ratio; dropped NAs instead of imputing.
returns_long <- prices %>%
  na.omit() %>%
  as.data.frame() %>%
  rownames_to_column(var = "date") %>%
  mutate(date = as.Date(date)) %>%
  pivot_longer(-date, names_to = "ticker", values_to = "price") %>%
  group_by(ticker) %>%
  arrange(date) %>%
  mutate(log_return = log(price / lag(price))) %>%
  filter(!is.na(log_return)) %>%
  ungroup()

# preview of the dataset
head(returns_long)

# some EDA

# plots the log_returns
ggplot(returns_long, aes(x = date, y = log_return)) +
  geom_line(color = "steelblue") +
  facet_wrap(~ ticker, scales = "free_y") +
  labs(title = "Daily Log Returns of TECH Stocks",
       x = "Date", y = "Log Return") +
  theme_minimal()

# plots histograms of the log_returns which will help us build priors
ggplot(returns_long, aes(x = log_return)) +
  geom_histogram(bins = 50, fill = "purple", alpha = 0.6) +
  facet_wrap(~ ticker, scales = "free") +
  labs(title = "Distribution of Log Returns", x = "Log Return") +
  theme_minimal()

# plotting ACF for each ticker which will help us decide the autocorrelation
# between our lag values
tickers <- unique(returns_long$ticker)

for (ticker in tickers) {
  returns_long %>%
    filter(ticker == ticker) %>%
    pull(log_return) %>%
    Acf(main = paste("ACF of", ticker, "Log Returns"))
}

# squared log returns plot which will help us detect volatility clustering
ggplot(returns_long, aes(x = date, y = log_return^2)) +
  geom_line(color = "darkred", linewidth = 0.3) +
  facet_wrap(~ ticker, scales = "free_y") +
  labs(
    title = "Squared Log Returns of TECH Stocks",
    x = "Date",
    y = expression(paste("(", log(R[t]), ")^2"))
  ) +
  theme_minimal()

# now we move onto the modelling with Stan but before that we will do some 
# engineering so we can feed the data into Stan as needed.

assets <- returns_long %>% 
  distinct(ticker) %>%
  mutate(asset_id = row_number())

returns_df <- returns_long %>% 
  left_join(assets, by = "ticker")

return_matrix <- returns_df %>%
  select(date, asset_id, log_return) %>%
  pivot_wider(names_from = asset_id, values_from = log_return) %>%
  arrange(date) %>%
  select(-date) %>%
  as.matrix()

sector_vector <- rep(1, length(unique(returns_df$asset_id)))

## now feeding all of the above into Stan
stan_data <- list(
  T = nrow(return_matrix),
  N = ncol(return_matrix),
  S = 1,
  sector_id = sector_vector,
  r = return_matrix
)

fit <- stan(
  file = "C:\\Users\\shahb\\Downloads\\project_stan.stan",
  data = stan_data,
  chains = 3, iter = 2000, warmup = 1000,
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  cores = 3
)

#get posterior samples
posterior_results <- extract(fit)

# Trace plots
traceplot(fit)

# Pairwise correlation plot
pairs(fit)

# R-hat and ESS
summary(fit)$summary[, c("mean", "sd", "rhat", "n_eff")]

# Posterior means
mu_a0_mean <- colMeans(posterior_results$mu_a0)
mu_a1_mean <- colMeans(posterior_results$mu_a1)
mu_b_mean <- colMeans(posterior_results$mu_b)
mu_phi_mean <- colMeans(posterior_results$mu_phi)

#volatility diagnostics
h_samples <- posterior_results$h
vol_mean <- apply(h_samples, c(2, 3), function(x) mean(sqrt(x)))
vol_ci_lower <- apply(h_samples, c(2, 3), function(x) quantile(sqrt(x), 0.025))
vol_ci_upper <- apply(h_samples, c(2, 3), function(x) quantile(sqrt(x), 0.975))

#volatility plot
matplot(vol_mean, type = "l", lty = 1, col = rainbow(6),
        ylab = "Posterior Mean Volatility", xlab = "Time (e.g. weeks)")
legend("topright", legend = c("META", "AAPL", "AMZN", "NFLX", "GOOGL", "XLK"),
       col = rainbow(6), lty = 1, cex = 0.8)
