//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

data {
  int<lower=1>   T;               // number of days
  int<lower=1>   N;               // number of stocks
  int<lower=1>   S;               // number of sectors
  int<lower=1,upper=S> sector_id[N];
  matrix[T, N]   r;               // log-returns
}

parameters {
  //── Sector-level hyperparameters ────────────────────────────────
  vector<lower=0>[S]     mu_a0;              // baseline variance
  vector<lower=0>[S]     tau_a0;             // spread around mu_a0

  vector[S]              mu_a1_logit_raw;    // unconstrained for logit(a1)
  vector<lower=0>[S]     tau_a1;             // spread around mu_a1

  vector[S]              mu_b_logit_raw;     // unconstrained for logit(b)
  vector<lower=0>[S]     tau_b;              // spread around mu_b

  vector[S]              mu_phi_logit_raw;   // unconstrained for logit(phi)
  vector<lower=0>[S]     tau_phi;            // spread around mu_phi

  vector[S]              mu_nu_log_raw;      // unconstrained for log(nu-2)
  vector<lower=0>[S]     tau_nu;             // spread around mu_nu

  //── Stock-level raw deviations ──────────────────────────────────
  vector[N]              a0_raw;
  vector[N]              a1_raw;
  vector[N]              b_raw;
  vector[N]              phi_raw;
  vector[N]              nu_raw;
}

transformed parameters {
  vector<lower=0,upper=1>[S] mu_a1 = inv_logit(mu_a1_logit_raw);
  vector<lower=0,upper=1>[S] mu_b  = inv_logit(mu_b_logit_raw);
  vector<lower=0,upper=1>[S] mu_phi = inv_logit(mu_phi_logit_raw);

  vector<lower=2>[N] nu;
  matrix[T, N] h;
  vector[N] a0;
  vector[N] a1;
  vector[N] b;
  vector[N] phi;

  for (i in 1:N) {
    int s = sector_id[i];

    // Hierarchical transforms
    a0[i]  = fmax(1e-6, mu_a0[s] + tau_a0[s] * a0_raw[i]);  // ensure a0 > 0
    a1[i]  = mu_a1[s] + tau_a1[s] * a1_raw[i];
    b[i]   = mu_b[s]  + tau_b[s]  * b_raw[i];
    phi[i] = mu_phi[s] + tau_phi[s] * phi_raw[i];
    nu[i]  = 2 + exp(mu_nu_log_raw[s] + tau_nu[s] * nu_raw[i]);  // nu >= 2

    // Soft + Hard Stationarity: reject if invalid
    if (a1[i] + b[i] >= 1)
      reject("a1 + b >= 1 for stock ", i, ": non-stationary");

    // Initialize h
    h[1, i] = a0[i] / (1 - a1[i] - b[i]);
    h[1, i] = fmax(h[1, i], 1e-6);

    // Recursive GARCH volatility
    for (t in 2:T) {
      h[t, i] = a0[i]
              + a1[i] * square(r[t-1, i])
              + b[i]  * h[t-1, i];

      // Safety clamp to ensure h[t, i] is positive
      h[t, i] = fmax(h[t, i], 1e-6);
    }
  }
}

model {
  //── Priors on sector-level hyperparameters ───────────────────────
  mu_a0               ~ normal(0.001, 0.001);
  tau_a0              ~ normal(0,     0.005)T[0,];

  mu_a1_logit_raw ~ normal(logit(0.05), 0.3);    // more conservative
  tau_a1 ~ normal(0, 0.05) T[0,];

  mu_b_logit_raw ~ normal(logit(0.85), 0.3);     // more conservative
  tau_b ~ normal(0, 0.025) T[0,];

  mu_phi_logit_raw    ~ normal(logit(0.45), 0.5);
  tau_phi             ~ normal(0,           0.05) T[0,];

  mu_nu_log_raw       ~ normal(log(18), 1);
  tau_nu              ~ normal(0,        1) T[0,];

  // Stock-level raw deviations
  a0_raw              ~ normal(0, 1);
  a1_raw              ~ normal(0, 1);
  b_raw               ~ normal(0, 1);
  phi_raw             ~ normal(0, 1);
  nu_raw              ~ normal(0, 1);

  //── Soft stationarity penalty: keep a1 + b ≈ 0.95 ────────────────
  for (i in 1:N)
    target += normal_lpdf(a1[i] + b[i] | 0.95, 0.02);

  //── Likelihood: AR(1)-GARCH with Student-t returns ──────────────
  for (i in 1:N) {
    r[1, i] ~ student_t(nu[i], 0,        sqrt(h[1, i]));
    for (t in 2:T)
      r[t, i] ~ student_t(nu[i], phi[i] * r[t-1, i], sqrt(h[t, i]));
  }
}

