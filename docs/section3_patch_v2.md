# Section III patch v2 (follows RQ2 design v2, decisions D21-D22) — DRAFT for the user

_Applies to `docs/section3_revised.md` (and later the paper). Three small text changes; nothing else in Section III
changes. Exact old/new text below._

## P13 (III-A, objective): client timeouts are overload signals (D21)
Old (definition of f_j, line 9):
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable),

New:
> $f_j \in \{0,1\}$ indicate that attempt $j$ returned HTTP 429 (Too Many Requests) or 503 (Service Unavailable), or
> exceeded the client timeout $T_{\text{out}}$ (in which case $L_j = T_{\text{out}}$),

## P14 (III-C, signals): controller target below the SLO (D22) and timeouts in F (D21)
Old (lines 42-43):
> - the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts returning 429 or 503), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{target}}}{L_{\text{target}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the attempts in interval $\tau$.

New:
> - the throttle rate $F_\tau = \frac{1}{n_c}\sum_{j\in\tau} f_j$ (fraction of attempts returning 429 or 503 or timing
>   out), and
> - the latency degradation $\Delta L_\tau = \dfrac{\hat{L}^{95}_\tau - L_{\text{ctrl}}}{L_{\text{ctrl}}}$, where
>   $\hat{L}^{95}_\tau$ is the p95 server latency of the successful attempts in interval $\tau$ ($\Delta L_\tau = 0$
>   if there is none) and $L_{\text{ctrl}} = \kappa\,L_{\text{target}}$, $0 < \kappa \le 1$, is the controller's
>   latency target. A margin $\kappa < 1$ keeps the probing of the additive increase from pushing the p95 above the SLO:
>   a controller that regulates the p95 exactly to $L_{\text{target}}$ violates the SLO for about 5% of attempts at its
>   set point, plus its overshoot. $\kappa = 1$ recovers the base algorithm. The tolerance $\epsilon$ is relative to
>   $L_{\text{ctrl}}$.

## P15 (III-C, retry policy and Algorithm 1 input, hyperparameter list)
- Retry policy (line 52): "...attempts that return 429, 503, or time out (client timeout $T_{\text{out}}$; the
  server may still be processing the abandoned attempt)..." (rest unchanged).
- Algorithm 1 input line: add `kappa` ("Input: ..., Target latency L_target, margin kappa").
- Hyperparameter list (line 87): add $\kappa$ and $T_{\text{out}}$.

## Note for Chapter IV (not Section III)
Evaluation setup: 120-s horizon runs, four scenarios, loaded cells fixed a priori, per-cell constraints (SLO <= 5%,
success >= 99% over calls requested before T - 20 s, throttle <= 10%), pre-registered comparisons, and the disclosure
that the scenarios were redesigned after v1's calibration showed no benefit of adaptation.
