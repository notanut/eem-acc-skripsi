"""Attempt outcome codes shared by the gateway, the controller and the metrics.

CLIENT_TIMEOUT is deliberately NOT an HTTP status (review L6, 2026-11-14): a real HTTP 504 from a proxy in the
LLM stage must not be confused with the client giving up after T_out.
"""

CLIENT_TIMEOUT = -1                              # the client abandoned the attempt after T_out
OVERLOAD = (429, 503, 504, CLIENT_TIMEOUT)       # f_j = 1 / counted in F (D21; 504 Gateway Timeout: D38)
RETRYABLE = (408, 429, 500, 502, 503, 504, CLIENT_TIMEOUT)   # retried by the gateway (D29, D33); others return at once
LATENCY_SAMPLE = (200, CLIENT_TIMEOUT)           # enter the interval p95; a timeout counts at T_out (D30)
