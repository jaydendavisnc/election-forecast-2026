# Model notes

## Goal

Forecast the 2026 U.S. House with a district-level model whose national polling layer is strong enough to support a real daily odds-over-time view and transparent enough to audit when the line moves.

## National layer

### Generic-ballot archive

The primary national signal is a dated generic-ballot archive rather than a single current average. Each row can contain:

- pollster
- field dates and/or publication date
- sample size
- LV / RV / Adults population tag
- Democratic and Republican toplines
- exact-date versus fallback-date flags
- exact-sample versus inferred-sample flags
- partisan/internal markers

### State-space filter

The model treats the national House margin as a latent state:

- `x_t = x_(t-1) + eta_t`
- `y_i,t = x_t + epsilon_i,t`

Where:

- `x_t` is the latent national House environment on day `t`
- `eta_t` is daily drift
- `epsilon_i,t` is poll-specific observation noise

This is better than a moving average because it separates real movement from noise and produces a coherent daily uncertainty estimate.

### Observation error

Observation variance is not just raw sampling error. It is widened when:

- the sample is Adults rather than RV / LV
- the row lacks exact field dates
- the row lacks sample size
- the population tag is missing
- the poll is flagged as partisan or internal

This lets the archive stay broad without pretending all rows deserve equal credibility.

## Trump approval cross-check

Trump approval is now in the model explicitly, but it is intentionally weak.

### Why include it

Presidential approval contains real information about the governing party’s political climate. Ignoring it entirely throws away useful context.

### Why keep it weak

Approval and the generic ballot are highly collinear. A strong approval term would effectively count the same national mood twice. So the model converts Trump net approval into an approval-implied House margin with a shallow slope, then combines that with the filtered generic ballot using a large prior SD.

In practice, that means:

- the generic ballot stays the primary national signal
- approval mostly acts as a stabilizing cross-check
- the approval adjustment is visible and auditable in the outputs

## District layer

The district forecast is built from:

1. 2024 House baseline
2. district presidential lean when available
3. persistence / carryover assumptions
4. open-seat handling
5. expert ratings when available
6. district polling when available
7. finance when available

The national layer moves every seat, but each district has its own intercept and uncertainty.

## Chamber simulation

For each simulation draw:

- sample a national House environment
- add correlated state error
- add district-specific uncertainty
- count seats won by each party

This yields:

- GOP hold probability
- expected seats
- interval estimates
- full seat distribution

## Trend history and the endpoint-spike fix

A major audit problem in the previous build was a model-definition break at the right edge of the chart:

- the historical line used stripped-down history inputs
- the current point used the full current district model

That created an artificial endpoint jump that looked like a real forecast spike.

This build fixes that by holding the **current campaign structure fixed** through the reconstructed daily curve:

- current open-seat structure
- current ratings
- current finance inputs
- current approval cross-check

Saved runs are still preserved separately. So there are now two distinct ideas in the project:

- **saved run history** = what the model literally output on past run dates
- **daily comparable curve** = how the forecast changes over time when the model definition is held constant

That separation is important for honest UI design.

## Approval chart

The UI now includes a separate Trump-approval chart with:

- approve line
- disapprove line
- 90% filter bands
- raw poll markers

That chart is an audit surface for the national context. It is not a second seat model.

## Remaining scientific gaps

The biggest remaining upgrade is formal calibration across earlier cycles. The main items still worth estimating empirically are:

- pollster house effects
- horizon-specific national error
- process noise over the cycle
- calibration of district and chamber probabilities
- the approval-to-House link strength

Those are the remaining gaps between this project and a fully production-grade public forecaster.
