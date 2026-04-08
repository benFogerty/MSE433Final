## Appendix A. Generator-Level Unit Commitment Formulation

This appendix presents the exact mathematical structure of the final generator-level unit commitment (UC) model used in the project. The model is solved on a rolling horizon and co-optimizes generator dispatch, thermal commitment decisions, renewable allocation, and battery storage operation.

### A.1 Sets and Indices

- \( t \in \mathcal{T} \): hourly time periods in the rolling horizon
- \( g \in \mathcal{G} \): all dispatchable generators
- \( \mathcal{G}^{bin} \subseteq \mathcal{G} \): binary commitment generators (`GAS`, `BIOFUEL`)
- \( \mathcal{G}^{cont} \subseteq \mathcal{G} \): continuous dispatch generators (`HYDRO`, `NUCLEAR`)

### A.2 Parameters

For each hour \( t \):

- \( D_t \): Ontario demand (MW)
- \( R_t \): renewable energy input available to the UC (MW)
- \( S_t \): stress-hour cost adder used to value difficult hours more strongly
- \( \bar{P}^{stor} \): battery power limit (MW)
- \( \bar{E}^{stor} \): battery energy limit (MWh)

For each generator \( g \):

- \( \bar{P}_{g,t} \): maximum output (MW)
- \( \underline{P}_{g,t} \): minimum output (MW)
- \( c_g^{var} \): variable production cost ($/MWh)
- \( c_g^{su} \): startup cost ($)
- \( c_g^{sd} \): shutdown cost ($)
- \( RU_g \): ramp-up limit (MW/h)
- \( RD_g \): ramp-down limit (MW/h)
- \( MU_g \): minimum up-time (hours)
- \( MD_g \): minimum down-time (hours)

Battery parameters:

- \( \eta^{ch} \): charging efficiency
- \( \eta^{dis} \): discharging efficiency
- \( E^{init} \): initial battery state of charge
- \( E^{final} \): required final state of charge

Penalty parameters:

- \( c^{curt} \): renewable curtailment penalty
- \( c^{shed} \): load-shedding penalty
- \( c^{over} \): overgeneration penalty
- \( c^{thr} \): small battery-throughput penalty

Initial UC conditions from the prior hour:

- \( u_{g,0} \): prior on/off state
- \( p_{g,0} \): prior generator output
- prior on/off durations for minimum up/down carryover

### A.3 Decision Variables

For each hour \( t \):

Renewable and system balancing:

- \( r_t^{dir} \ge 0 \): renewable power used directly to serve load
- \( r_t^{curt} \ge 0 \): curtailed renewable power
- \( l_t^{shed} \ge 0 \): load shed
- \( o_t^{over} \ge 0 \): overgeneration slack

Battery:

- \( ch_t \ge 0 \): battery charging power
- \( dis_t \ge 0 \): battery discharging power
- \( e_t \ge 0 \): battery state of charge
- \( m_t \in \{0,1\} \): battery charge/discharge mode indicator

Generators:

- \( p_{g,t} \ge 0 \): generator dispatch
- \( u_{g,t} \in \{0,1\} \): commitment state for \( g \in \mathcal{G}^{bin} \)
- \( su_{g,t} \in \{0,1\} \): startup indicator for \( g \in \mathcal{G}^{bin} \)
- \( sd_{g,t} \in \{0,1\} \): shutdown indicator for \( g \in \mathcal{G}^{bin} \)

### A.4 Objective Function

The model minimizes total operating cost over the rolling horizon:

```text
minimize

sum over t in T of:
  [ sum over g in G of (c_var_g + S_t) * p_g,t ]
  + [ sum over g in G_bin of (c_su_g * su_g,t + c_sd_g * sd_g,t) ]
  + c_curt * r_curt_t
  + c_shed * l_shed_t
  + c_over * o_over_t
  + c_thr * ch_t
```

Equivalently, in compact notation:

\( \min \sum_{t \in \mathcal{T}} \left[ \sum_{g \in \mathcal{G}} (c_g^{var} + S_t)p_{g,t} + \sum_{g \in \mathcal{G}^{bin}} (c_g^{su} su_{g,t} + c_g^{sd} sd_{g,t}) + c^{curt} r_t^{curt} + c^{shed} l_t^{shed} + c^{over} o_t^{over} + c^{thr} ch_t \right] \)

Interpretation:

- generator dispatch incurs variable production cost
- binary thermal units also incur startup and shutdown costs
- curtailment, load shedding, and overgeneration are penalized
- a small throughput penalty discourages unnecessary battery cycling
- the stress-hour adder increases the model’s incentive to preserve flexibility in historically difficult hours

### A.5 Constraints

#### 1. Renewable allocation

All renewable input must be allocated each hour:

\[
r_t^{dir} + r_t^{curt} + ch_t = R_t \qquad \forall t
\]

This means renewable energy can be used directly, stored in the battery, or curtailed.

#### 2. System power balance

\[
r_t^{dir} + dis_t + \sum_{g \in \mathcal{G}} p_{g,t} + l_t^{shed} - o_t^{over} = D_t
\qquad \forall t
\]

Demand must be met by renewables, battery discharge, and dispatchable generation, with slack variables for infeasibility or excess supply.

#### 3. Battery charge/discharge exclusivity

\[
ch_t \le \bar{P}^{stor} m_t \qquad \forall t
\]

\[
dis_t \le \bar{P}^{stor}(1 - m_t) \qquad \forall t
\]

The battery cannot charge and discharge at the same hour.

#### 4. Battery state of charge dynamics

\[
e_t = e_{t-1} + \eta^{ch} ch_t - \frac{1}{\eta^{dis}} dis_t
\qquad \forall t
\]

For the first modeled hour:

\[
e_1 = E^{init} + \eta^{ch} ch_1 - \frac{1}{\eta^{dis}} dis_1
\]

Battery bounds:

\[
0 \le e_t \le \bar{E}^{stor} \qquad \forall t
\]

Terminal condition:

\[
e_{|\mathcal{T}|} = E^{final}
\]

This prevents the model from unrealistically emptying the battery at the end of each block.

#### 5. Generator output limits for binary units

For \( g \in \mathcal{G}^{bin} \):

\[
p_{g,t} \le \bar{P}_{g,t} u_{g,t} \qquad \forall t
\]

\[
p_{g,t} \ge \underline{P}_{g,t} u_{g,t} \qquad \forall t
\]

If a binary generator is off, it cannot produce. If it is on, it must remain above minimum stable generation.

#### 6. Commitment state transition

For \( g \in \mathcal{G}^{bin} \):

\[
u_{g,t} - u_{g,t-1} = su_{g,t} - sd_{g,t} \qquad \forall t
\]

For the first hour, \(u_{g,0}\) comes from the prior observed state.

Also:

\[
su_{g,t} + sd_{g,t} \le 1 \qquad \forall t
\]

A unit cannot both start up and shut down in the same hour.

#### 7. Ramp constraints

For binary units:

\[
p_{g,t} - p_{g,t-1} \le RU_g + \bar{P}_{g,t} su_{g,t}
\qquad \forall g \in \mathcal{G}^{bin}, t
\]

\[
p_{g,t-1} - p_{g,t} \le RD_g + \bar{P}_{g,t-1} sd_{g,t}
\qquad \forall g \in \mathcal{G}^{bin}, t
\]

For continuous units:

\[
p_{g,t} - p_{g,t-1} \le RU_g
\qquad \forall g \in \mathcal{G}^{cont}, t
\]

\[
p_{g,t-1} - p_{g,t} \le RD_g
\qquad \forall g \in \mathcal{G}^{cont}, t
\]

Special continuous-unit ramp assumptions used in the final model:

- `NUCLEAR`: ramp rate set to 1% of rated power per minute, or 60% of rated power per hour
- `HYDRO`: allowed to move from 0 to \(P_{max}\) within the hour

#### 8. Minimum up-time constraints

For each binary unit:

\[
\sum_{\tau = t-MU_g+1}^{t} su_{g,\tau} \le u_{g,t}
\qquad \forall g \in \mathcal{G}^{bin}, t
\]

This ensures that if a unit has started recently, it must remain on.

#### 9. Minimum down-time constraints

\[
\sum_{\tau = t-MD_g+1}^{t} sd_{g,\tau} \le 1 - u_{g,t}
\qquad \forall g \in \mathcal{G}^{bin}, t
\]

This ensures that if a unit has shut down recently, it must remain off.

Initial carryover for minimum up/down is approximated using recent observed historical operating durations before the first modeled hour.

### A.6 Policy Interpretation in the Project

The same UC structure is used for multiple policy comparisons, with only the renewable-input definition changing:

- `historical_actual`: observed storage and observed system operation baseline
- `no_storage_uc`: UC with battery disabled
- `ieso_forecast_uc`: UC using raw IESO renewable forecast
- `forecast_informed_uc`: UC using the project’s best ML renewable forecast
- `perfect_foresight_uc`: UC using actual renewable output

The main reported result uses `forecast_informed_uc` compared against `historical_actual`, while the no-storage and perfect-foresight cases are used for decomposition and validation.

### A.7 Modeling Notes

Important assumptions in the final formulation:

- the model is system-level, not nodal
- no transmission constraints are included
- no reserve constraints are included
- no voltage or network feasibility constraints are included
- storage is charged from modeled renewable energy
- the rolling horizon is solved in sequential blocks, with generator commitment state carried forward between blocks
- the final main case uses a 168-hour rolling horizon

### A.8 Practical Meaning

This formulation is more physically realistic than a simple storage-only dispatch model because it embeds battery decisions inside a full UC environment with:

- thermal commitment logic
- startup and shutdown costs
- minimum stable generation
- ramping limits
- minimum up/down requirements
- explicit state-of-charge tracking
- hourly system balance

As a result, the storage results reported in the project are more operationally defensible and more closely connected to how storage would actually interact with the Ontario generation fleet.
