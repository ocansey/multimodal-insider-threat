# Methodology

What was decided, why, and which decisions made the numbers worse.

---

## 1. The unit of analysis

A **user-day**: one person, one working day, running 04:00 to 04:00.

The boundary is not midnight, and that is not a detail. Scenario 1 of the CERT
data is a person who starts staying late and eventually works through the small
hours. Cut at midnight and that single sitting becomes two unremarkable
half-days on either side of the most interesting moment, with the fourteen-hour
session represented nowhere. Every night-shift worker is mangled the same way.
Four in the morning is the quietest hour in the data.

Days with fewer than three events are dropped. An account that produced two
events is not evidence of anything, and scoring it spends analyst attention on
noise. The count of dropped days is reported.

---

## 2. Three modalities, and why they are three

**Behaviour** is a sequence of typed events. The vocabulary is fourteen tokens
including three special ones, which is deliberately small: the model should
learn that copying a file to removable media at 2am from someone else's machine
is unusual *from the combination* of token, hour and machine flags, not from a
token so specific it has memorised the answer.

Each event carries five covariates — hour, after-hours, weekend, own machine,
shared machine. Five, because every one has to be defensible as something a
SIEM would genuinely have, and because a wide per-event feature vector is where
leakage hides.

"Own machine" is learned, not assumed — the release has no asset register, so
it is the machine a person logs into most often **across the training window
only**. Deriving it from the whole period would let a desk move in month
fourteen influence a feature used to score month three.

**Content** is the text produced and consumed. Twelve documents per user-day,
sampled with a per-source budget so that email is not crowded out by web
traffic, encoded once by a frozen sentence transformer, then projected 384 → 64
by a PCA fitted on training rows only.

Documents are padded to a fixed count rather than averaged before the model
sees them. Averaging a day's documents destroys exactly the signal that
matters: one unusual message among forty routine ones survives as a token and
vanishes as a mean. The attention pooling inside the model does the
aggregating, and it can learn to ignore the routine forty.

**Context** is organisational and psychometric, resolved *as of the day being
scored*. The release ships one LDAP snapshot per month precisely because people
move; pinning everyone to January would make every promotion look like an
anomaly for the rest of the year. `pandas.merge_asof` does the temporal join.

Context enters through FiLM — it produces a per-channel scale and shift applied
to the behaviour representation — rather than being concatenated. The FiLM
layer is initialised centred on one, so at the start of training the model
ignores context and has to earn its use of it.

---

## 3. Training

Self-supervised. No label is ever seen by an optimiser.

**Masked event modelling.** Fifteen percent of real tokens are replaced with
`<mask>` and predicted. Padding is never masked — predicting padding is free
accuracy that makes the loss look better and teaches nothing. Every non-empty
row is guaranteed at least one target, otherwise short days contribute nothing
and the model never learns them.

**Cross-modal alignment.** InfoNCE between a day's behaviour embedding and its
own content embedding, against other days in the batch, both directions.

Early stopping watches the **calibration window**, which is later in time than
training and earlier than test. Watching a random slice of training days — the
usual default — measures memorisation of a period rather than transfer to the
next one.

The training window is capped at 40,000 user-days, sampled evenly across
people. The objective is self-supervised and the marginal value of the
hundred-thousandth ordinary Tuesday is close to zero while its wall-clock cost
is not. Every person keeps a quota, so nobody is dropped entirely.

---

## 4. Splits

Chronological, in three blocks: 50% train, 15% calibrate, 35% test, by day.

Known-malicious days are removed from the **training** window even though
training is unsupervised. A detector fitted on the campaign it is meant to find
is not a detector, and self-supervised does not mean immune — the model would
happily learn that exfiltration is a normal Tuesday.

The calibration window does three jobs: early stopping, the median/MAD scales
used to standardise score components, and fitting the component weights. The
test window does one job, once.

---

## 5. Scoring, and the experiment

Four components per user-day:

| Component | What it measures |
|---|---|
| `masked_behaviour` | How surprising the day's events are, given the rest of the day |
| `cross_modal_disagreement` | How far what the person did diverges from what they wrote |
| `peer_distance` | Mahalanobis distance from the reference population's centre |
| `self_distance` | Distance from the person's own recent history |

They are standardised by median and MAD — not mean and standard deviation,
because every one of them is heavy-tailed by construction and a handful of
extreme days would otherwise set the scale for everything else. The
standardisation is fitted on the calibration window, so the scale of a score
does not shift when the test period arrives, which is what a live deployment
would face.

Weights are chosen by random search over the simplex, scored on
precision-at-capacity on the calibration window. Two hundred draws, four
numbers, trivial to explain. If the calibration window contains no malicious
days at all the search returns equal weights and says so, because fitting to
zero positives is fitting to noise.

### The three reference populations

`peer_distance` is computed against one of three comparison classes, and the
difference between them is the point of the project:

- **global** — the whole organisation. What an unmodified density model does.
- **self** — the person's own trailing history, strictly backward-looking.
- **peer** — other people holding the same role, with fallback to functional
  unit and then business unit when a role has fewer than fifteen members.

All three are scored **from the same trained model**, so the comparison
isolates the choice of reference rather than confounding it with a different
fit.

References are fitted on **training-window embeddings only**. Fitting a role's
centroid on the same days you are about to score against it pulls every day
towards its own group mean, and the most anomalous days are the ones dragging
the mean towards themselves. That mistake improves results and is invisible in
the output.

Covariance estimates are shrunk towards a scaled identity. The embedding is
128-dimensional and a role may have forty people in it; an unshrunk covariance
there is singular, and inverting it produces spectacular distances driven
entirely by directions the data never populated.

---

## 6. Evaluation

Every measure is defined against a **daily analyst budget** rather than a
threshold, because a threshold is not something a SOC sets and a queue is.

- **Precision at capacity** — of the top *k* users on a given day, what
  fraction were genuinely malicious.
- **Campaign recall** — campaigns counted once, not user-days.
- **Time to detection** — days between the start of a campaign and its first
  alert. Zero is ideal; a system that catches everyone on their last day scores
  identically on every other metric and is worth nothing.
- **Alert concentration** — how many distinct people absorb the budget.
- **Over-alerting ratio by role** — share of alerts divided by share of
  user-days. One means proportionate; five means a role is being audited five
  times more heavily than everyone else for doing its job.

Confidence intervals on campaign recall are bootstrapped **over campaigns**,
not over days. With a handful of campaigns that interval is wide, and it should
be — the honest summary of "we caught five of six" is not a point estimate.
Papers on this dataset routinely quote three decimals on a quantity estimated
from fewer than ten independent events.

---

## 7. Ablations

**Modality:** `full`, `no_content`, `no_context`, `behaviour_only`. Removal is
implemented by zeroing the input rather than changing the architecture, so the
comparison measures information carried rather than model capacity.

**Normalisation:** `global`, `self`, `peer`, from one trained model.

**Text encoder:** a pretrained sentence transformer against a hashed bag of
character n-grams through a fixed random projection. The second is not a
strawman, it is the correct floor — any claim that the content modality helps
should be shown against it, because a gain a random projection also achieves is
a statement about document *volume*, not about language.

---

## 8. Decisions that made the numbers worse

Listed because a methods section containing only choices that helped is not a
methods section.

1. **Excluding known-malicious days from training.** Easy positives, thrown
   away.
2. **Testing on a later time window** rather than a random split.
3. **Scoring per day at a fixed daily capacity** rather than reporting AUROC
   over a pooled ranking, which would look considerably better.
4. **Capping the sequence at 128 events.** Doubling it quadrupled training time
   and moved nothing, but it is a cap and it is stated.
5. **Reporting alert concentration and over-alerting ratio at all.** These are
   the metrics most likely to make a model look bad, which is precisely why
   they are here.

---

## References

Glasser, J. & Lindauer, B. (2013). Bridging the Gap: A Pragmatic Approach to
Generating Insider Threat Data. *IEEE Security and Privacy Workshops*.

Perez, E. et al. (2018). FiLM: Visual Reasoning with a General Conditioning
Layer. *AAAI*.

Oord, A. van den, Li, Y. & Vinyals, O. (2018). Representation Learning with
Contrastive Predictive Coding. *arXiv:1807.03748*.

Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for
large-dimensional covariance matrices. *Journal of Multivariate Analysis*.
