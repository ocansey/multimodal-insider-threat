# Unusual compared to whom?

**A multimodal transformer for insider-threat detection that scores a person's day against their role peers rather than against the organisation — because a systems administrator working at 3am is not an anomaly, and a model that says otherwise will spend every alert it has on the night shift.**

*Isaiah Thompson Ocansey, PhD* · [ocanthom@gmail.com](mailto:ocanthom@gmail.com) · [LinkedIn](https://www.linkedin.com/in/itor/) · [GitHub](https://github.com/ocansey)

---

## The problem with the usual framing

Insider-threat detection is normally posed as: learn what normal looks like, flag departures from it. That framing contains a question it never asks out loud — **normal for whom?**

Answer it with "the organisation" and you get the failure everyone in security operations has lived through. The people who look strangest against a company-wide baseline are systems administrators, field sales, on-call engineers, anyone whose job is irregular by design. They are not threats. They are the night shift, and they will consume the entire alert budget every day until somebody turns the tool off.

Answer it with "this person's own history" and you fix that and buy something worse. An insider who has been quietly exfiltrating for three weeks has *established that behaviour as their own normal*. Self-relative scoring habituates to exactly the campaigns it exists to catch, and it habituates harder the longer the campaign runs.

This project answers it with **the person's role peers**, and treats the choice as the experiment rather than as an implementation detail. The organisation has already written down who is comparable to whom — it is sitting in the directory — and almost nobody reads it.

## What the model actually is

A transformer that reads one person's working day across three genuinely different kinds of data:

| Modality | What it is | How it enters the model |
|---|---|---|
| **Behaviour** | The ordered sequence of typed events — logon, USB connect, file copy to removable media, web visit, mail sent internally or externally — each with its hour and four binary facts (after hours, weekend, own machine, shared machine) | Embedded and passed through a 4-layer transformer encoder |
| **Content** | The text the person wrote and read that day: emails, web pages, document contents | Encoded once by a frozen sentence transformer, then attended over — never averaged, because one unusual message among forty routine ones must survive |
| **Context** | Role, business unit, department, team, supervisory status, and the Big Five psychometric scores | Produces FiLM parameters that **modulate** the behaviour representation, rather than being concatenated onto it |

That last row is the architectural expression of the argument. Context should change *how behaviour is read*, not sit beside it as one more feature to average in. A director working at 11pm and a production-line worker working at 11pm are not the same event, and a model that staples the role on as a one-hot vector has to spend capacity rediscovering that. FiLM lets the role reach into the representation and rescale it directly.

Fusion is cross-attention: behaviour queries attend over content. What someone did is read in the light of what they wrote.

### Training without labels, because there are none worth having

There are a few dozen malicious user-days among hundreds of thousands. Supervised training on that is not hard, it is impossible — and every published F1 on this dataset has, somewhere, allowed one campaign to appear on both sides of a split.

So training is self-supervised, on two objectives:

1. **Masked event modelling.** Hide fifteen percent of a day's events and predict them. Teaches the model what a plausible day looks like for this kind of person.
2. **Cross-modal alignment.** An InfoNCE loss pulling a day's behaviour embedding towards its own content embedding and away from other days' in the batch.

The second is where the detector I find most interesting comes from. A model trained to predict what someone *wrote* from what they *did* fails on days where the two stop agreeing — where the file access pattern is that of a busy analyst but the browsing is job listings, or where the mail traffic looks routine and the device activity does not. **Cross-modal disagreement is an anomaly signal you get for free from an alignment objective, and no single-modality model has access to it at all.**

## How it is evaluated

Not by AUROC. A ranked list of half a million user-days is not a product; a morning queue is.

- **Precision at daily capacity.** Each day, the top *k* users by score among those active. Ten is the default, because that is roughly what one analyst can actually work.
- **Campaign recall and time to detection.** A campaign runs for weeks. Catching it on any of those days is a catch; catching it on day two rather than day nineteen is the entire value of the system. Campaigns are counted once, and the number attached to each is how many days ran before the first alert.
- **Alert concentration.** How many *distinct people* the budget lands on. A model that flags the same six administrators every day has a respectable AUROC and is useless, and no conventional metric will tell you that.
- **Over-alerting ratio by role.** Which roles get audited more heavily than their headcount justifies, for doing their jobs.

AUROC and average precision are reported alongside, because comparison with the literature would be impossible without them — and because the gap between a healthy AUROC and a poor precision-at-capacity is itself the argument for everything above.

---

## Results

<!-- results-pending -->

**Not yet run on real data.** The CERT release is not redistributable and is not reachable from the environment this was developed in, so the numbers below are deliberately absent rather than estimated. The pipeline is complete and tested; running it takes two commands once the data is downloaded (see below). When it has run, this section carries the real tables and `reports/tables/headline.csv` backs every figure in it.

What the run will produce:

- headline discrimination and campaign recall for each of `global` / `self` / `peer` normalisation — the central experiment
- the capacity sweep at 5, 10, 25 and 50 reviews per day
- time-to-detection per campaign and per scenario
- the modality ablation: `full`, `no_content`, `no_context`, `behaviour_only`
- which roles absorb the alert budget under each normalisation

A test guards this section: `tests/test_no_fake_results.py` fails if any file under `reports/` was produced from the synthetic fixture, and warns if this marker is still here once real results exist.

---

## Running it

### Check the pipeline first — no download needed

```bash
git clone https://github.com/ocansey/multimodal-insider-threat.git
cd multimodal-insider-threat
pip install -r requirements.txt

make test     # 32 tests, no data required
make smoke    # generate a synthetic org, run the whole pipeline end to end
```

`make smoke` builds a fake 60-person company with planted insider campaigns, runs preparation, training, scoring and evaluation, and prints the tables. It takes about ten minutes on two CPU cores and proves every code path works before you spend an hour on a download.

**Nothing it produces is a result.** The fixture is stamped `synthetic: true`, the loader refuses to open it without an explicit override, and every output row carries `data_source = SYNTHETIC-FIXTURE`. This matters more than it sounds: shipping a data simulator so that tests can run makes it very easy to publish a simulator number by accident.

### Then the real data

1. Download from CMU — free, no registration:
   [Insider Threat Test Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247). Take **`r4.2.tar.bz2`** and **`answers.tar.bz2`**.

2. Reduce it. This runs where the data is and never moves the raw text anywhere:

```bash
pip install sentence-transformers          # or use --text-encoder hashing
python scripts/prepare_local.py --raw ~/Downloads --out data/artifacts/cert
```

**Extraction is optional.** Unpacked, the release is about 3 GB and `http.csv` is 1.7 GB of that; added to the tarballs you already have, that is over four gigabytes of free disk for a study whose output is three hundred megabytes. If the space is not there, do not extract — point `--raw` at the tarball or the folder holding it and the loader streams members straight out of the archive. That costs roughly double the read time for `http.csv`, because bzip2 cannot seek and the pipeline reads each table twice, and it costs no disk at all.

Twenty minutes to an hour either way, mostly reading `http.csv` and running the encoder. Peak memory around 4 GB. Add `--sample-users 100` for a five-minute trial pass first.

3. Train and evaluate:

```bash
python scripts/run_experiment.py --artifacts data/artifacts/cert
```

### What the reduction step does, and why

The raw release is 1.5 GB, most of it web-page text. The preparation step keeps the twelve most informative documents per user-day, encodes them once, projects the embeddings from 384 dimensions to 64 with a PCA **fitted on the training window only**, and writes about 300 MB of arrays containing no raw message content.

The PCA detail is not housekeeping. Fitting it on everything would let the test period shape the representation of the training period — leakage with no label in sight, which is the kind that never shows up as an obviously wrong number.

---

## The parts that were harder than expected

Four things in here were bugs first, and each is documented at the point where it was fixed rather than quietly patched.

**Every email in the corpus was classified as external.** Empty CC and BCC cells arrive from pandas as float `NaN`, and `str(nan)` is the literal text `nan`, which does not end in the company's domain. The internal class was empty. No metric would have shown it — the token still existed, the sequences were still the right length, and the model trained perfectly happily on a vocabulary with a hole in it. `tests/test_pipeline_invariants.py` pins the behaviour now.

**One person's second day outranked every genuine anomaly.** The self-relative baseline divides by the within-person spread. With a single previous day that spread is exactly zero. The AUROC barely moved — one absurd score among thousands is invisible in an average — and the top-ten queue was ruined. Fixed with a minimum-history requirement and a floor on the denominator, both tested.

**The day boundary was in the wrong place.** Cutting at midnight splits a continuous fourteen-hour after-hours session into two unremarkable half-days on either side of the most interesting moment, and mangles every night-shift worker the same way. The day now runs 04:00 to 04:00, which is the quietest hour in the data.

**Sessionisation took eighteen seconds on a toy dataset**, which extrapolated to over an hour on the real release. Rewriting the per-source event budget as vectorised group arithmetic instead of a Python loop took it to half a second — a 37× speedup on the step that has to process thirty million rows.

**Extraction ran the disk out of space**, halfway through `http.csv`, leaving a truncated file that still looked like a file. The fix was not to ask for a bigger disk: `mint/sources.py` resolves every table to a byte stream and reads members directly out of the `.tar.bz2`, so the release never has to be unpacked at all. A test asserts the archive path and the extracted path produce byte-identical token arrays, because a convenience that quietly changes the numbers is worse than the inconvenience it removes.

## Repository layout

```
src/mint/
  schema.py       the CERT files as they actually arrive; the event vocabulary
  simulate.py     a synthetic organisation, for tests only — read its docstring
  sources.py      resolves each table to a stream: directory, or straight from the tarball
  sessionise.py   five unsorted logs -> one ordered day per person
  text.py         the content modality; pretrained encoder and an offline floor
  prepare.py      raw release -> compact, transferable artefacts
  artifacts.py    the handoff format, and the guard against publishing fixtures
  model.py        the fusion transformer and its two self-supervised objectives
  training.py     the training loop and the scoring pass
  scoring.py      global vs self vs peer — the experiment
  evaluate.py     capacity queues, campaign recall, time to detection, burden
  pipeline.py     one call, every table

scripts/  prepare_local.py · run_experiment.py
tests/    32 tests: invariants, hand-worked metrics, and the no-fake-results guard
docs/     METHODOLOGY.md · DATA_CARD.md · ETHICS_AND_DEPLOYMENT.md
config/   every threshold and split boundary a reviewer might argue with
```

## Honest limitations

- **The data is synthetic at source.** CERT's release was generated by ExactData, not collected from a real company. Behaviour is more regular than reality and the campaigns are scripted. It is also the only openly available dataset carrying per-person behaviour, free text *and* organisational context over a long enough window to study how a campaign unfolds. Results here are evidence that the method works on this data, not that it works in your building.
- **Peer groups need a population.** With a thousand people and eight roles the groups are large enough to estimate a covariance from. A two-hundred-person company would fall back to coarser groupings for most roles, and the whole argument weakens accordingly. `min_peer_group` controls the fallback and the group size used is reported per score.
- **Nothing here shows that acting on an alert helps.** The system produces a ranked queue with lead time attached. Whether contacting the people on it changes an outcome is not a question this or any offline dataset can answer.
- **Psychometric scores are in the model and I am uneasy about it.** They are in the release, the ablation measures what they are worth, and the ethics note argues they should be dropped in any real deployment regardless of what that measurement says.

## Data

Glasser, J. & Lindauer, B. (2013). *Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data.* IEEE Security and Privacy Workshops.

Released by the CERT Division, Software Engineering Institute, Carnegie Mellon University, with ExactData LLC, under DARPA I2O sponsorship. Not redistributed here — `scripts/prepare_local.py` reads your own copy in place.

## Licence

MIT for the code. The dataset carries CMU's own terms. See [LICENSE](LICENSE).
