# Data card — CERT Insider Threat Test Dataset

## Provenance

| | |
|---|---|
| **Publisher** | CERT Division, Software Engineering Institute, Carnegie Mellon University, with ExactData LLC |
| **Sponsor** | DARPA I2O |
| **Citation** | Glasser, J. & Lindauer, B. (2013). *Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data.* IEEE Security and Privacy Workshops |
| **Download** | https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 |
| **Release used** | r4.2 — roughly 1,000 employees over about 17 months |
| **Redistribution** | Not redistributed in this repository. `scripts/prepare_local.py` reads your own copy in place. |

## The most important fact about it

**The data is synthetic.** It was generated to look like a real organisation,
not collected from one. Behaviour is more regular than reality, the
organisation never reorganises chaotically, nobody has a genuinely strange but
innocent month, and the insider campaigns follow scripts.

That is a real limitation and it belongs at the top rather than in a footnote.
It is also why the dataset exists: no company will release seventeen months of
its employees' email, browsing and file access with the insiders labelled, and
without something in its place there is no shared basis for comparing methods
at all.

What follows from it: results on this data are evidence that a method works
**on this data**. They are not an estimate of what it would do in a real
building, and this repository does not present them as one.

## Files

| File | Rows (r4.2) | Content |
|---|---|---|
| `logon.csv` | ~0.9 M | Logon / logoff, per user and machine |
| `device.csv` | ~0.4 M | Removable-media connect / disconnect |
| `file.csv` | ~0.4 M | File access, with filename and content |
| `http.csv` | ~28 M | Web visits, with URL and page text — 1.7 GB alone |
| `email.csv` | ~2.6 M | One row per delivery: to/cc/bcc, from, size, attachments, body |
| `psychometric.csv` | 1,000 | Big Five (OCEAN) scores, one row per person |
| `LDAP/*.csv` | 1,000 × 17 | Monthly directory snapshots: role, unit, department, team, supervisor |
| `answers/` | — | The malicious events, per scenario |

The monthly LDAP snapshots matter more than they look. People change role
mid-study; a pipeline that pins everyone to the January snapshot will call
every promotion an anomaly for the rest of the year.

## The three campaigns in r4.2

| Scenario | Shape |
|---|---|
| 1 | Logs in after hours, copies data to a thumb drive, leaves the company |
| 2 | Browses job sites, then steals data on the way out |
| 3 | A disgruntled systems administrator downloads keylogging tools and plants a logic bomb |

They are not equally hard, which is why results are broken out per scenario.
Scenario 2 is the one where the content modality should earn its place — the
job-hunting is visible in text long before it is visible in event counts.

## Class balance

Malicious user-days are on the order of a few dozen among hundreds of
thousands: a prevalence around 0.01%. Two consequences the pipeline is built
around:

- Supervised training is not viable, hence the self-supervised objectives.
- Accuracy is meaningless and AUROC is misleading, hence the capacity-based
  evaluation.

## What this pipeline keeps, and what it discards

| Kept | Discarded |
|---|---|
| Typed event sequences, 128 per user-day | Raw event volume beyond the per-source budget |
| Hour, weekend, after-hours, own/shared machine | Machine names, filenames, URLs |
| 12 document embeddings per user-day, 64-d | All raw message text |
| Role, unit, department, team, supervisor, OCEAN | Employee names, email addresses |

The reduction is roughly forty to one, and the artefacts contain no readable
message content — which is also what makes them safe to move between machines.

## Known handling decisions

| Issue | Handling |
|---|---|
| Empty CC/BCC cells arrive as `NaN` and stringify to `"nan"` | Cleaned before address parsing; a test pins it (this bug once classified 100% of mail as external) |
| The same message appears once per delivery | Rows where the sender is not the user are typed as receipts |
| `answers/` layout differs between releases | Both the per-scenario directory and a single `answers.csv` are read |
| Role changes mid-study | Context resolved per day via `merge_asof` against monthly snapshots |
| Users with no LDAP snapshot before their first active day | Assigned the earliest available snapshot rather than dropped |
| Roles too small for a covariance estimate | Fall back to functional unit, then business unit |

## Sensitive attributes

`role`, `business_unit`, `functional_unit`, `department`, `team`,
`supervisor`, and the five psychometric scores.

Role is used deliberately and centrally — it defines the peer group, and the
whole argument of the project is that comparing people to their role peers is
*fairer* as well as more accurate than comparing them to the organisation.

The psychometric scores are a different matter. They are in the release, the
ablation measures what they contribute, and
[ETHICS_AND_DEPLOYMENT.md](ETHICS_AND_DEPLOYMENT.md) argues they should be
removed from any real deployment regardless of what that measurement says.
Personality-score-driven suspicion of employees is not a thing to build.

## Population caveats

One thousand people, one organisation, seventeen months, 2010-era working
patterns. No remote work at scale, no cloud SaaS, no personal devices, no
messaging platforms. A modern enterprise generates a different shape of
telemetry entirely.

The *method* — peer-relative scoring over fused behaviour, content and context
— is designed to transfer. The trained weights are not.
