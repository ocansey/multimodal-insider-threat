# Ethics and deployment

Insider-threat detection is workplace surveillance. Building it well and
building it responsibly are the same problem, not competing ones, and a system
that ignores the second will fail at the first — because people who believe
they are being unfairly watched route around the monitoring, and because a
queue that audits the night shift for existing gets ignored within a fortnight.

## What this system may do

Produce a ranked daily queue of accounts whose behaviour warrants a **human
looking at it**, with a stated reason and a stated confidence.

That is the whole permitted scope. Every alert is a request for human
attention, and the human decides.

## What it may not do

- **Trigger any automated action.** No account lockouts, no access revocation,
  no automatic escalation. A model with a 0.01% base rate and a capacity-limited
  queue is wrong most of the time by construction.
- **Feed performance management.** The score is not evidence of productivity,
  loyalty or intent, and any organisation that lets it leak into an appraisal
  has converted a security tool into a disciplinary one.
- **Operate covertly.** Staff should know the system exists, what data it uses
  and what happens when it fires. Covert behavioural scoring is unlawful in
  much of the world and corrosive everywhere else.
- **Be used on individuals rather than populations.** Pointing it at one person
  a manager already suspects inverts the statistics entirely: the base rate is
  no longer 0.01% and the calibration is meaningless.

## The psychometric problem

The CERT release includes Big Five personality scores. They are in the model
here so that the ablation can measure what they are worth — publishing "we left
them out" without evidence is weaker than publishing the number.

**They should not be in a deployed system**, whatever the ablation says.
Scoring employees as more suspicious because of a personality assessment is
indefensible independently of whether it improves AUROC: the assessment is
usually self-reported, its predictive validity for anything at this timescale
is contested, and the failure mode — a person permanently flagged as
higher-risk for a trait — is not one a security team should be creating.

`--arm no_context` measures the cost of removing the whole context block.
Splitting psychometrics out separately from the organisational context is the
first change I would make before anyone deployed this.

## Peer grouping is a fairness mechanism, not just an accuracy one

The core method compares people to their role peers. That is usually presented
as an accuracy argument — and it is — but the fairness consequence is larger.

Under global normalisation, the alert budget lands on whoever has the most
irregular *legitimate* working pattern. In practice that means systems
administrators, shift workers, field staff, people in different time zones, and
often people with caring responsibilities or disabilities that shape when they
work. None of those are threat indicators. All of them attract repeated
unexplained scrutiny under a company-wide baseline.

Peer-relative scoring removes that by construction: an administrator is
compared to administrators. The `over_alerting_ratio` column in the burden
table is there to make the effect measurable rather than asserted, and it
should be checked every term in any deployment, not once at go-live.

There is a real tension to name. Peer grouping can also *hide* a threat when a
whole group drifts together — a compromised team, or a normalised bad practice.
The system is not a substitute for controls that do not depend on comparison.

## Governance a real deployment would need

- **A stated retention period** for scores and for the underlying artefacts.
- **An appeal route.** A person who was queued repeatedly and cleared every
  time should be able to find that out and have it looked at.
- **Recalibration every term** — base rates and working patterns drift, and a
  system calibrated in 2010 conditions produces 2010-shaped errors.
- **A standing burden audit**, by role and by any protected characteristic the
  organisation holds, with a named owner.
- **A kill switch owned outside the security team.**

## On publishing this

The code is public and the dataset is synthetic and openly published, so
nothing here exposes a real person. If you point this at real telemetry, the
artefacts stop being safe to move around: they contain per-person behavioural
embeddings, and while they hold no readable text, they are personal data in
every sense that matters legally.
