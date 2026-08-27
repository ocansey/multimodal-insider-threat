"""A synthetic organisation that writes CERT-shaped files.

Read this section before reading the code, because misunderstanding what this
module is for would be worse than not having it.

**Nothing produced here is ever a result.** It exists so that the pipeline can
be developed, unit-tested and run in continuous integration without the real
1.5 GB release, and so that a reviewer who clones the repository can watch the
whole thing work end to end in ninety seconds before deciding whether to spend
an hour downloading real data. Every artefact it writes is stamped
``synthetic: true`` in the manifest, :func:`mint.artifacts.load` refuses to
load a synthetic artefact unless explicitly asked, and
``tests/test_no_fake_results.py`` asserts that no file under ``reports/`` was
ever produced from one.

What it does model, because the pipeline breaks in interesting ways without
them:

* **Role structure.** People in the same role behave more like each other than
  like the organisation as a whole. Without this the peer-relative machinery
  has nothing to find and every normalisation strategy scores identically,
  which would make the smoke test worthless.
* **Individual habit.** Each person has a stable working rhythm — start hour,
  volume, weekend propensity — so that self-relative baselines have something
  to be relative to.
* **Drift.** Volumes rise slowly across the period, so that a model trained on
  the first half and applied to the second half meets the same distribution
  shift the real data has.
* **Three campaign shapes** loosely echoing the r4.2 scenarios: after-hours
  removable-media exfiltration, a job-hunting departure, and a privileged user
  going quiet then acting. They ramp over days rather than appearing fully
  formed, because time-to-detection is meaningless against a step function.

What it deliberately does **not** model is the natural-language content of the
real emails and web pages. Text here is drawn from a small template bank; it
is enough to exercise the encoding path and the shapes of every array, and
nowhere near enough to draw a conclusion about whether text helps. That
question is answered on real data or not at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = [
    "Salesman", "ITAdmin", "Electrical Engineer", "Production Line Worker",
    "Mechanical Engineer", "Director", "Technical Writer", "Software Engineer",
]
BUSINESS_UNITS = ["1 - Research and Engineering", "2 - Manufacturing",
                  "3 - Sales and Marketing", "4 - Administration"]

#: Deliberately bland template text. Realistic enough in structure, nowhere
#: near realistic enough to evaluate the content modality on.
EMAIL_TEMPLATES = [
    "Attached is the {noun} you asked for. Let me know if anything looks off.",
    "Can we move the {noun} review to Thursday? I have a conflict.",
    "Following up on the {noun} - still waiting on sign-off from {role}.",
    "Please find the updated {noun} schedule for next quarter.",
    "Quick question about the {noun} process before I send this on.",
]
WEB_TEMPLATES = [
    "internal wiki page describing the {noun} procedure",
    "vendor documentation for {noun} equipment",
    "news article about {noun} industry trends",
    "training material covering {noun} compliance",
]
JOB_HUNT_TEMPLATES = [
    "job listing for senior {role} position competitive salary",
    "resume builder professional templates {role}",
    "recruiter contact page submit your CV {role}",
]
EXFIL_TEMPLATES = [
    "how to transfer large files to personal storage",
    "usb drive encryption bypass utility download",
    "keylogger software free trial download",
]
NOUNS = ["budget", "schematic", "roster", "compliance", "logistics",
         "procurement", "safety", "throughput"]


@dataclass
class SimConfig:
    n_users: int = 120
    n_days: int = 180
    start_date: str = "2010-01-04"
    n_insiders: int = 6
    seed: int = 7
    pcs_per_user: int = 1
    n_shared_pcs: int = 4
    #: Base events per user-day before individual and role modifiers.
    base_events: int = 22
    campaign_length_days: tuple[int, int] = (10, 30)
    text_dim: int = 384
    scenario_mix: tuple[int, ...] = field(default_factory=lambda: (1, 1, 2, 2, 3, 3))


class Organisation:
    """Generates a consistent fake company and writes CERT-shaped CSVs."""

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._build_people()
        self._build_campaigns()

    # -- population ---------------------------------------------------------
    def _build_people(self) -> None:
        c, rng = self.cfg, self.rng
        n = c.n_users
        users = [f"AAA{i:04d}" for i in range(n)]
        roles = rng.choice(ROLES, n)
        units = rng.choice(BUSINESS_UNITS, n)

        # Role effects: what a role does to a person's rhythm. These are the
        # structure the peer-relative model is supposed to discover.
        role_start = {r: rng.uniform(6.5, 10.0) for r in ROLES}
        role_volume = {r: rng.uniform(0.6, 1.8) for r in ROLES}
        role_weekend = {r: rng.uniform(0.02, 0.30) for r in ROLES}
        role_removable = {r: rng.uniform(0.0, 0.25) for r in ROLES}
        # Administrators legitimately work odd hours on shared machines. If
        # the model cannot learn this it will spend its whole alert budget on
        # them, which is the single most common way these systems fail in
        # production.
        role_start["ITAdmin"] = 6.0
        role_weekend["ITAdmin"] = 0.45
        role_removable["ITAdmin"] = 0.40

        self.people = pd.DataFrame({
            "user": users,
            "role": roles,
            "business_unit": units,
            "functional_unit": rng.choice(["1", "2", "3"], n),
            "department": rng.choice([f"Dept-{k}" for k in range(6)], n),
            "team": rng.choice([f"Team-{k}" for k in range(12)], n),
            "supervisor": rng.choice(["", *users[:8]], n),
            "own_pc": [f"PC-{i:04d}" for i in range(n)],
            # Individual habit, drawn around the role mean.
            "start_hour": [role_start[r] + rng.normal(0, 0.7) for r in roles],
            "volume": [role_volume[r] * rng.lognormal(0, 0.25) for r in roles],
            "p_weekend": [np.clip(role_weekend[r] + rng.normal(0, 0.05), 0, 1)
                          for r in roles],
            "p_removable": [np.clip(role_removable[r] + rng.normal(0, 0.05), 0, 1)
                            for r in roles],
            "O": rng.integers(10, 60, n), "C": rng.integers(10, 60, n),
            "E": rng.integers(10, 60, n), "A": rng.integers(10, 60, n),
            "N": rng.integers(10, 60, n),
        })
        self.shared_pcs = [f"PC-SHARED-{k}" for k in range(c.n_shared_pcs)]
        self.dates = pd.date_range(c.start_date, periods=c.n_days, freq="D")

    def _build_campaigns(self) -> None:
        """Plant a handful of insiders with ramped, dated campaigns."""
        c, rng = self.cfg, self.rng
        chosen = rng.choice(self.people.index, c.n_insiders, replace=False)
        rows = []
        # The campaign has to fit inside the back half of the period, which is
        # not automatic for short runs: with 60 days and a 30-day campaign the
        # earliest legal start is after the latest legal one. Clamp rather than
        # crash, so a small fixture is still usable in a unit test.
        earliest = int(c.n_days * 0.55)
        for k, idx in enumerate(chosen):
            scenario = int(c.scenario_mix[k % len(c.scenario_mix)])
            lo, hi = c.campaign_length_days
            hi = min(hi, max(lo + 1, c.n_days - earliest - 1))
            length = int(rng.integers(lo, max(lo + 1, hi)))
            latest = max(earliest + 1, c.n_days - length - 1)
            start = int(rng.integers(earliest, latest))
            rows.append({
                "user": self.people.loc[idx, "user"],
                "scenario": scenario,
                "start_day": start,
                "end_day": start + length,
            })
        self.campaigns = pd.DataFrame(rows)

    # -- day generation -----------------------------------------------------
    def _campaign_state(self, user: str, day: int) -> tuple[int, float]:
        """Return (scenario, intensity in 0-1) for this user on this day."""
        m = self.campaigns[self.campaigns["user"] == user]
        if m.empty:
            return 0, 0.0
        row = m.iloc[0]
        if not (row["start_day"] <= day <= row["end_day"]):
            return 0, 0.0
        span = max(1, row["end_day"] - row["start_day"])
        # Linear ramp: early campaign days are barely distinguishable, which
        # is exactly what makes time-to-detection a real measurement.
        return int(row["scenario"]), float((day - row["start_day"]) / span)

    def _day_events(self, person: pd.Series, day: int, date: pd.Timestamp):
        rng = self.rng
        scenario, intensity = self._campaign_state(person["user"], day)
        is_weekend = date.weekday() >= 5

        if is_weekend and rng.random() > person["p_weekend"] + 0.4 * intensity:
            return []

        drift = 1.0 + 0.25 * (day / self.cfg.n_days)   # slow volume growth
        n_events = max(3, int(rng.poisson(
            self.cfg.base_events * person["volume"] * drift * (1 + 0.6 * intensity))))

        start = person["start_hour"]
        if scenario == 1:
            start -= 9.0 * intensity          # drifts into the small hours
        elif scenario == 3:
            start -= 3.0 * intensity

        events = []
        pc_own, pc_shared = person["own_pc"], rng.choice(self.shared_pcs)
        events.append(("logon", start, pc_own))

        for _ in range(n_events):
            hour = float(np.clip(start + abs(rng.normal(0, 3.2)), 0, 23.9))
            on_shared = rng.random() < (0.05 + 0.35 * intensity * (scenario == 3))
            pc = pc_shared if on_shared else pc_own
            r = rng.random()

            if r < 0.45:
                events.append(("http_visit", hour, pc))
            elif r < 0.62:
                events.append(("email_send_internal", hour, pc))
            elif r < 0.74:
                events.append(("email_receive", hour, pc))
            elif r < 0.82:
                events.append(("file_open", hour, pc))
            elif r < 0.88:
                events.append(("email_send_external", hour, pc))
            elif r < 0.94:
                events.append(("device_connect", hour, pc))
            else:
                p = person["p_removable"] + 0.55 * intensity * (scenario in (1, 2))
                events.append(("file_copy_to_removable" if rng.random() < p
                               else "file_open", hour, pc))

        if scenario == 2:
            for _ in range(int(6 * intensity)):
                events.append(("http_visit", float(np.clip(
                    start + abs(rng.normal(0, 2)), 0, 23.9)), pc_own))

        events.append(("logoff", float(min(23.9, max(e[1] for e in events) + 0.4)),
                       pc_own))
        events.sort(key=lambda e: e[1])
        return [(t, h, pc, scenario, intensity) for t, h, pc in events]

    # -- text ---------------------------------------------------------------
    def _text_for(self, token: str, scenario: int, intensity: float,
                  role: str) -> str | None:
        rng = self.rng
        if token.startswith("email"):
            bank = EMAIL_TEMPLATES
        elif token == "http_visit":
            if scenario == 2 and rng.random() < intensity:
                return rng.choice(JOB_HUNT_TEMPLATES).format(role=role)
            if scenario in (1, 3) and rng.random() < intensity * 0.6:
                return rng.choice(EXFIL_TEMPLATES)
            bank = WEB_TEMPLATES
        elif token.startswith("file"):
            bank = ["document {noun} revision", "spreadsheet {noun} export"]
        else:
            return None
        return rng.choice(bank).format(noun=rng.choice(NOUNS), role=role)

    # -- writing ------------------------------------------------------------
    def write(self, out_dir: Path) -> dict:
        """Write CERT-shaped CSVs plus an answers file. Returns a manifest."""
        out_dir = Path(out_dir)
        (out_dir / "LDAP").mkdir(parents=True, exist_ok=True)

        rows = {k: [] for k in ("logon", "device", "file", "http", "email")}
        answers = []
        eid = 0

        for _, person in self.people.iterrows():
            for day, date in enumerate(self.dates):
                for token, hour, pc, scenario, intensity in self._day_events(
                        person, day, date):
                    eid += 1
                    ts = date + pd.Timedelta(hours=float(hour))
                    stamp = ts.strftime("%m/%d/%Y %H:%M:%S")
                    text = self._text_for(token, scenario, intensity,
                                          person["role"])
                    base = {"id": f"{{E{eid:09d}}}", "date": stamp,
                            "user": person["user"], "pc": pc}
                    malicious = scenario > 0 and self._is_malicious(token, scenario)

                    if token in ("logon", "logoff"):
                        rows["logon"].append({**base, "activity":
                                              "Logon" if token == "logon" else "Logoff"})
                        table = "logon"
                    elif token.startswith("device"):
                        rows["device"].append({**base, "activity":
                                               "Connect" if token.endswith("connect")
                                               else "Disconnect"})
                        table = "device"
                    elif token.startswith("file"):
                        rows["file"].append({
                            **base,
                            "filename": ("R:\\export.zip"
                                         if token == "file_copy_to_removable"
                                         else "C:\\work\\doc.docx"),
                            "content": text or ""})
                        table = "file"
                    elif token == "http_visit":
                        rows["http"].append({**base, "url": "http://example.invalid/p",
                                             "content": text or ""})
                        table = "http"
                    else:
                        external = token == "email_send_external"
                        received = token == "email_receive"
                        other = f"AAA{int(self.rng.integers(0, self.cfg.n_users)):04d}"
                        rows["email"].append({
                            **base,
                            "to": (f"{person['user']}@company.invalid" if received
                                   else "outside@elsewhere.invalid" if external
                                   else f"{other}@company.invalid"),
                            "cc": "", "bcc": "",
                            "from": (f"{other}@company.invalid" if received
                                     else f"{person['user']}@company.invalid"),
                            "size": int(self.rng.integers(400, 60000)),
                            "attachments": int(self.rng.random() < 0.15),
                            "content": text or ""})
                        table = "email"

                    if malicious:
                        answers.append({"id": base["id"], "date": stamp,
                                        "user": person["user"],
                                        "scenario": scenario, "table": table})

        for name, recs in rows.items():
            pd.DataFrame(recs).to_csv(out_dir / f"{name}.csv", index=False)

        self.people[["user", "O", "C", "E", "A", "N"]].rename(
            columns={"user": "user_id"}).assign(
            employee_name=self.people["user"]).to_csv(
            out_dir / "psychometric.csv", index=False)

        ldap = self.people[["user", "role", "business_unit", "functional_unit",
                            "department", "team", "supervisor"]].rename(
            columns={"user": "user_id"})
        ldap.insert(0, "employee_name", self.people["user"])
        ldap["email"] = self.people["user"] + "@company.invalid"
        for month in pd.date_range(self.dates[0], self.dates[-1], freq="MS"):
            ldap.to_csv(out_dir / "LDAP" / f"{month:%Y-%m}.csv", index=False)

        pd.DataFrame(answers).to_csv(out_dir / "answers.csv", index=False)

        manifest = {
            "synthetic": True,
            "generator": "mint.simulate.Organisation",
            "warning": "SMOKE-TEST FIXTURE. Never report a number computed from this.",
            "n_users": int(self.cfg.n_users),
            "n_days": int(self.cfg.n_days),
            "n_insiders": int(self.cfg.n_insiders),
            "n_malicious_events": len(answers),
            "seed": int(self.cfg.seed),
            "campaigns": self.campaigns.to_dict("records"),
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return manifest

    @staticmethod
    def _is_malicious(token: str, scenario: int) -> bool:
        if scenario == 1:
            return token in ("file_copy_to_removable", "device_connect")
        if scenario == 2:
            return token in ("http_visit", "file_copy_to_removable")
        if scenario == 3:
            return token in ("http_visit", "file_open", "device_connect")
        return False


def build_fixture(out_dir: Path, **kwargs) -> dict:
    """Convenience entry point used by the tests and the Makefile."""
    return Organisation(SimConfig(**kwargs)).write(Path(out_dir))
