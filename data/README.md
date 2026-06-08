# Dataset Instructions

This repository does **not** include raw dataset files.
Place downloaded files in the paths below before running any notebook.

---

## D1 — CERT Insider Threat Dataset r4.2

**Source:** Carnegie Mellon University SEI
**Download:** https://resources.sei.cmu.edu/tools/downloads/insider-threat/

After downloading, place files at:
```
data/cert_r4.2/
    device.csv
    email.csv
    file.csv
    http.csv
    logon.csv
    psychometric.csv
    LDAP/
```

---

## D2 — Insider Threat Dataset for Corporate Environments

**Source:** Kaggle — ahmeduzaki
**URL:** https://www.kaggle.com/datasets/ahmeduzaki/insider-threat-dataset-for-corporate-environments

Place file at:
```
data/corporate/insider_threat_clean_dataset.csv
```

---

## D3 — Insider Threat Dataset for Classified Environments

**Source:** Kaggle — efchbd1013
**URL:** https://www.kaggle.com/datasets/efchbd1013/insider-threat-dataset-for-classified-environments

Place file at:
```
data/classified/insider_threat_dataset.csv
```

---

## Kaggle Paths (if running on Kaggle)

```python
CERT_BASE = "/kaggle/input/cert-insider-threat-dataset-r42"
CORP_BASE = "/kaggle/input/datasets/ahmeduzaki/insider-threat-dataset-for-corporate-environments"
CLASS_BASE = "/kaggle/input/datasets/efchbd1013/insider-threat-dataset-for-classified-environments"
```
