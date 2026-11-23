# master_thesis

Skripte und Notebooks für Datenvorbereitung, Index-Berechnung, Forecasting und Visualisierung im Rahmen der Masterarbeit.

## Voraussetzungen

- Python 3.10+ (empfohlen)
- `pip`
- `venv` (im Standard-Python enthalten)

## Virtuelle Umgebung erstellen

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Python-Requirements

Die installierten Pakete sind identisch zu `pw-work-new`:

- wheel
- numpy
- pandas
- tqdm
- gcsfs

## Hinweise

- Einige Notebooks sind groß; für Versionsverwaltung ggf. Git LFS verwenden.
- Je nach Script können zusätzliche System-Tools oder Zugangsdaten für Datenquellen nötig sein.
