from pathlib import Path
import pandas as pd
import xarray as xr

class RegionalMeanCalculator:
    """
    Berechnet Regionsmittelwerte für definierte Großregionen
    basierend auf NUTS-Codes (Spaltennamen im DataFrame).
    """

    def __init__(self):

        self.regions = {
            "Norddeutschland": ['DEF0', 'DE60','DE80','DE93','DE94','DE50'],
            "Süddeutschland": ["DE12", "DE11", "DE25", "DE21", "DE22", "DE23", "DE27", 'DE13', 'DE14'],
            "Westdeutschland": ['DEA3','DEA4','DEA5','DEA1','DEA2','DEB1','DEB2','DEB3','DE71', 'DE72','DE73'],
            "Ostdeutschland": ['DEE0','DE30','DE40','DED5','DED2', 'DED4','DEG0'],
            "50Hertz": ['DE30', 'DE40', 'DE60', 'DE80', 'DE93', 'DED2', 'DED4', 'DED5', 'DEE0', 'DEG0'],
            "Amprion": ['DE26', 'DEA1', 'DEA2', 'DEA3', 'DEA5', 'DEB1', 'DEB2', 'DEB3', 'DEC0'],
            "TransnetBW": ['DE11', 'DE12', 'DE13', 'DE14'],
            "TenneT": ['DE21', 'DE22', 'DE23', 'DE24', 'DE25', 'DE27', 'DE50', 'DE71', 'DE72', 'DE73', 'DE91', 'DE92', 'DE94', 'DEA4', 'DEF0']
        }

    def calculate_mean(
        self,
        df: pd.DataFrame,
        region_name: str
    ) -> pd.Series:

        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' ist nicht definiert.")

        nuts_codes = self.regions[region_name]

        available_codes = [c for c in nuts_codes if c in df.columns]

        if len(available_codes) == 0:
            raise ValueError("Keine passenden NUTS-Codes im DataFrame gefunden.")

        regional_mean = df[available_codes].mean(axis=1)
        regional_mean.name = region_name

        return regional_mean

    def load_energy_values_from_prepared(
        self,
        path: Path,
        energy_name: str,
        region_name: str | None = None,
    ) -> xr.DataArray:
        """
        Lädt vorbereitete Energiedaten (Long-Format) und gibt eine
        xarray.DataArray-Zeitreihe zurück.

        Parameters
        ----------
        path : Path
            Pfad zur vorbereiteten CSV.
        energy_name : str
            Name der Energiespalte.
        region_name : str | None
            Name einer Region aus self.regions.
            Falls None, wird über alle NUTS-Gebiete gemittelt.

        Returns
        -------
        xr.DataArray
            Zeitreihe der Energievariable.
        """

        df = pd.read_csv(path, parse_dates=["Date"])

        if energy_name not in df.columns:
            raise ValueError(
                f"Spalte '{energy_name}' fehlt in {path}"
            )

        # Long -> Wide
        df_wide = (
            df.pivot_table(
                index="Date",
                columns="NUTS_ID",
                values=energy_name,
                aggfunc="mean",
            )
            .sort_index()
        )

        # Regionale Mittelung
        if region_name is not None:
            series = self.calculate_mean(
                df=df_wide,
                region_name=region_name,
            )
        else:
            series = df_wide.mean(axis=1)
            series.name = energy_name

        return xr.DataArray(
            data=series.to_numpy(),
            dims=("time",),
            coords={"time": series.index.to_numpy()},
            name=series.name,
        )