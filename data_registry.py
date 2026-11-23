######################
# Nützliche Funktionen #
#####################
from domino.core import LaggedAnalyser
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
from dask.distributed import Client
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from typing import List, Optional


@dataclass
class Entry:
    category: str
    name: str
    timestep: str
    grid: str
    path: Path


class Bibliography:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.entries: List[Entry] = []

    # ---------------------------------
    # Eintrag hinzufügen
    # ---------------------------------
    def add(self, category: str, name: str, timestep: str, grid: str, relative_path: str):
        entry = Entry(category, name, timestep, grid, self.base_path / relative_path)
        self.entries.append(entry)

    # ---------------------------------
    # Zugriff über Index
    # ---------------------------------
    def get_by_index(self, index: int) -> Entry:
        return self.entries[index]

    # ---------------------------------
    # Zugriff über Kategorie
    # ---------------------------------
    def get_by_category(self, category: str) -> List[Entry]:
        return [e for e in self.entries if e.category == category]

    # ---------------------------------
    # Zugriff über Name
    # ---------------------------------
    def get_by_name(self, name: str) -> Optional[Entry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    # ---------------------------------
    # Zugriff über Kategorie + Name
    # ---------------------------------
    def get(self, category: str, name: str) -> Optional[Path]:
        for e in self.entries:
            if e.category == category and e.name == name:
                return e.path
        return None

    # ---------------------------------
    # Übersicht
    # ---------------------------------
    def overview(self):
        print("\n=== DATA BIBLIOGRAPHY ===")
        for i, e in enumerate(self.entries):
            print(f"[{i}] {e.category} | {e.name} -> {e.path}")
