import numpy as np
import pandas as pd
import warnings
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.model_selection import BaseCrossValidator


class AnchorSplit(BaseCrossValidator):
    """
    Cross-validator guiado por uma data-âncora (`exec_date`).

    - exec_date define o *weekday* e a *ordem* daquele weekday no mês
      (ex.: segunda terça-feira).  
    - Para cada ano anterior gera-se a mesma combinação; se cair em
      feriado/fim-de-semana, volta para o dia útil anterior.  
    - **Teste** = `test_size_bd` dias úteis a partir do anchor.  
    - **Treino** = todas as linhas anteriores (warning se < 252).

    Parameters
    ----------
    exec_date : str | pd.Timestamp
        Data real em que o modelo rodará.
    n_splits : int
        Máximo de folds desejados.
    holidays : list[pd.Timestamp]
        Lista de feriados (não-úteis).
    test_size_bd : int, default=42
        Quantidade de dias úteis no conjunto-teste.
    min_date : str | pd.Timestamp | None, optional
        Limite inferior para o primeiro dia de teste.
    """

    def __init__(
        self, 
        exec_date, 
        n_splits, 
        holidays,
        test_size_bd: int = 42, 
        min_date=None
    ):

        self.exec_date   = pd.Timestamp(exec_date)
        self.n_splits    = n_splits
        self.holidays    = holidays
        self.test_size   = int(test_size_bd)
        self.min_date    = pd.Timestamp(min_date) if min_date else None

        # padrão capturado da âncora
        self._wk   = self.exec_date.weekday() # Dia da semana (0 a 6).
        self._ord  = 1 + (self.exec_date.day - 1) // 7 # Ordem no mês.


    # ---------- utilidades internas ----------
    def _anchor_for_year(self, year: int) -> pd.Timestamp:
        """Retorna anchor (weekday & ordem) p/ determinado ano."""
        first = pd.Timestamp(year, self.exec_date.month, 1)
        delta = (self._wk - first.weekday()) % 7
        anchor = first + pd.Timedelta(days=delta + 7 * (self._ord - 1))
        # recua até dia útil se necessário
        while anchor.weekday() >= 5 or anchor in self.holidays:
            anchor -= pd.Timedelta(days=1)
        return anchor


    # ---------- interface scikit-learn ----------
    def split(self, X, y=None, groups=None):
        """
        X deve ser uma série/array datetime64[ns] alinhada às linhas de
        features passadas ao estimador.
        """
        dates = pd.DatetimeIndex(X)
        if not np.issubdtype(dates.dtype, np.datetime64):
            raise ValueError("X deve conter dtype datetime64[ns].")

        folds = 0
        year = self.exec_date.year - 1
        while folds < self.n_splits and year >= dates[0].year:
            anchor = self._anchor_for_year(year)
            if self.min_date is not None and anchor < self.min_date:
                break

            offset = CustomBusinessDay(
                n=self.test_size - 1, 
                holidays=self.holidays
            )  

            test_end = anchor + offset

            test_idx = np.where((dates >= anchor) & (dates <= test_end))[0]
            if test_idx.size == 0:
                year -= 1
                continue

            train_idx = np.where(dates < anchor)[0]
            if train_idx.size < 252:
                warnings.warn(
                    f"Fold {folds + 1}: treino com {train_idx.size} linhas (<252).",
                    RuntimeWarning)

            yield train_idx, test_idx
            folds += 1
            year  -= 1


    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits



holidays = []
cv = AnchorSplit(
        '2025-01-10', 
        7, 
        holidays, 
        min_date='2018-01-01')

X_data = pd.date_range(
    start='2018-01-01', 
    end='2025-01-10', 
    freq='B')  # Frequência de dias úteis

cv.split(X_data)
for train_idx, test_idx in cv.split(X_data):
    print(f"Train: {X_data[train_idx].min()} - {X_data[train_idx].max()}")
    print(f"Test:  {X_data[test_idx].min()} ({X_data[test_idx].min().day_name()}) - {X_data[test_idx].max()}")
    print("-" * 40)