from sklearn.model_selection import TimeSeriesSplit
from typing import Generator
import pandas as pd
import numpy as np


class MonthlyTimeSeriesSplit:
    """
    Implementa um divisor de séries temporais para validação cruzada em
    blocos mensais, permitindo a configuração de requisitos mínimos para
    o último mês.
    
    Parâmetros:
    -----------
    date_column : str
        Nome da coluna que contém os dados de tempo.

    n_splits : int, padrão=5
        Número de divisões (splits) para a validação cruzada.
        
    max_train_size : int, padrão=None
        Tamanho máximo para o conjunto de treino. Se None, não há
        limite.
        
    test_size : int, padrão=None
        Tamanho do conjunto de teste. Se None, será determinado automa-
        ticamente.
        
    gap : int, padrão=0
        Número de observações a serem ignoradas entre treino e teste.
        
    min_days_in_last_month : int, padrão=None
        Número mínimo de dias que o último mês deve conter para ser
        incluído nos splits. Se None, qualquer entrada no último mês faz
        com que ele seja incluído.
    """

    def __init__(self,
                 date_column: str,
                 n_splits: int = 5,
                 max_train_size: int|None = None,
                 test_size: int|None = None,
                 gap: int = 0,
                 min_days_in_last_month: int|None = None
                 ) -> None:
        
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.date_column = date_column
        self.min_days_in_last_month = min_days_in_last_month


    def split(self,
              X: pd.DataFrame,
              y: pd.DataFrame|None = None,
              groups: np.ndarray|None = None
              ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera os índices para os conjuntos de treino e teste em blocos
        mensais, excluindo o último mês caso não contenha dias sufici-
        entes.

        Parâmetros:
        -----------
        X : DataFrame
            Conjunto de dados de entrada com uma coluna de datas espe-
            cificada em `date_column`.
        
        y : array-like, opcional
            Array de valores alvo (não é usado diretamente no método).
        
        groups : array-like, opcional
            Grupos de dados (não é usado diretamente no método).
        
        Retorna:
        --------
        generator
            Gera tuplas de índices para treino e teste.
        """

        # Converte a coluna de datas e cria a coluna de blocos mensais
        X = X.copy()
        X['_Month'] = pd.to_datetime(X[self.date_column]).dt.to_period('M')
        unique_months = X['_Month'].unique()
        
        # Verifica se o último mês atende à condição de dias mínimos
        if self.min_days_in_last_month is not None:
            last_month = unique_months[-1]
            last_month_data = X[X['_Month'] == last_month]
            max_day_in_last_month = last_month_data[self.date_column].dt.day.max()
            
            # Remove o último mês se ele não atender ao mínimo de dias
            # especificado
            if max_day_in_last_month < self.min_days_in_last_month:
                unique_months = unique_months[:-1]
        
        # Usa o TimeSeriesSplit para dividir os blocos mensais
        month_splits = TimeSeriesSplit(
            n_splits=self.n_splits,
            max_train_size=self.max_train_size,
            test_size=self.test_size,
            gap=self.gap
        )
        
        for train_month_idx, test_month_idx in month_splits.split(unique_months):
            train_months = unique_months[train_month_idx]
            test_months = unique_months[test_month_idx]
            
            # Seleciona os índices do DataFrame com base nos meses de
            # treino e teste
            train_indices = X[X['_Month'].isin(train_months)].index
            test_indices = X[X['_Month'].isin(test_months)].index
            
            yield train_indices, test_indices


    def get_n_splits(self,
                     X: pd.DataFrame|None = None,
                     y: pd.DataFrame|None = None,
                     groups: np.ndarray|None = None
                     ) -> int:
        """
        Retorna o número de splits.

        Parâmetros:
        -----------
        X : DataFrame, opcional
            Conjunto de dados de entrada (não é usado diretamente).
        
        y : array-like, opcional
            Array de valores alvo (não é usado diretamente).
        
        groups : array-like, opcional
            Grupos de dados (não é usado diretamente).
        
        Retorna:
        --------
        int
            Número de splits configurado.
        """
        
        return self.n_splits
    

    def describe_splits(self,
                        X: pd.DataFrame
                        ) -> None:
        """
        Exibe os meses utilizados em cada divisão de treino e teste para
        os splits definidos pela validação cruzada.

        Parâmetros:
        -----------
        X : DataFrame
            Conjunto de dados de entrada que contém a coluna de datas
            especificada no parâmetro `date_column` durante a iniciali-
            zação da classe.
        
        Exemplo de saída:
        -----------------
        treino: ['1/23' '2/23' '3/23']; teste: ['4/23']
        """

        for train, val in self.split(X):

            meses_treino = np.unique(
                X.loc[train].date.dt.month.astype(str) +\
                '/' +\
                X.loc[train].date.dt.year.astype(str).str[2:]
            )

            meses_teste = np.unique(
                X.loc[val].date.dt.month.astype(str) +\
                '/' +\
                 X.loc[val].date.dt.year.astype(str).str[2:])
            
            print(f'treino: {meses_treino}; teste: {meses_teste}')
