"""
Модуль проверок качества данных для датасета Fish.

Используется в ноутбуке по EDA и в тестах.
Каждая функция возвращает DataFrame с результатом или raises AssertionError.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# === Конфигурация ===
DATA_PATH = Path("../assets/data/Fish.csv")
EXPECTED_COLUMNS = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]
EXPECTED_SPECIES = {"Bream", "Parkki", "Perch", "Pike", "Roach", "Smelt", "Whitefish"}
NUMERIC_COLS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загружает CSV и делает базовые ассерты схемы."""
    df = pd.read_csv(path)
    assert not df.empty, "Таблица пустая"
    assert df.columns.is_unique, "Есть дубликаты имён столбцов"
    assert set(df.columns) == set(EXPECTED_COLUMNS), (
        f"Схема не совпадает. Ожидали {EXPECTED_COLUMNS}, получили {list(df.columns)}"
    )
    return df


def check_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Паспорт: имя поля, тип, пропуски, уникальные значения."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(dropna=False),
    })


def check_positive(df: pd.DataFrame) -> pd.DataFrame:
    """Проверка положительности числовых полей."""
    rows = []
    for col in NUMERIC_COLS:
        rows.append({
            "column": col,
            "min": df[col].min(),
            "max": df[col].max(),
            "n_negative": int((df[col] < 0).sum()),
            "n_zero": int((df[col] == 0).sum()),
        })
    return pd.DataFrame(rows)


def check_length_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Проверка согласованности длин: Length1 ≤ Length2 ≤ Length3."""
    mask = (df["Length1"] > df["Length2"]) | (df["Length2"] > df["Length3"])
    return df.loc[mask, ["Species", "Length1", "Length2", "Length3"]]


def check_species_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Список видов и их численность."""
    counts = df["Species"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})


def check_duplicates(df: pd.DataFrame) -> dict:
    """Полные дубликаты и дубликаты по измерениям."""
    return {
        "full_duplicates": int(df.duplicated().sum()),
        "measurement_duplicates": int(
            df.duplicated(subset=[c for c in df.columns if c != "Weight"]).sum()
        ),
    }


def check_species_values(df: pd.DataFrame) -> pd.DataFrame:
    """Неожиданные категории вида."""
    unexpected = set(df["Species"].unique()) - EXPECTED_SPECIES
    if unexpected:
        return pd.DataFrame({"unexpected_species": list(unexpected)})
    return pd.DataFrame({"unexpected_species": []})


def check_weight_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Потенциальные выбросы массы по правилу k×IQR."""
    q1 = df["Weight"].quantile(0.25)
    q3 = df["Weight"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (df["Weight"] < lower) | (df["Weight"] > upper)
    return df.loc[mask, ["Species", "Weight"]]


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика числовых признаков."""
    return df[NUMERIC_COLS].describe().T.round(2)


# === Быстрые тесты (запуск через pytest или python -m data_check) ===
def test_load_data():
    df = load_data()
    assert len(df) == 159, f"Ожидали 159 строк, получили {len(df)}"
    assert df.shape[1] == 7, f"Ожидали 7 столбцов, получили {df.shape[1]}"


def test_no_missing():
    df = load_data()
    assert df.isna().sum().sum() == 0, "Обнаружены пропуски"


def test_no_full_duplicates():
    df = load_data()
    assert df.duplicated().sum() == 0, "Обнаружены полные дубликаты"


def test_species_values():
    df = load_data()
    actual = set(df["Species"].unique())
    assert actual == EXPECTED_SPECIES, f"Виды не совпадают: {actual ^ EXPECTED_SPECIES}"


def test_length_consistency():
    df = load_data()
    bad = check_length_consistency(df)
    assert bad.empty, f"Нарушен порядок длин в {len(bad)} строках"


if __name__ == "__main__":
    print("=== Запуск быстрых тестов data_check ===")
    test_load_data()
    test_no_missing()
    test_no_full_duplicates()
    test_species_values()
    test_length_consistency()
    print("✅ Все тесты прошли")

    print("\n=== Отчёт о качестве ===")
    df = load_data()
    print("\nСхема:")
    print(check_schema(df))
    print("\nПоложительность:")
    print(check_positive(df))
    print("\nБаланс видов:")
    print(check_species_balance(df))
    print("\nДубликаты:", check_duplicates(df))
    print("\nНарушения порядка длин:")
    print(check_length_consistency(df))
    print("\nПотенциальные выбросы массы:")
    print(check_weight_outliers(df))
    print("\nОписательная статистика:")
    print(describe_numeric(df))