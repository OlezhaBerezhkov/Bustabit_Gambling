import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import powerlaw


def run_sprint_1(bets_path: str, min_bets: int = 50):
    if not os.path.exists(bets_path):
        print(f"Ошибка: Файл {bets_path} не найден!")
        return None

    df = pd.read_csv(bets_path)
    df['CashedOut'] = df['CashedOut'].fillna(0.0)
    df['Profit'] = df['Profit'].fillna(-df['Bet'])

    df['PlayDate'] = pd.to_datetime(df['PlayDate'])
    df = df.sort_values(by=['Username', 'PlayDate']).reset_index(drop=True)

    user_counts = df['Username'].value_counts()
    active_users = user_counts[user_counts >= min_bets].index
    filtered_df = df[df['Username'].isin(active_users)].copy()
    filtered_df['IsWin'] = np.where(filtered_df['CashedOut'] > 0, 1, 0)

    filtered_df['Prev_IsWin'] = filtered_df.groupby('Username')['IsWin'].shift(1)
    test_df = filtered_df.dropna(subset=['Prev_IsWin'])
    contingency_table = pd.crosstab(test_df['Prev_IsWin'], test_df['IsWin'])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    losses = filtered_df[filtered_df['IsWin'] == 0]['Profit'].abs().values
    fit = powerlaw.Fit(losses, discrete=True, verbose=False)

    print("=== ИТОГИ ВЫПОЛНЕНИЯ ШАГА 1 ===")
    print(f"Уникальных активных игроков: {filtered_df['Username'].nunique()}")
    print(f"Итоговое количество транзакций: {filtered_df.shape[0]}")
    print("-" * 30)
    print("МАТРИЦА ПЕРЕХОДОВ ИСХОДОВ (t-1 -> t):")
    print(contingency_table)
    print(f"Статистика Хи-квадрат: {chi2:.4f}")
    print(f"p-value (Марковость): {p_value:.4e}")
    print("-" * 30)
    print(f"Нижняя граница Парето (xmin): {fit.xmin:.2f}")
    print(f"Индекс тяжести хвоста (alpha): {fit.alpha:.4f}")
    print(f"Статистика Колмогорова-Смирнова (D): {fit.power_law.D:.4f}")
    print("===============================")
    return filtered_df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BETS_FILE = os.path.join(BASE_DIR, "data", "bustabit.csv")
    cleaned_data = run_sprint_1(BETS_FILE, min_bets=50)
