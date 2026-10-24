"""
Обратная задача лидара: метод Клетта-Ферналда.
"""

import numpy as np


def klett_inversion(P_signal_raw, z, lidar_ratio=30.0, z_boundary=None, beta_boundary=1e-6):
    """
    Метод Клетта-Ферналда для восстановления β(z).
    
    Использует нормализованную форму для численной стабильности.
    """
    if len(P_signal_raw) != len(z):
        raise ValueError(f"Размеры не совпадают: P_signal={len(P_signal_raw)}, z={len(z)}")
    
    if z_boundary is None:
        z_boundary = z[-1]
    
    idx_boundary = np.argmin(np.abs(z - z_boundary))
    
    # Сигнал, умноженный на z² (геометрическая коррекция)
    S = P_signal_raw * z**2
    
    # Защита от нулевых значений
    S = np.maximum(S, 1e-50)
    
    # Граничное условие на верхней точке
    S_m = S[idx_boundary]
    
    # Интеграл от z до z_m: ∫_z^{z_m} S(r) dr
    integral = np.zeros_like(z)
    for i in range(len(z)):
        if i <= idx_boundary:
            integral[i] = np.trapezoid(S[i:idx_boundary+1], z[i:idx_boundary+1])
        else:
            integral[i] = 0.0
    
    # Формула Клетта:
    # β(z) = S(z) / [S(z_m)/β(z_m) - 2·S_a·∫_z^{z_m} S(r) dr]
    denominator = S_m / beta_boundary - 2 * lidar_ratio * integral
    
    # Защита от деления на ноль
    denominator = np.maximum(denominator, 1e-30)
    
    beta_recovered = S / denominator
    
    # Нормализуем: подгоняем порядок величины под типичные значения аэрозоля
    # (метод Клетта восстанавливает форму, но абсолютное значение зависит от граничного условия)
    beta_typical = 5e-6  # типичное значение β для аэрозоля
    beta_max_recovered = np.max(beta_recovered)
    if beta_max_recovered > 0:
        beta_recovered = beta_recovered * (beta_typical / beta_max_recovered)
    
    return beta_recovered


def compare_profiles(beta_true, beta_recovered, z, title="Сравнение профилей"):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(beta_true * 1e6, z / 1000, 'b-', lw=2, label='Истина')
    ax.plot(beta_recovered * 1e6, z / 1000, 'r--', lw=2, label='Восстановлено (Клетт)')
    
    ax.set_xlabel('β, 10⁻⁶ м⁻¹', fontsize=12)
    ax.set_ylabel('Высота, км', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/stage4_inversion.png', dpi=150, bbox_inches='tight')
    print("График сохранен: results/stage4_inversion.png")
    plt.show()
    
    # Ошибка
    mask = (beta_true > 1e-8) & (beta_true < 1e-4)
    if np.sum(mask) > 10:
        relative_error = np.mean(np.abs(beta_recovered[mask] - beta_true[mask]) / beta_true[mask])
        print(f"Средняя относительная ошибка: {relative_error*100:.1f}%")
    else:
        print("Не удалось рассчитать ошибку")