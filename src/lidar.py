"""
Многоканальная модель лидара.
Рассчитывает лидарный сигнал для 4 длин волн с учётом синтетических слоёв аэрозоля.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from src.atmosphere import get as get_atmosphere


# Длины волн LED-излучателей, м
WAVELENGTHS = {
    'blue': 405e-9,    # 405 нм
    'green': 532e-9,   # 532 нм
    'red': 650e-9,     # 650 нм
    'ir': 850e-9       # 850 нм
}


def aerosol_profile(z):
    """
    Синтетический профиль аэрозоля (базовый на 532 нм).
    
    Слои:
    - Пограничный слой (0-2 км): высокий аэрозоль, экспоненциальное затухание
    - Свободная тропосфера (2-10 км): низкий фон
    - Слой пыли (3-4 км): гауссов пик
    - Стратосферный аэрозоль (15-20 км): тонкий слой
    """
    beta = np.zeros_like(z, dtype=float)
    
    # Пограничный слой (экспоненциальное затухание)
    beta += 5e-6 * np.exp(-z / 1500)
    
    # Слой пыли на 3-4 км (гауссов пик)
    beta += 1e-5 * np.exp(-((z - 3500) / 500)**2)
    
    # Стратосферный аэрозоль на 18 км (тонкий слой)
    beta += 5e-7 * np.exp(-((z - 18000) / 1000)**2)
    
    return beta


def beta_aero_wavelength(z, lambda_m, alpha_angstrom=1.5):
    """
    Коэффициент обратного рассеяния аэрозоля на длине волны lambda_m.
    
    Параметры:
        z: высота, м
        lambda_m: длина волны, м
        alpha_angstrom: ангарстремовский показатель (1.5 для мелкого аэрозоля)
    
    Возвращает:
        beta: коэффициент обратного рассеяния, 1/м
    """
    beta_ref = aerosol_profile(z)  # базовый профиль на 532 нм
    lambda_ref = 532e-9
    return beta_ref * (lambda_m / lambda_ref) ** (-alpha_angstrom)


def calculate_signal(z_max, add_noise=False, noise_std=0.05):
    """
    Рассчитывает лидарный сигнал для 4 длин волн.
    
    Возвращает:
        z_lidar: массив высот, м
        results: словарь с результатами для каждой длины волны
    """
    z_lidar = np.linspace(100, z_max, 1000)
    T, p, rho = get_atmosphere(z_lidar)
    
    beta_mol_sea = 1.2e-6
    alpha_mol_sea = 1.0e-5
    
    beta_mol = beta_mol_sea * (rho / 1.225)
    alpha_mol = alpha_mol_sea * (rho / 1.225)
    
    results = {}
    
    for name, lambda_m in WAVELENGTHS.items():
        beta_aero = beta_aero_wavelength(z_lidar, lambda_m, alpha_angstrom=1.5)
        alpha_aero = 10 * beta_aero
        
        beta_tot = beta_aero + beta_mol
        alpha_tot = alpha_aero + alpha_mol
        
        tau = cumulative_trapezoid(alpha_tot, z_lidar, initial=0)
        
        # Ненормированный сигнал (для инверсии Клетта)
        P_signal_raw = (beta_tot / (z_lidar**2)) * np.exp(-2 * tau)
        
        # Нормированный сигнал (для визуализации)
        P_signal = P_signal_raw / np.max(P_signal_raw)
        
        # Добавляем шум
        if add_noise:
            P_signal_noisy = P_signal + np.random.normal(0, noise_std, size=P_signal.shape)
            P_signal_noisy = np.maximum(P_signal_noisy, 0)
        else:
            P_signal_noisy = P_signal
        
        results[name] = {
            'beta_aero': beta_aero,
            'beta_mol': beta_mol,
            'tau': tau,
            'P_signal': P_signal,
            'P_signal_raw': P_signal_raw,  # Ненормированный для инверсии
            'P_signal_noisy': P_signal_noisy,
            'lambda': lambda_m
        }
    
    return z_lidar, results