"""
Простейшая модель лидара.
Рассчитывает лидарный сигнал с учётом молекулярного и аэрозольного рассеяния.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from src.atmosphere import get as get_atmosphere


def calculate_signal(z_max, beta_aero=2.0e-6, alpha_aero=1.0e-5):
    """
    Рассчитывает лидарный сигнал до высоты z_max.
    
    Параметры:
        z_max: максимальная высота, м
        beta_aero: коэффициент обратного рассеяния аэрозоля, 1/м (по умолчанию 2e-6)
        alpha_aero: коэффициент экстинкции аэрозоля, 1/м (по умолчанию 1e-5)
    
    Возвращает:
        z_lidar: массив высот, м
        beta_mol: молекулярное обратное рассеяние, 1/м
        tau: оптическая толщина
        P_signal: нормированный лидарный сигнал
    """
    # Сетка высот от 100 м до z_max
    z_lidar = np.linspace(100, z_max, 1000)
    
    # Параметры атмосферы
    T, p, rho = get_atmosphere(z_lidar)
    
    # Параметры молекулярного рассеяния на уровне моря
    beta_mol_sea = 1.2e-6   # Обратное рассеяние молекул, 1/м
    alpha_mol_sea = 1.0e-5  # Экстинкция молекул, 1/м
    
    # Молекулярные коэффициенты пропорциональны плотности воздуха
    # Плотность на уровне моря ≈ 1.225 кг/м³
    beta_mol = beta_mol_sea * (rho / 1.225)
    alpha_mol = alpha_mol_sea * (rho / 1.225)
    
    # Суммарные коэффициенты (аэрозоль + молекулы)
    beta_tot = beta_aero + beta_mol
    alpha_tot = alpha_aero + alpha_mol
    
    # Оптическая толщина (интеграл экстинкции от 0 до z)
    tau = cumulative_trapezoid(alpha_tot, z_lidar, initial=0)
    
    # Лидарный сигнал: P(z) ~ (β/z²) * exp(-2τ)
    # - β/z²: обратное рассеяние с геометрическим ослаблением
    # - exp(-2τ): двухпроходное затухание (туда и обратно)
    P_signal = (beta_tot / (z_lidar**2)) * np.exp(-2 * tau)
    
    # Нормируем на максимум для наглядности
    P_signal = P_signal / np.max(P_signal)
    
    return z_lidar, beta_mol, tau, P_signal