"""
Модель резинового метеозонда (баллона).
Рассчитывает параметры шара и силы, действующие на него.
"""

import numpy as np
from src.atmosphere import get as get_atmosphere


def create(M_shell, M_payload, V_max, V_0, Cd=0.45, k_elastic=500.0, verbose=True):
    """
    Создаёт словарь с параметрами шара.
    """
    R_He = 2077.0
    
    T0, p0, _ = get_atmosphere(0)
    M_He = p0 * V_0 / (R_He * T0)
    
    balloon = {
        'M_shell': M_shell,
        'M_payload': M_payload,
        'V_max': V_max,
        'V_0': V_0,
        'Cd': Cd,
        'R_He': R_He,
        'M_He': M_He,
        'k_elastic': k_elastic,
        'burst': False
    }
    
    if verbose:
        print(f"Масса гелия: {M_He:.3f} кг")
        T0, p0, rho0 = get_atmosphere(0)
        F_b = (rho0 - p0/(R_He*T0)) * 9.81 * V_0
        W = (M_shell + M_payload + M_He) * 9.81
        print(f"Подъёмная сила: {F_b:.2f} Н, Вес: {W:.2f} Н")
        if F_b > W:
            print("✓ Шар взлетит!")
        else:
            print("✗ Шар НЕ взлетит!")
    
    return balloon


def get_forces(balloon, z, v_z):
    """
    Рассчитывает силы, действующие на шар на высоте z.
    """
    T, p, rho = get_atmosphere(z)
    
    # Объём гелия по закону идеального газа
    V_He = balloon['M_He'] * balloon['R_He'] * T / p
    
    # Упругость оболочки: давление внутри = внешнее + натяжение резины
    stretch = V_He / balloon['V_0']
    delta_p = balloon['k_elastic'] * max(0, stretch - 1)
    p_inside = p + delta_p
    
    # Пересчитываем объём с учётом упругости
    V_He = balloon['M_He'] * balloon['R_He'] * T / p_inside
    V = min(V_He, balloon['V_max'])
    
    # Проверка разрыва
    if V_He >= balloon['V_max'] and not balloon['burst']:
        balloon['burst'] = True
        #print(f"⚠ Разрыв оболочки на высоте {z/1000:.1f} км!")
    
    # Подъёмная сила
    F_b = (rho - p_inside / (balloon['R_He'] * T)) * 9.81 * V
    
    # Аэродинамическое сопротивление
    R_sph = (3 * V / (4 * np.pi)) ** (1 / 3)
    A_sph = np.pi * R_sph ** 2
    F_d = 0.5 * rho * balloon['Cd'] * A_sph * v_z * abs(v_z)
    
    M_tot = balloon['M_shell'] + balloon['M_payload'] + balloon['M_He']
    a_z = (F_b - M_tot * 9.81 - F_d) / M_tot
    
    return a_z, V