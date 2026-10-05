"""
Главный скрипт симулятора аэрозольного зонда.
Собирает все модули вместе и запускает симуляцию.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Импортируем наши модули
from src.atmosphere import get as get_atmosphere
from src import balloon
from src.wind import get as get_wind
from src.lidar import calculate_signal as calc_lidar_signal

# Создаём папку для результатов
os.makedirs('results', exist_ok=True)


# ==========================================================
# ОДУ ДВИЖЕНИЯ
# ==========================================================
def dynamics(t, y, balloon_params):
    """
    Функция описывает систему дифференциальных уравнений движения шара.
    Вызывается solve_ivp многократно для интегрирования.
    
    Параметры:
        t: текущее время, с
        y: вектор состояния [z, v_z, x, y_hor]
        balloon_params: словарь с параметрами шара
    
    Возвращает:
        [v_z, a_z, u, v]: производные по времени
    """
    z, v_z, x, y_hor = y
    z = max(z, 0.0)  # Защита от ухода под землю
    
    # Рассчитываем силы на шаре
    a_z, V = balloon.get_forces(balloon_params, z, v_z)
    
    # Получаем ветер на текущей высоте
    u, v = get_wind(z)
    
    return [v_z, a_z, u, v]


# ==========================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================================
def plot_results(t, z, x, y_hor, ceiling_idx, z_lidar, beta_mol, tau, P_signal):
    """Строит 6 графиков результатов симуляции"""
    fig = plt.figure(figsize=(15, 10))
    
    # График 1: Высота от времени
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(t / 60, z / 1000, 'b-', lw=2)
    ax1.axhline(z[ceiling_idx] / 1000, color='r', ls='--', 
                label=f'Потолок: {z[ceiling_idx]/1000:.1f} км')
    ax1.set_xlabel('Время, мин')
    ax1.set_ylabel('Высота, км')
    ax1.set_title('1. Высота от времени')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Профиль ветра
    ax2 = plt.subplot(2, 3, 2)
    z_prof = np.linspace(0, 20000, 400)
    u_prof, _ = get_wind(z_prof)
    ax2.plot(u_prof, z_prof / 1000, 'r-', lw=2)
    ax2.set_xlabel('Скорость ветра, м/с')
    ax2.set_ylabel('Высота, км')
    ax2.set_title('2. Профиль ветра')
    ax2.grid(True, alpha=0.3)
    
    # График 3: 2D-траектория дрейфа
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(x / 1000, y_hor / 1000, 'm-', lw=2, label='Траектория')
    ax3.plot(0, 0, 'go', markersize=8, label='Старт')
    ax3.plot(x[ceiling_idx] / 1000, y_hor[ceiling_idx] / 1000, 
             'ro', markersize=8, label='Потолок')
    ax3.set_xlabel('X, км')
    ax3.set_ylabel('Y, км')
    ax3.set_title('3. 2D-траектория дрейфа')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # График 4: Коэффициент обратного рассеяния
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(beta_mol * 1e6, z_lidar / 1000, 'b-', lw=2, label='Молекулы')
    ax4.set_xlabel('β, 10⁻⁶ м⁻¹')
    ax4.set_ylabel('Высота, км')
    ax4.set_title('4. Коэф. обратного рассеяния')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # График 5: Оптическая толщина
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(tau, z_lidar / 1000, 'g-', lw=2)
    ax5.set_xlabel('Оптическая толщина τ')
    ax5.set_ylabel('Высота, км')
    ax5.set_title('5. Затухание сигнала (τ)')
    ax5.grid(True, alpha=0.3)
    
    # График 6: Лидарный сигнал
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(P_signal, z_lidar / 1000, 'k-', lw=2)
    ax6.set_xscale('log')
    ax6.set_xlabel('Сигнал P(z), отн. ед. (log)')
    ax6.set_ylabel('Высота, км')
    ax6.set_title('6. Лидарный сигнал')
    ax6.grid(True, alpha=0.3)
    
    # Общий заголовок
    plt.suptitle('Этап 1: Баллистика, Дрейф и Лидар', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Сохраняем и показываем
    plt.savefig('results/stage1_modular.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: results/stage1_modular.png")
    plt.show()


# ==========================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ==========================================================
def main():
    """Точка входа в программу"""
    print("=" * 50)
    print("СИМУЛЯТОР АЭРОЗОЛЬНОГО ЗОНДА")
    print("=" * 50)
    
    # 1. Создаём шар с грузом 1.5 кг
    print("\n[1] Создание шара...")
    balloon_params = balloon.create(
        M_shell=0.60,
        M_payload=1.50,
        V_max=10.00,
        V_0=3.00
    )
    
    # 2. Запускаем симуляцию полёта
    print("\n[2] Запуск симуляции полёта...")
    y0 = [0.0, 0.0, 0.0, 0.0]  # Начальные условия: [z, v_z, x, y_hor]
    t_span = (0, 7200)          # 2 часа
    t_eval = np.linspace(*t_span, 1000)
    
    # Решаем ОДУ
    sol = solve_ivp(
        dynamics,
        t_span,
        y0,
        args=(balloon_params,),  # Передаём параметры шара в функцию dynamics
        t_eval=t_eval,
        method='RK45',
        max_step=5,
        rtol=1e-6,
        atol=1e-8
    )
    
    # Извлекаем результаты
    t = sol.t
    z = sol.y[0]
    v_z = sol.y[1]
    x = sol.y[2]
    y_hor = sol.y[3]
    
    # Находим потолок
    ceiling_idx = np.argmax(z)
    print(f"\nМаксимальная высота: {z[ceiling_idx]/1000:.2f} км")
    print(f"Время до потолка: {t[ceiling_idx]/60:.1f} мин")
    print(f"Снос по горизонтали: {np.sqrt(x[ceiling_idx]**2 + y_hor[ceiling_idx]**2)/1000:.2f} км")
    
    # 3. Рассчитываем лидарный сигнал
    print("\n[3] Расчёт лидарного сигнала...")
    z_lidar, beta_mol, tau, P_signal = calc_lidar_signal(z[ceiling_idx])
    
    # 4. Строим графики
    print("\n[4] Построение графиков...")
    plot_results(t, z, x, y_hor, ceiling_idx, z_lidar, beta_mol, tau, P_signal)
    
    print("\n✓ Готово!")


# Запуск программы
if __name__ == '__main__':
    main()