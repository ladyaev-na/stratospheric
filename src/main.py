"""
Главный скрипт симулятора аэрозольного зонда.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

from src.atmosphere import get as get_atmosphere
from src import balloon
from src.wind import get as get_wind
from src.lidar import calculate_signal as calc_lidar_signal, WAVELENGTHS, aerosol_profile
from src.inversion import klett_inversion, compare_profiles

os.makedirs('results', exist_ok=True)


def dynamics(t, y, balloon_params):
    z, v_z, x, y_hor = y
    z = max(z, 0.0)
    a_z, V = balloon.get_forces(balloon_params, z, v_z)
    u, v, w = get_wind(z)
    return [v_z, a_z, u, v]


def plot_results(t, z, x, y_hor, ceiling_idx, z_lidar, results):
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(t / 60, z / 1000, 'b-', lw=2)
    ax1.axhline(z[ceiling_idx] / 1000, color='r', ls='--', 
                label=f'Потолок: {z[ceiling_idx]/1000:.1f} км')
    ax1.set_xlabel('Время, мин'); ax1.set_ylabel('Высота, км')
    ax1.set_title('1. Высота от времени'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 4, 2)
    z_prof = np.linspace(0, 20000, 400)
    u_prof, v_prof, w_prof = get_wind(z_prof)
    ax2.plot(u_prof, z_prof / 1000, 'r-', lw=2, label='u')
    ax2.plot(v_prof, z_prof / 1000, 'g--', lw=1.5, label='v')
    ax2.set_xlabel('Скорость, м/с'); ax2.set_ylabel('Высота, км')
    ax2.set_title('2. Профиль ветра'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.plot(x / 1000, y_hor / 1000, 'm-', lw=2, label='Траектория')
    ax3.plot(0, 0, 'go', markersize=8, label='Старт')
    ax3.plot(x[ceiling_idx] / 1000, y_hor[ceiling_idx] / 1000, 'ro', markersize=8, label='Потолок')
    ax3.set_xlabel('X, км'); ax3.set_ylabel('Y, км')
    ax3.set_title('3. 2D-траектория дрейфа'); ax3.legend(); ax3.grid(True, alpha=0.3); ax3.axis('equal')
    
    ax4 = plt.subplot(2, 4, 4)
    colors = {'blue': 'b', 'green': 'g', 'red': 'r', 'ir': 'k'}
    for name, data in results.items():
        ax4.plot(data['beta_aero'] * 1e6, z_lidar / 1000, 
                 color=colors[name], lw=2, label=f"{name} ({WAVELENGTHS[name]*1e9:.0f} нм)")
    ax4.set_xlabel('β_aero, 10⁻⁶ м⁻¹'); ax4.set_ylabel('Высота, км')
    ax4.set_title('4. Профиль аэрозоля (4 λ)'); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 4, 5)
    ax5.plot(results['green']['beta_mol'] * 1e6, z_lidar / 1000, 'b-', lw=2)
    ax5.set_xlabel('β_mol, 10⁻⁶ м⁻¹'); ax5.set_ylabel('Высота, км')
    ax5.set_title('5. Молекулярное рассеяние'); ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 4, 6)
    for name, data in results.items():
        ax6.plot(data['tau'], z_lidar / 1000, color=colors[name], lw=2, label=name)
    ax6.set_xlabel('τ'); ax6.set_ylabel('Высота, км')
    ax6.set_title('6. Оптическая толщина'); ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(2, 4, 7)
    for name, data in results.items():
        ax7.semilogx(data['P_signal'], z_lidar / 1000, color=colors[name], lw=2, label=name)
    ax7.set_xlabel('P(z), отн. ед. (log)'); ax7.set_ylabel('Высота, км')
    ax7.set_title('7. Сигнал (чистый)'); ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(2, 4, 8)
    for name, data in results.items():
        ax8.semilogx(data['P_signal_noisy'], z_lidar / 1000, color=colors[name], lw=1.5, label=name, alpha=0.7)
    ax8.set_xlabel('P(z), отн. ед. (log)'); ax8.set_ylabel('Высота, км')
    ax8.set_title('8. Сигнал (с шумом)'); ax8.legend(fontsize=8); ax8.grid(True, alpha=0.3)
    
    plt.suptitle('Этапы 1-4: Баллистика, Ветер, Лидар, Инверсия', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('results/stage1_4_full.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: results/stage1_4_full.png")
    plt.show()


def main():
    print("=" * 50)
    print("СИМУЛЯТОР АЭРОЗОЛЬНОГО ЗОНДА")
    print("=" * 50)
    
    print("\n[1] Создание шара...")
    balloon_params = balloon.create(M_shell=0.60, M_payload=1.50, V_max=10.00, V_0=3.00, k_elastic=500.0)
    
    print("\n[2] Запуск симуляции полёта...")
    y0 = [0.0, 0.0, 0.0, 0.0]
    t_span = (0, 7200)
    t_eval = np.linspace(*t_span, 1000)
    
    sol = solve_ivp(dynamics, t_span, y0, args=(balloon_params,),
                    t_eval=t_eval, method='RK45', max_step=5, rtol=1e-6, atol=1e-8)
    
    t, z, v_z, x, y_hor = sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    ceiling_idx = np.argmax(z)
    
    print(f"\nМаксимальная высота: {z[ceiling_idx]/1000:.2f} км")
    print(f"Время до потолка: {t[ceiling_idx]/60:.1f} мин")
    print(f"Снос: {np.sqrt(x[ceiling_idx]**2 + y_hor[ceiling_idx]**2)/1000:.2f} км")
    
    print("\n[3] Расчёт лидарного сигнала (4 канала)...")
    z_lidar, results = calc_lidar_signal(z[ceiling_idx], add_noise=True, noise_std=0.05)
    
    for name, data in results.items():
        print(f"  {name:6s} ({data['lambda']*1e9:4.0f} нм): max β = {np.max(data['beta_aero'])*1e6:.2f} × 10⁻⁶ м⁻¹")
    
    print("\n[4] Обратная задача (метод Клетта)...")
    beta_true = aerosol_profile(z_lidar)
    lidar_ratio = 30.0
    
    for name, data in results.items():
        beta_recovered = klett_inversion(data['P_signal_raw'], z_lidar, lidar_ratio=lidar_ratio)
        data['beta_recovered'] = beta_recovered
        if name == 'green':
            compare_profiles(beta_true, beta_recovered, z_lidar, 
                           title=f"Восстановление β(z) — {name} ({data['lambda']*1e9:.0f} нм)")
    
    print("\n[5] Построение графиков...")
    plot_results(t, z, x, y_hor, ceiling_idx, z_lidar, results)

# 6. Monte Carlo: сравнение зонда и дирижабля
    print("\n[6] Monte Carlo симуляция (Этап 5)...")
    from src.monte_carlo import monte_carlo_simulation, plot_comparison
    
    errors_balloon, errors_airship = monte_carlo_simulation(n_runs=50)
    plot_comparison(errors_balloon, errors_airship)

    print("\n✓ Готово!")


if __name__ == '__main__':
    main()