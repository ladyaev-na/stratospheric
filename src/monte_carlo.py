"""
Этап 5: Monte Carlo симуляция.
"""

import numpy as np
from scipy.integrate import solve_ivp
from src import balloon
from src.wind import get as get_wind
from src.lidar import calculate_signal as calc_lidar_signal, aerosol_profile
from src.inversion import klett_inversion


def run_balloon_simulation(wind_type='standard'):
    """Симуляция полёта зонда."""
    balloon_params = balloon.create(M_shell=0.60, M_payload=1.50, 
                                     V_max=10.00, V_0=3.00, k_elastic=500.0, verbose=False)
    
    def dynamics(t, y):
        z, v_z, x, y_hor = y
        z = max(z, 0.0)
        a_z, V = balloon.get_forces(balloon_params, z, v_z)
        u, v, w = get_wind(z, wind_type=wind_type)
        return [v_z, a_z, u, v]
    
    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(dynamics, (0, 7200), y0, t_eval=np.linspace(0, 7200, 500),
                    method='RK45', max_step=5, rtol=1e-6, atol=1e-8)
    
    return sol.y[0], sol.y[2], sol.y[3]


def run_airship_simulation():
    """Симуляция полёта дирижабля (без горизонтального сноса)."""
    balloon_params = balloon.create(M_shell=0.60, M_payload=1.50, 
                                     V_max=10.00, V_0=3.00, k_elastic=500.0, verbose=False)
    
    def dynamics(t, y):
        z, v_z = y
        z = max(z, 0.0)
        a_z, V = balloon.get_forces(balloon_params, z, v_z)
        return [v_z, a_z]
    
    y0 = [0.0, 0.0]
    sol = solve_ivp(dynamics, (0, 7200), y0, t_eval=np.linspace(0, 7200, 500),
                    method='RK45', max_step=5, rtol=1e-6, atol=1e-8)
    
    return sol.y[0]


def calculate_recovery_error(z_max, is_airship=False):
    """
    Рассчитывает ошибку восстановления аэрозоля.
    
    Для зонда: добавляем искажение из-за сноса ветром
    Для дирижабля: используем идеальный сигнал
    """
    z_lidar, results = calc_lidar_signal(z_max, add_noise=False)
    beta_true = aerosol_profile(z_lidar)
    
    # Восстанавливаем профиль
    beta_recovered = klett_inversion(results['green']['P_signal_raw'], z_lidar, lidar_ratio=30.0)
    
    # Для зонда добавляем искажение (имитация влияния дрейфа)
    if not is_airship:
        # Зонд сносится ветром, измеряет разные воздушные массы
        # Имитируем это как случайное искажение профиля
        distortion = 1.0 + 1.5 * np.random.randn(len(z_lidar))
        distortion = np.maximum(distortion, 0.3)  # не меньше 30%
        beta_recovered = beta_recovered * distortion
    
    # Рассчитываем ошибку (без ограничения)
    mask = (beta_true > 1e-8) & (beta_true < 1e-4)
    if np.sum(mask) > 10:
        error = np.mean(np.abs(beta_recovered[mask] - beta_true[mask]) / beta_true[mask])
    else:
        error = 0.0
    
    return error


def monte_carlo_simulation(n_runs=50):
    """Monte Carlo симуляция."""
    print(f"\nЗапуск Monte Carlo ({n_runs} симуляций)...")
    
    errors_balloon = []
    errors_airship = []
    
    for i in range(n_runs):
        wind_type = np.random.choice(['weak', 'standard', 'strong'])
        
        # Зонд
        z_balloon, x, y_hor = run_balloon_simulation(wind_type)
        z_max_balloon = np.max(z_balloon)
        error_balloon = calculate_recovery_error(z_max_balloon, is_airship=False)
        errors_balloon.append(error_balloon)
        
        # Дирижабль
        z_airship = run_airship_simulation()
        z_max_airship = np.max(z_airship)
        error_airship = calculate_recovery_error(z_max_airship, is_airship=True)
        errors_airship.append(error_airship)
        
        if (i + 1) % 10 == 0:
            print(f"  Выполнено {i+1}/{n_runs} симуляций...")
    
    return np.array(errors_balloon), np.array(errors_airship)


def plot_comparison(errors_balloon, errors_airship):
    """Строит график сравнения ошибок."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма
    axes[0].hist(errors_balloon * 100, bins=20, alpha=0.7, label='Зонд', color='red')
    axes[0].hist(errors_airship * 100, bins=20, alpha=0.7, label='Дирижабль', color='blue')
    axes[0].set_xlabel('Ошибка восстановления, %', fontsize=12)
    axes[0].set_ylabel('Количество запусков', fontsize=12)
    axes[0].set_title('Распределение ошибок (Monte Carlo)', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Статистика
    categories = ['Средняя', 'Медианная', 'Максимальная']
    x = np.arange(len(categories))
    width = 0.35
    
    stats_balloon = [np.mean(errors_balloon)*100, np.median(errors_balloon)*100, np.max(errors_balloon)*100]
    stats_airship = [np.mean(errors_airship)*100, np.median(errors_airship)*100, np.max(errors_airship)*100]
    
    axes[1].bar(x - width/2, stats_balloon, width, label='Зонд', color='red')
    axes[1].bar(x + width/2, stats_airship, width, label='Дирижабль', color='blue')
    axes[1].set_xlabel('Статистика', fontsize=12)
    axes[1].set_ylabel('Ошибка, %', fontsize=12)
    axes[1].set_title('Сравнение ошибок', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/stage5_monte_carlo.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен: results/stage5_monte_carlo.png")
    plt.show()
    
    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТЫ MONTE CARLO ({len(errors_balloon)} запусков)")
    print(f"{'='*50}")
    print(f"Зонд:       средняя = {np.mean(errors_balloon)*100:.1f}%, медиана = {np.median(errors_balloon)*100:.1f}%")
    print(f"Дирижабль:  средняя = {np.mean(errors_airship)*100:.1f}%, медиана = {np.median(errors_airship)*100:.1f}%")
    print(f"{'='*50}")