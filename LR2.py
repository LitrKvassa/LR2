import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import click
import sys

N = 100
G = 1.0
dt = 0.05
num_steps = 1000

cm = 10.0
m_min = 0.1
m_max = 10.0

x_min, x_max = -50.0, 50.0
y_min, y_max = -50.0, 50.0

def draw_grid(ax):
    x_lines = np.linspace(x_min, x_max, grid + 1)
    y_lines = np.linspace(y_min, y_max, grid + 1)
    for i in x_lines:
        ax.axvline(i, color='#555555', linewidth=0.25, zorder=1)
    for i in y_lines:
        ax.axhline(i, color='#555555', linewidth=0.25, zorder=1)

class quadrant:
    def __init__(self, x, y, s):
        self.x = x  
        self.y = y  
        self.s = s
        self.m = 0.0
        self.com = np.zeros(2)
        self.c = None
        self.part_idx = None

    def insert(self, idx, pos, m):
        if self.m == 0.0:
            self.part_idx = idx
            self.m = m[idx, 0]
            self.com = pos[idx].copy()
            return
        if self.c is None:
            self.subdivide()
            old_idx = self.part_idx
            self.part_idx = None
            if old_idx is not None:
                self.insc(old_idx, pos, m)
        self.insc(idx, pos, m)
        new_m = self.m + m[idx, 0]
        self.com = (self.com * self.m + pos[idx] * m[idx, 0]) / new_m
        self.m = new_m

    def subdivide(self):
        hs = self.s / 4.0
        s = self.s / 2.0
        self.c = [
            quadrant(self.x - hs, self.y - hs, s),
            quadrant(self.x + hs, self.y - hs, s),
            quadrant(self.x - hs, self.y + hs, s),
            quadrant(self.x + hs, self.y + hs, s)
        ]

    def insc(self, idx, pos, m):
        px, py = pos[idx]
        sw = 0 if px < self.x else 1
        ne = 0 if py < self.y else 2
        self.c[sw + ne].insert(idx, pos, m)

    def get_a(self, p_idx, pos, theta):
        if self.m == 0.0 or self.part_idx == p_idx:
            return np.zeros(2)
        rij = self.com - pos[p_idx]
        d = np.sqrt(np.sum(rij**2) + sooth**2)
        if self.c is None or (self.s / d) < theta:
            inv_d3 = 1.0 / (d**3)
            return G * self.m * rij * inv_d3
        else:
            a = np.zeros(2)
            for i in self.c:
                a += i.get_a(p_idx, pos, theta)
            return a

def calc_bh(p, m, theta=.5):
    min_p = np.min(p, axis=0) - 2.0
    max_p = np.max(p, axis=0) + 2.0
    s = max(np.max(max_p - min_p), 40.0)
    center = min_p + (max_p - min_p) / 2.0
    root = quadrant(center[0], center[1], s)
    for i in range(len(p)):
        root.insert(i, p, m)
    a = np.zeros_like(p)
    for i in range(len(p)):
        a[i] = root.get_a(i, p, theta)
    return a

def calc_pm(p, m, dx, dy):
    rho = np.zeros((grid, grid))
    for i in range(len(p)):
        if np.isnan(p[i, 0]) or np.isnan(p[i, 1]):
            continue
        if p[i, 0] < x_min or p[i, 0] >= x_max or p[i, 1] < y_min or p[i, 1] >= y_max:
            continue
        
        gx = (p[i, 0] - x_min) / dx - 0.5
        gy = (p[i, 1] - y_min) / dy - 0.5
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        tx = gx - ix
        ty = gy - iy
        
        for nx, weight_x in [(0, 1 - tx), (1, tx)]:
            for ny, weight_y in [(0, 1 - ty), (1, ty)]:
                idx_x = (ix + nx) % grid
                idx_y = (iy + ny) % grid
                rho[idx_x, idx_y] += m[i, 0] * weight_x * weight_y

    U = np.zeros((grid, grid))
    x_g, y_g = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
    x_coords = x_min + (x_g + 0.5) * dx
    y_coords = y_min + (y_g + 0.5) * dy
    
    for ix in range(grid):
        for iy in range(grid):
            if rho[ix, iy] > 0:
                rx = x_coords - x_coords[ix, iy]
                ry = y_coords - y_coords[ix, iy]
                r_sq = rx**2 + ry**2 + sooth**2
                U -= G * rho[ix, iy] / np.sqrt(r_sq)

    F_x, F_y = np.gradient(-U)
    F_x /= dx
    F_y /= dy

    a = np.zeros_like(p)
    for i in range(len(p)):
        if np.isnan(p[i, 0]) or np.isnan(p[i, 1]):
            continue
        if p[i, 0] < x_min or p[i, 0] >= x_max or p[i, 1] < y_min or p[i, 1] >= y_max:
            continue
        gx = (p[i, 0] - x_min) / dx - 0.5
        gy = (p[i, 1] - y_min) / dy - 0.5
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        tx = gx - ix
        ty = gy - iy
        
        for nx, weight_x in [(0, 1 - tx), (1, tx)]:
            for ny, weight_y in [(0, 1 - ty), (1, ty)]:
                idx_x = (ix + nx) % grid
                idx_y = (iy + ny) % grid
                a[i, 0] += F_x[idx_x, idx_y] * weight_x * weight_y
                a[i, 1] += F_y[idx_x, idx_y] * weight_x * weight_y
    return a

def run_c_calc():
    global fin_pos_bh, fin_pos_pm
    
    pos_bh = init_pos.copy()
    v_bh = init_v.copy()
    pos_pm = init_pos.copy()
    v_pm = init_v.copy()
    m = init_m.copy()
    
    dx = (x_max - x_min) / grid
    dy = (y_max - y_min) / grid

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    colors_bh = np.array(['blue'] * (N + 1), dtype=object)
    colors_pm = np.array(['red'] * (N + 1), dtype=object)
    colors_bh[0] = 'white'
    colors_pm[0] = 'white'
    
    sizes = 6 + m.flatten() * 5
    sizes[0] = 100
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title("Метод PP")
    ax1.set_facecolor('#111111')
    draw_grid(ax1)
    scatter_bh = ax1.scatter(pos_bh[:, 0], pos_bh[:, 1], s=sizes, c=colors_bh, zorder=3)

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("Метод PM")
    ax2.set_facecolor('#111111')
    draw_grid(ax2)
    scatter_pm = ax2.scatter(pos_pm[:, 0], pos_pm[:, 1], s=sizes, c=colors_pm, zorder=3)
    
    text_x = x_min + (x_max - x_min) * 0.05
    text_y = y_max - (y_max - y_min) * 0.08
    
    step_text_bh = ax1.text(text_x, text_y, '', color='white', fontsize=8, zorder=4)
    step_text_pm = ax2.text(text_x, text_y, '', color='white', fontsize=8, zorder=4)
    
    plt.tight_layout()

    a_bh = calc_bh(pos_bh, m)

    a_pm = calc_pm(pos_pm, m, dx, dy)
    
    cen_a_bh = np.zeros(2)
    cen_a_pm = np.zeros(2)

    def update(frame):
        nonlocal pos_bh, v_bh, a_bh, pos_pm, v_pm, a_pm, cen_a_bh, cen_a_pm

        v_half_bh = v_bh + a_bh * (dt / 2.0)
        pos_bh += v_half_bh * dt

        pos_bh[0] = [0.0, 0.0]
        v_half_bh[0] = [0.0, 0.0]
        
        a_bh = calc_bh(pos_bh, m)
        v_bh = v_half_bh + a_bh * (dt / 2.0)
        v_bh[0] = [0.0, 0.0]
        cen_a_bh = a_bh[0].copy()

        v_half_pm = v_pm + a_pm * (dt / 2.0)
        pos_pm += v_half_pm * dt

        pos_pm[0] = [0.0, 0.0]
        v_half_pm[0] = [0.0, 0.0]
        
        a_pm = calc_pm(pos_pm, m, dx, dy)
        v_pm = v_half_pm + a_pm * (dt / 2.0)
        v_pm[0] = [0.0, 0.0]
        cen_a_pm = a_pm[0].copy()

        scatter_bh.set_offsets(pos_bh)
        scatter_pm.set_offsets(pos_pm)
        scatter_bh.set_sizes(m.flatten()*8)
        scatter_pm.set_sizes(m.flatten()*8)
        
        step_text_bh.set_text(f"Шаг: {frame}")
        step_text_pm.set_text(f"Шаг: {frame}")

        return scatter_bh, scatter_pm, step_text_bh, step_text_pm

    ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)
    ani.save("animation.gif", writer=PillowWriter(fps=20))
    
    fin_pos_bh = pos_bh.copy()
    fin_pos_pm = pos_pm.copy()
    
    F_bh = cm * cen_a_bh
    F_pm = cm * cen_a_pm

    F_bh_m = np.sqrt(np.sum(F_bh**2))
    F_pm_m = np.sqrt(np.sum(F_pm**2))
    
    click.secho("Результирующая сила на ЦТ:")
    click.echo(f"Метод PP:\n\tВектор силы = [{F_bh[0]:.2f}, {F_bh[1]:.2f}],\n\tМодуль |F| = {F_bh_m:.3f}")
    click.echo(f"Метод PM:\n\tВектор силы = [{F_pm[0]:.2f}, {F_pm[1]:.2f}],\n\tМодуль |F| = {F_pm_m:.3f}")

    plt.close(fig)

def exec_calc():
    global init_m, init_pos, init_v, grid, sooth

    width_x = x_max - x_min
    width_y = y_max - y_min
    max_side = max(width_x, width_y)
    
    opt_cell_size = 1.25
    grid = int(np.clip(max_side / opt_cell_size, 16, 64))

    m_par = np.random.uniform(m_min, m_max, size=(N, 1))
    pos_par = np.random.uniform(x_min / 1.1, x_max / 1.1, size=(N, 2))
    v_par = np.zeros((N, 2))
    
    for i in range(N):
        x, y = pos_par[i]
        r = np.sqrt(x**2 + y**2)
        #защита от деления на ноль если частица ровно в центре
        if r < 0.5:
            r = 0.5
        v_ = np.sqrt(G * cm / r) * np.random.uniform(0.9, 1.1)
        #придание нач скорости и небольшого вращения
        v_par[i, 0] = -y / r * v_  #vx
        v_par[i, 1] =  x / r * v_  #vy
    
    init_m = np.zeros((N + 1, 1))
    init_pos = np.zeros((N + 1, 2))
    init_v = np.zeros((N + 1, 2))
    
    init_m[0, 0] = cm
    init_pos[0] = [0.0, 0.0]
    init_v[0] = [0.0, 0.0]
    
    init_m[1:] = m_par
    init_pos[1:] = pos_par
    init_v[1:] = v_par

    dx = width_x / grid
    dy = width_y / grid
    cell_size = max(dx, dy)
    v_scale = np.max(np.abs(init_v)) * dt * 4.0
    sooth = max(cell_size * 0.25, v_scale, 0.15)

    run_c_calc()
    show_plot()
    click.secho("\nРезультат сохранён как 'animation.gif'.\n")

def show_plot():
    fig_init, (ax_init1, ax_init2) = plt.subplots(1, 2, figsize=(12, 6))
    
    colors_bh = np.array(['blue'] * (N + 1), dtype=object)
    colors_pm = np.array(['red'] * (N + 1), dtype=object)
    colors_bh[0] = 'white'
    colors_pm[0] = 'white'
    
    sizes = 6 + init_m.flatten() * 5
    sizes[0] = 5
    
    ax_init1.set_xlim(x_min, x_max)
    ax_init1.set_ylim(y_min, y_max)
    ax_init1.set_title("Шаг 0: метод PP")
    ax_init1.set_facecolor('#111111')
    draw_grid(ax_init1)
    ax_init1.scatter(init_pos[:, 0], init_pos[:, 1], 
                     s=sizes, c=colors_bh, zorder=3)
    
    ax_init2.set_xlim(x_min, x_max)
    ax_init2.set_ylim(y_min, y_max)
    ax_init2.set_title("Шаг 0: метод PM")
    ax_init2.set_facecolor('#111111')
    draw_grid(ax_init2)
    ax_init2.scatter(init_pos[:, 0], init_pos[:, 1], 
                     s=sizes, c=colors_pm, zorder=3)
    
    plt.tight_layout()
    plt.show()
    
    fig_final, (ax_fin1, ax_fin2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax_fin1.set_xlim(x_min, x_max)
    ax_fin1.set_ylim(y_min, y_max)
    ax_fin1.set_title(f"Шаг {num_steps}: метод PP")
    ax_fin1.set_facecolor('#111111')
    draw_grid(ax_fin1)
    ax_fin1.scatter(fin_pos_bh[:, 0], fin_pos_bh[:, 1],
                    s=init_m.flatten()*8, c=colors_bh, zorder=3)
    ax_fin2.set_xlim(x_min, x_max)
    ax_fin2.set_ylim(y_min, y_max)
    ax_fin2.set_title(f"Шаг {num_steps}: метод PM")
    ax_fin2.set_facecolor('#111111')
    draw_grid(ax_fin2)
    ax_fin2.scatter(fin_pos_pm[:, 0], fin_pos_pm[:, 1],
                    s=init_m.flatten()*8, c=colors_pm, zorder=3)
    plt.tight_layout()
    plt.show()

def outputf():
    click.clear()
    click.echo("\t\t<Текущие параметры>")
    click.echo(f"Количество тел:\t\t{N}")
    click.echo(f"Шаг по времени:\t\t{dt}")
    click.echo(f"Количество шагов:\t{num_steps}")
    click.echo(f"Масса центрального тела:\t\t{cm}")
    click.echo(f"Диапазон масс частиц:\t\t[{m_min} : {m_max}]")
    click.echo(f"Границы X:\t\t\t[{x_min} : {x_max}]")
    click.echo(f"Границы Y:\t\t\t[{y_min} : {y_max}]")

def inputf():
    click.clear()
    global N, dt, num_steps, sooth, cm, m_min, m_max, x_min, x_max, y_min, y_max
    click.echo("Ввод параметров\n\t<ENTER без ввода значения чтобы оставить текущее значение>")
    new_N = click.prompt("Количество тел, int", default=N, type=click.IntRange(min=1))
    new_dt = click.prompt("Шаг по времени, float", default=dt, type=click.FloatRange(min=1e-6))
    new_num_steps = click.prompt("Количество кадров, int", default=num_steps, type=click.IntRange(min=1))
    new_center_mass = click.prompt("Масса центрального тела, float", default=cm, type=click.FloatRange(min=0.0))
    
    while True:
        new_mass_min = click.prompt("Минимальная масса частиц (m_min)", default=m_min, type=click.FloatRange(min=1e-1))
        new_mass_max = click.prompt("Максимальная масса частиц (m_max)", default=m_max, type=click.FloatRange(min=1e-1))
        if new_mass_max >= new_mass_min:
            break
        click.secho("m_max должно быть больше или равно m_min", fg="red")
        
    while True:
        new_xmin = click.prompt("Минимум по X (x_min), float", default=x_min, type=float)
        new_xmax = click.prompt("Максимум по X (x_max), float", default=x_max, type=float)
        if new_xmax > new_xmin:
            break
        click.secho("x_max должно быть больше x_min")
        
    while True:
        new_ymin = click.prompt("Минимум по Y (y_min), float", default=y_min, type=float)
        new_ymax = click.prompt("Максимум по Y (y_max), float", default=y_max, type=float)
        if new_ymax > new_ymin:
            break
        click.secho("y_max должно быть больше y_min")
        
    N, dt, num_steps, cm = new_N, new_dt, new_num_steps, new_center_mass
    m_min, m_max = new_mass_min, new_mass_max
    x_min, x_max, y_min, y_max = new_xmin, new_xmax, new_ymin, new_ymax
    click.secho("\n<Данные обновлены>")

@click.command()
def main():
    click.clear()
    while True:
        click.echo("\t\t<Меню>")
        click.echo("1. Текущие параметры")
        click.echo("2. Ввод параметров")
        click.echo("3. Расчёт")
        click.echo("4. Выход")
        choice = click.prompt("<Ввод>", type=click.Choice(['1', '2', '3', '4']), show_choices=False)
        if choice == '1':
            outputf()
            input("<ENTER для возврата в меню>")
            click.clear()
        elif choice == '2':
            inputf()
            input("<ENTER для возврата в меню>")
            click.clear()
        elif choice == '3':
            exec_calc()
            input("<ENTER для возврата в меню>")
            click.clear()
        elif choice == '4':
            click.secho("\nРабота программы завершена")
            sys.exit(0)

if __name__ == "__main__":
    main()