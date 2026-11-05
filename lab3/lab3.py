#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
#ex1
N = 8
F1 = np.zeros((N, N), dtype=complex)
for m in range(N):
    for k in range(N):
        F1[m, k] = np.exp(-2j * np.pi * m * k / N)

n = np.arange(N)
vertical = np.reshape(n, (N, 1))
horizontal = np.reshape(n, (1, N))
ind = vertical * horizontal

F2 = np.exp(-2j * np.pi * ind / N)

print(f'F1 == F2: {np.allclose(F1, F2)}')
FH = F2.conj().T
I = np.eye(N)
print(f'FH @ F == N * I: {np.allclose(FH @ F2, N * I)}')
print(f'||FH @ F - N * I|| = {np.linalg.norm(FH @ F2 - N * I)}')

fig, axs = plt.subplots(N, 2, figsize=(10, 12))
for m in range(N):
    axs[m, 0].plot(np.real(F2[m]))
    axs[m, 1].plot(np.imag(F2[m]))
    axs[m, 0].set_ylabel(f'line {m}')
axs[0, 0].set_title('Real')
axs[0, 1].set_title('Imaginar')
fig.tight_layout()
plt.savefig('./ex1.pdf')
plt.show()

#%%
#ex2 - static
fs = 1000
t = np.arange(fs) / fs
f = 10
x = np.sin(2 * np.pi * f * t)

y = x * np.exp(-2j * np.pi * t)
dist_y = np.abs(y)
k = 620

# Figura 1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(np.arange(fs), x, c=dist_y, cmap='viridis', s=3)
axs[0].plot([k, k], [0, x[k]], 'r', linewidth=2)
axs[0].plot(k, x[k], 'ro')
axs[0].axhline(0, color='k', linewidth=0.5)
axs[0].set_xlabel('Timp (esantioane)')
axs[0].set_ylabel('Amplitudine')

axs[1].scatter(y.real, y.imag, c=dist_y, cmap='viridis', s=3)
axs[1].plot([0, y[k].real], [0, y[k].imag], 'r', linewidth=2)
axs[1].plot(y[k].real, y[k].imag, 'ro')
axs[1].axhline(0, color='k', linewidth=0.5)
axs[1].axvline(0, color='k', linewidth=0.5)
axs[1].set_aspect('equal')
axs[1].set_xlim(-1.1, 1.1)
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_xlabel('Real')
axs[1].set_ylabel('Imaginar')
fig.tight_layout()
plt.savefig('./ex2_fig1.pdf')
plt.show()

# Figura 2
omega = [f/2, f, 2*f, 4*f + 3]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
for i, o in enumerate(omega):
    z = x * np.exp(-2j * np.pi * o * t)
    axs[i].scatter(z.real, z.imag, c=np.abs(z), cmap='viridis', s=3)
    axs[i].axhline(0, color='k', linewidth=0.5)
    axs[i].axvline(0, color='k', linewidth=0.5)
    axs[i].set_aspect('equal')
    axs[i].set_xlim(-1.1, 1.1)
    axs[i].set_ylim(-1.1, 1.1)
    axs[i].set_xlabel('Real')
    axs[i].set_ylabel('Imaginar')
    axs[i].set_title(f'omega = {o} Hz')
plt.savefig('./ex2_fig2.pdf')
plt.show()

#%%
#ex2 - animat
backend = plt.get_backend()
plt.switch_backend('TkAgg')
step = 10

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].set_xlim(0, fs)
axs[0].set_ylim(-1.1, 1.1)
axs[1].set_xlim(-1.1, 1.1)
axs[1].set_ylim(-1.1, 1.1)
axs[1].set_aspect('equal')
pt_x, = axs[0].plot([], [], 'ro')
pt_y, = axs[1].plot([], [], 'ro')
for m in range(0, fs, step):
    s = slice(m, m + step)
    axs[0].scatter(np.arange(m, m + step), x[s], c=dist_y[s], cmap='viridis', vmin=0, vmax=1, s=3)
    axs[1].scatter(y[s].real, y[s].imag, c=dist_y[s], cmap='viridis', vmin=0, vmax=1, s=3)
    pt_x.set_data([m], [x[m]])
    pt_y.set_data([y[m].real], [y[m].imag])
    plt.pause(0.01)
plt.show(block=True)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
Z = [x * np.exp(-2j * np.pi * o * t) for o in omega]
pts = []
for i, o in enumerate(omega):
    axs[i].set_xlim(-1.1, 1.1)
    axs[i].set_ylim(-1.1, 1.1)
    axs[i].set_aspect('equal')
    axs[i].set_title(f'omega = {o} Hz')
    pts.append(axs[i].plot([], [], 'ro')[0])
for m in range(0, fs, step):
    s = slice(m, m + step)
    for i, z in enumerate(Z):
        axs[i].scatter(z[s].real, z[s].imag, c=np.abs(z[s]), cmap='viridis', vmin=0, vmax=1, s=3)
        pts[i].set_data([z[m].real], [z[m].imag])
    plt.pause(0.01)
plt.show(block=True)

plt.switch_backend(backend)

#%%
#ex3
fs = 1000
N = fs
t = np.arange(N) / fs
f1, f2, f3 = 8, 30, 65
x = np.cos(2 * np.pi * f1 * t) + 3 * np.cos(2 * np.pi * f2 * t) + 0.7 * np.cos(2 * np.pi * f3 * t)

omega = np.arange(1, 101)

F = np.exp(-2j * np.pi * np.outer(omega, np.arange(N)) / N)
X = F @ x

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(t, x)
axs[0].set_xlabel('Timp (s)')
axs[0].set_ylabel('x(t)')
axs[1].stem(omega, np.abs(X))
axs[1].set_xlabel('Frecventa (Hz)')
axs[1].set_ylabel('|X(omega)|')
fig.tight_layout()
plt.savefig('./ex3.pdf')
plt.show()
