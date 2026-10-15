import numpy as np
import matplotlib.pyplot as plt

# ex 1a
t = np.arange(0, 0.03, 0.0005)

# ex 1b

def x(t):
    return np.cos(520 * np.pi * t + np.pi / 3)

def y(t):
    return np.cos(280 * np.pi * t - np.pi / 3)

def z(t):
    return np.cos(120 * np.pi * t + np.pi / 3)

fig, axs = plt.subplots(3)
axs[0].plot(t, x(t))
axs[1].plot(t, y(t))
axs[2].plot(t, z(t))
plt.savefig('./lab1/ex1b.pdf')
plt.show()

# ex 1c esantionare la 200 Hz
fs = 200
n = np.arange(0, 0.03, 1/fs)

xn = x(n)
yn = y(n)
zn = z(n)

fig2, axs2 = plt.subplots(3)
axs2[0].stem(n, xn)
axs2[1].stem(n, yn)
axs2[2].stem(n, zn)
plt.savefig('./lab1/ex1c.pdf')
plt.show()

# prima figura
fig1, axs1 = plt.subplots(2)

# ex 2a sinus 400 Hz 1600 esantioane
t_a = np.linspace(0, 0.01, 1600)
s_a = np.sin(2 * np.pi * 400 * t_a)
axs1[0].plot(t_a[:100], s_a[:100])
axs1[0].set_title('Sinus 400 Hz')

# ex 2b sinus 800 Hz 3 secunde
t_b = np.linspace(0, 3, 30000)
s_b = np.sin(2 * np.pi * 800 * t_b)
axs1[1].plot(t_b[:100], s_b[:100])
axs1[1].set_title('Sinus 800 Hz')

fig1.tight_layout()
plt.savefig('./lab1/ex2_fig1.pdf')

# a doua figura
fig2, axs2 = plt.subplots(2)

# ex 2c sawtooth 240 Hz
t_c = np.linspace(0, 0.05, 5000)
s_c = 2 * np.mod(t_c * 240, 1) - 1
axs2[0].plot(t_c, s_c)
axs2[0].set_title('Sawtooth 240 Hz')

# ex 2d square 300 Hz
t_d = np.linspace(0, 0.05, 5000)
s_d = np.sign(np.sin(2 * np.pi * 300 * t_d))
axs2[1].plot(t_d, s_d)
axs2[1].set_title('Square 300 Hz')

fig2.tight_layout()
plt.savefig('./lab1/ex2_fig2.pdf')

# a treia figura
fig3, axs3 = plt.subplots(2)

# ex 2e semnal 2d aleator
s_e = np.random.rand(128, 128)
axs3[0].imshow(s_e)
axs3[0].set_title('Aleator 2D')

# ex 2f semnal 2d personalizat
s_f = np.zeros((128, 128))
s_f[13:37, 67:76] = 1
axs3[1].imshow(s_f)
axs3[1].set_title('Personalizat 2D')

fig3.tight_layout()
plt.savefig('./lab1/ex2_fig3.pdf')
plt.show()

# 3. a)
# fs = 2000 Hz => T = 1 / 2000 = 0.0005 secunde

# 3. b)
# 1h = 3600s
# nr esantioane = 3600 * 2000 = 7200000
# fiecare esantion are 4 biti deci in total avem 28800000 biti
# impartim la 8 pentru a afla numarul de bytes => 3600000 bytes = 3.6 MB