#%%
import numpy as np
import matplotlib.pyplot as plt
import sounddevice
import scipy.io.wavfile, scipy.signal

#%%
#ex1
A = 2
f = 20
phi = np.pi / 3
t = np.linspace(0, 0.5, 1000)

sig_sin = A * np.sin(2 * np.pi * f * t + phi)
sig_cos = A * np.cos(2 * np.pi * f * t + phi - np.pi/2)

fig, axs = plt.subplots(2)
axs[0].plot(t, sig_sin)
axs[0].set_title('Sinus')
axs[1].plot(t, sig_cos)
axs[1].set_title('Cosinus')
fig.tight_layout()
plt.savefig('./ex1.pdf')
plt.show()

#%%
#ex2
A = 1
f = 20
phis = [np.pi / 3, np.pi, 2 * np.pi / 3, 6 * np.pi / 13]
t = np.linspace(0, 0.1, 1000)

plt.figure()

for phi in phis:
    sig = A * np.sin(2 * np.pi * f * t + phi)
    plt.plot(t, sig, label=f'faza = {phi:.2f}')

plt.legend()
plt.savefig('./ex2_4faze.pdf')
plt.show()
#%%
#ex2
t = np.linspace(0, 0.5, 1000)
A = 1
f = 20
phi = 2 * np.pi / 3
SNR = [0.1, 1, 10, 100]

x = A * np.sin(2 * np.pi * f * t + phi)
z = np.random.normal(0, 1, len(x))

norm_x = np.linalg.norm(x)
norm_z = np.linalg.norm(z)

fig, axs = plt.subplots(4)

for i, snr in enumerate(SNR):
    gamma = norm_x / (np.sqrt(snr) * norm_z)
    sig = x + gamma * z

    axs[i].plot(t, sig)
    axs[i].set_title(f'SNR = {snr}')

fig.tight_layout()
plt.savefig('./ex2_noise.pdf')
plt.show()

#%%
#ex3
fs = 48000

# semnalele din lab1 regenerate la fs ca sa se auda corect
# ex 2a sinus 400 Hz
t_a = np.linspace(0, 2, 2 * fs)
s_a = np.sin(2 * np.pi * 400 * t_a)

# ex 2b sinus 800 Hz 3 secunde
t_b = np.linspace(0, 3, 3 * fs)
s_b = np.sin(2 * np.pi * 800 * t_b)

# ex 2c sawtooth 240 Hz
t_c = np.linspace(0, 2, 2 * fs)
s_c = 2 * np.mod(t_c * 240, 1) - 1

# ex 2d square 300 Hz
t_d = np.linspace(0, 2, 2 * fs)
s_d = np.sign(np.sin(2 * np.pi * 300 * t_d))

sounddevice.default.device = 5
for s in [s_a, s_b, s_c, s_d]:
    sounddevice.play(s, fs)
    sounddevice.wait()

    
scipy.io.wavfile.write('./sig_b.wav', fs, s_b)
rate, sig = scipy.io.wavfile.read('./sig_b.wav')
sounddevice.play(sig, fs)
sounddevice.wait()

#%%
#ex4

fig, axs = plt.subplots(3)
t = np.linspace(0, 0.01, 1000)
sig_sin = 3 * np.sin(2 * np.pi * 440 * t)
sig_saw = 3 * np.mod(t * 440, 1) - 1

axs[0].plot(t, sig_sin)
axs[0].set_title('Sinus 440 Hz')
axs[1].plot(t, sig_saw)
axs[1].set_title('Sawtooth 440 Hz')
axs[2].plot(t, sig_sin + sig_saw)
axs[2].set_title('Suma semnale')

fig.tight_layout()
plt.savefig('./ex4.pdf')
plt.show()

#%% 
#ex5
fs = 48000
t = np.linspace(0, 3, 3 * fs)
s1 = np.sin(2 * np.pi * 400 * t)
s2 = np.sin(2 * np.pi * 1200 * t)
s = np.concatenate([s1, s2])

# se aud doua tonuri unul dupa altul, al doilea e mai inalt pentru ca are frecventa mai mare

sounddevice.play(s, fs)
sounddevice.wait()
#%%
#ex6
fs = 100
f = [fs/2, fs/4, 0]
n = np.arange(0, 0.1, 1 / fs)

fig, axs = plt.subplots(3)

for i, freq in enumerate(f):
    sig = np.sin(2 * np.pi * freq * n)
    axs[i].stem(n, sig)
    axs[i].set_title(f'Frecventa = {freq} Hz')

fig.tight_layout()
plt.savefig('./ex6.pdf')
plt.show()

# a) teoretic sin(pi * n) = 0 pentru orice n, esantionam exact in trecerile prin zero
#    in grafic apar totusi valori de ordinul 1e-15 (erori de rotunjire ale lui pi), iar axa y se scaleaza dupa ele
#    de aceea arata ca un semnal aleator, desi practic e nul
# b) avem 4 esantioane pe perioada: 0, 1, 0, -1 si se repeta
# c) sin(0) = 0, semnal constant nul

#%%
#ex7
fs = 1000
t = np.arange(0, 0.1, 1 / fs)
s = np.sin(2 * np.pi * 30 * t)

s_dec = s[::4]
t_dec = t[::4]

s_dec2 = s[1::4]
t_dec2 = t[1::4]

fig, axs = plt.subplots(3)
axs[0].plot(t, s, '.-')
axs[0].set_title('Original 1000 Hz')
axs[1].plot(t_dec, s_dec, '.-')
axs[1].set_title('Decimat 250 Hz')
axs[2].plot(t_dec2, s_dec2, '.-')
axs[2].set_title('Decimat 250 Hz de la al doilea element')

fig.tight_layout()
plt.savefig('./ex7.pdf')
plt.show()

# a) semnalul decimat are de 4 ori mai putine esantioane, dar frecventa ramane 30 Hz
#    pentru ca 30 Hz e mult sub 250 / 2
# b) obtinem aceeasi sinusoida, doar ca esantioanele sunt luate decalat cu 1 ms

#%%
#ex8
x = np.linspace(-np.pi/2, np.pi/2, 1000)
sin_real = np.sin(x)
sin_pade = (x - 7*x**3/60)/(1+x**2/20)

err_taylor = np.abs(sin_real - x)
err_pade = np.abs(sin_real - sin_pade)

fig, axs = plt.subplots(3)
axs[0].plot(x, x, color='red', label='Taylor')
axs[0].plot(x, sin_real, color='blue', label='sin')
axs[0].plot(x, sin_pade, color='green', label='Pade')
axs[0].legend()
axs[0].set_title('Aproximari')

axs[1].plot(x, err_taylor, color='red', label='Taylor')
axs[1].plot(x, err_pade, color='green', label='Pade')
axs[1].legend()
axs[1].set_title('Eroare')

axs[2].semilogy(x, err_taylor, color='red', label='Taylor')
axs[2].semilogy(x, err_pade, color='green', label='Pade')
axs[2].legend()
axs[2].set_title('Eroare (log)')

fig.tight_layout()
plt.savefig('./ex8.pdf')
plt.show()

# sin(x) ~ x e buna doar pentru valori mici, eroarea creste spre capete
# Pade e mult mai precisa pe tot intervalul
