#%%
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io.wavfile

#%%
#ex1
def DFT(x):
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
    for m in range(N):
        X[m] = np.sum(x * np.exp(-2j * np.pi * m * n / N))
    return X

def FFT(A):
    n = len(A)

    if n == 1:
        return np.array([A[0]], dtype=complex)

    A_par = A[0::2]
    A_imp = A[1::2]

    Y_par = FFT(A_par)
    Y_imp = FFT(A_imp)

    omega_n = np.exp(-2j * np.pi / n)
    omega = 1

    Y = np.zeros(n, dtype=complex)

    for k in range(n // 2):
        Y[k] = Y_par[k] + omega * Y_imp[k]
        Y[k + n // 2] = Y_par[k] - omega * Y_imp[k]
        omega = omega * omega_n

    return Y

x = np.random.rand(64)
print(f'DFT == np.fft: {np.allclose(DFT(x), np.fft.fft(x))}')
print(f'FFT == np.fft: {np.allclose(FFT(x), np.fft.fft(x))}')

Ns = [128, 256, 512, 1024, 2048, 4096, 8192]
t_dft, t_fft, t_np = [], [], []

for N in Ns:
    x = np.random.rand(N)

    start = time.perf_counter()
    DFT(x)
    t_dft.append(time.perf_counter() - start)

    start = time.perf_counter()
    FFT(x)
    t_fft.append(time.perf_counter() - start)

    start = time.perf_counter()
    np.fft.fft(x)
    t_np.append(time.perf_counter() - start)

    print(f'N = {N}: DFT {t_dft[-1]:.4f}s, FFT {t_fft[-1]:.4f}s, numpy {t_np[-1]:.6f}s')

plt.figure()
plt.plot(Ns, t_dft, 'o-', label='DFT (implementare proprie)')
plt.plot(Ns, t_fft, 'o-', label='FFT (implementare proprie)')
plt.plot(Ns, t_np, 'o-', label='numpy.fft')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Timp (s)')
plt.title('Timp de executie')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('./ex1.pdf')
plt.show()

# DFT e O(N^2), FFT e O(N log N): la fiecare dublare a lui N timpul DFT creste ~de 4 ori, al FFT ~de 2 ori
# numpy.fft e tot O(N log N), dar implementat in C, deci cu cateva ordine de marime mai rapid

#%%
#ex2
f0 = 3                # frecventa semnalului original (Hz)
fs = 5                # frecventa de esantionare sub-Nyquist (fs < 2 * f0 = 6 Hz)
f1 = f0 + fs          # 8 Hz
f2 = f0 + 2 * fs      # 13 Hz
# sin(2 pi (f0 + k fs) n / fs) = sin(2 pi f0 n / fs + 2 pi k n) = sin(2 pi f0 n / fs) -> aceleasi esantioane

T = 1
t = np.linspace(0, T, 2000)
n = np.arange(0, T, 1 / fs)

x0 = np.sin(2 * np.pi * f0 * t)
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)
x_n = np.sin(2 * np.pi * f0 * n)

fig, axs = plt.subplots(4, figsize=(8, 8))
axs[0].plot(t, x0)
axs[0].set_title(f'Semnal original {f0} Hz')
axs[1].plot(t, x0)
axs[1].plot(n, x_n, 'o', color='gold')
axs[1].set_title(f'Esantionat cu fs = {fs} Hz')
axs[2].plot(t, x1, color='red')
axs[2].plot(n, x_n, 'o', color='gold')
axs[2].set_title(f'{f1} Hz, aceleasi esantioane')
axs[3].plot(t, x2, color='green')
axs[3].plot(n, x_n, 'o', color='gold')
axs[3].set_title(f'{f2} Hz, aceleasi esantioane')
for ax in axs:
    ax.set_xlim(0, T)
    ax.axhline(0, color='k', linewidth=0.5)
axs[3].set_xlabel('Timp (s)')
fig.tight_layout()
plt.savefig('./ex2.pdf')
plt.show()

# toate trei semnalele trec prin aceleasi puncte galbene, deci din esantioane nu putem sti care a fost semnalul initial

#%%
#ex3
fs = 30               # peste Nyquist pentru toate trei semnalele (fs > 2 * 13 Hz)
n = np.arange(0, T, 1 / fs)

fig, axs = plt.subplots(4, figsize=(8, 8))
axs[0].plot(t, x0)
axs[0].set_title(f'Semnal original {f0} Hz')
axs[1].plot(t, x0)
axs[1].plot(n, np.sin(2 * np.pi * f0 * n), 'o', color='gold')
axs[1].set_title(f'{f0} Hz esantionat cu fs = {fs} Hz')
axs[2].plot(t, x1, color='red')
axs[2].plot(n, np.sin(2 * np.pi * f1 * n), 'o', color='gold')
axs[2].set_title(f'{f1} Hz esantionat cu fs = {fs} Hz')
axs[3].plot(t, x2, color='green')
axs[3].plot(n, np.sin(2 * np.pi * f2 * n), 'o', color='gold')
axs[3].set_title(f'{f2} Hz esantionat cu fs = {fs} Hz')
for ax in axs:
    ax.set_xlim(0, T)
    ax.axhline(0, color='k', linewidth=0.5)
axs[3].set_xlabel('Timp (s)')
fig.tight_layout()
plt.savefig('./ex3.pdf')
plt.show()

# acum fiecare semnal are esantioane diferite, deci nu mai exista aliere
# pentru semnalul de 3 Hz ajunge orice fs > 6 Hz; am ales 30 Hz ca sa fie corect esantionate si 8 Hz si 13 Hz

#%%
#ex4
f_max = 200
print(f'fs minim pentru contrabas: fs > {2 * f_max} Hz')

# frecventa maxima din semnal e B = 200 Hz, deci din Nyquist fs > 2 * 200 = 400 Hz
# (si cu teorema de esantionare trece-banda, latimea benzii e 200 - 40 = 160 Hz,
#  floor(200 / 160) = 1, deci tot fs >= 2 * 200 / 1 = 400 Hz)

#%%
#ex5
# inregistrarea vocalelor "a, e, i, o, u" e in vocale.wav (spectrograma ei e desenata la ex6)
# da, vocalele se pot distinge: fiecare are alt set de benzi

#%%
#ex6
# a) citire semnal
fs, x = scipy.io.wavfile.read('./vocale.wav')
x = x[:, 0]                # fisierul e stereo, pastram doar un canal
N = len(x)

# b) grupuri de 1% din semnal, cu suprapunere de 50%
dim = N // 100
pas = dim // 2

# c) FFT pe fiecare grup
coloane = []
for start in range(0, N - dim + 1, pas):
    grup = x[start:start + dim]
    fft_grup = np.abs(np.fft.fft(grup))
    coloane.append(fft_grup[:dim // 2])    # a doua jumatate e oglindirea primei

# d) fiecare FFT devine o coloana a matricei
S = np.array(coloane).T

# e) afisare (in dB, altfel se vad doar cele mai puternice frecvente)
plt.figure(figsize=(10, 5))
plt.imshow(20 * np.log10(S + 1), origin='lower', aspect='auto', extent=[0, N / fs, 0, fs / 2])
plt.colorbar(label='dB')
plt.ylim(0, 5000)
plt.xlabel('Timp (s)')
plt.ylabel('Frecventa (Hz)')
plt.title('Spectrograma vocale')
plt.savefig('./ex6.pdf')
plt.show()

#%%
#ex7
P_semnal_db = 90
SNR_db = 80
P_zgomot_db = P_semnal_db - SNR_db
print(f'P_zgomot = {P_zgomot_db} dB')

# SNR_dB = 10 log10(P_semnal / P_zgomot) = 10 log10(P_semnal / P0) - 10 log10(P_zgomot / P0)
# deci P_zgomot_dB = P_semnal_dB - SNR_dB = 90 - 80 = 10 dB
# in valori absolute: P_zgomot = 10^(10/10) * P0 = 10 * P0 (iar P_semnal = 10^9 * P0, raport 10^8)
