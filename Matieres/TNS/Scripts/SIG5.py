from numpy import (abs, arange, array, cos,sin, pi, log10, zeros, ones,
                   real, angle, concatenate, argmax)
from scipy.signal import freqz, zpk2tf, lfilter, spectrogram as spgram
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from dataclasses import dataclass

plt.rcParams["text.usetex"] = False
plt.close('all')

SAVE = '/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/ResTP5/'


@dataclass
class Filtre:
    ordre: int
    num:   object
    den:   object
    zpk:   tuple

def tfd(seq, fe):
    N=len(seq)
    Seq=fft(seq)
    nu=fftfreq(N)
    f=nu*fe
    return Seq, nu, f



# 1– PREPARATION
k=1.0
zero=0.0+0j
pa=0.0+0.4j    # pôle a :  j0.4
pb=-0.6+0.0j    # pôle b : -0.6
pc=0.0-0.8j    # pôle c : -j0.8

def make_filtre_1er_ordre(pole, z=0.0+0j, gain=1.0):
    zs=array([z])
    ps=array([pole])
    num,den=zpk2tf(zs, ps, gain)
    return Filtre(ordre=1, num=num, den=den, zpk=(zs, ps, gain))

Hza=make_filtre_1er_ordre(pa)
Hzb=make_filtre_1er_ordre(pb)
Hzc=make_filtre_1er_ordre(pc)

Nu_prep=arange(0, 1, 1/2048)
_, Ha=freqz(Hza.num,Hza.den,worN=Nu_prep*2*pi)
_, Hb=freqz(Hzb.num,Hzb.den,worN=Nu_prep*2*pi)
_, Hc=freqz(Hzc.num,Hzc.den,worN=Nu_prep*2*pi)

Ha_n=abs(Ha)/abs(Ha).max()
Hb_n=abs(Hb)/abs(Hb).max()
Hc_n=abs(Hc)/abs(Hc).max()

theta=arange(0, 2*pi, 0.01)

# Figure 1 : DPZ + reponse en frequence ─────────────────────────────────
fig1,(ax1, ax2)=plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle(" DPZ et reponse en frequence des 3 filtres du 1er ordre")

ax1.plot(cos(theta), sin(theta), 'k--', lw=0.8, label='cercle unite')
for pole, lbl, c in [(pa, 'pole a : j0.4',  'tab:blue'),
                      (pb, 'pole b : -0.6',   'tab:orange'),
                      (pc, 'pole c : -j0.8',  'tab:green')]:
    ax1.plot(real(pole), pole.imag,'x',ms=10,mew=2.5,color=c,label=lbl)
ax1.plot(real(zero), zero.imag,'ko',ms=8,mfc='none',label='zéro en 0')
ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal'); ax1.grid(True)
ax1.set_xlabel('Re'); ax1.set_ylabel('Im')
ax1.set_title('DPZ (x = pole, o = zero)')
ax1.legend(fontsize=8)

ax2.plot(Nu_prep, Ha_n,label='|Hza|– pôle j0.4')
ax2.plot(Nu_prep, Hb_n,label='|Hzb|– pôle -0.6')
ax2.plot(Nu_prep, Hc_n,label='|Hzc|– pôle -j0.8')
ax2.set_xlabel('frequence normalisee nu')
ax2.set_ylabel('module normalise')
ax2.set_title('reponse en frequence (normalisee par le gain max)')
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig(SAVE + 'fig1.png', dpi=150)



# 1– LE CHIRP LINEAIRE D'AMPLITUDE CONSTANTE


Tobs = 5.0
Dobs = Tobs / 3
fmin = 400.0         # Hz
fmax = 2000.0        # Hz
beta = (fmax - fmin) / Tobs   # [Hz/s]

def genere_chirp(fe_g, fmin_g, fmax_g, Tobs_g, Dobs_g):
    """
    Génère le chirp linéaire échantillonné à fe_g.
    Retourne : chirp_amp, temps, Na, Nb, Nc
    """
    beta_g    = (fmax_g - fmin_g) / Tobs_g
    temps     = arange(0, Tobs_g, 1.0 / fe_g)
    phi_i     = 2 * pi * (fmin_g * temps + 0.5 * beta_g * temps**2)
    chirp_amp = cos(phi_i)

    N     = len(temps)
    nDobs = int(Dobs_g * fe_g)

    Na = arange(0, nDobs)
    Nb = arange(N // 2 - nDobs // 2, N // 2 + nDobs // 2)
    Nc = arange(N - nDobs, N)

    return chirp_amp, temps, Na, Nb, Nc


# V.1.2 – Représentation TC (fe_a très grande pour simuler le continu)
fe_a = 40000.0
chirpa, tempsa, Naa, Nab, Nac = genere_chirp(fe_a, fmin, fmax, Tobs, Dobs)

# V.1.3 – Echantillonnage à fe1 = 4000 Hz
# Shannon : fe > 2 * fmax = 4000 Hz  → fe1 = 4000 Hz est à la limite
fe1 = 4000.0
chirpe1, tempe1, Ne1a, Ne1b, Ne1c = genere_chirp(fe1, fmin, fmax, Tobs, Dobs)

# TFD sur chacun des 3 intervalles
Chirpe1a, Nu1a, f1a = tfd(chirpe1[Ne1a], fe1)
Chirpe1b, Nu1b, f1b = tfd(chirpe1[Ne1b], fe1)
Chirpe1c, Nu1c, f1c = tfd(chirpe1[Ne1c], fe1)

# ── Figure 2 : représentations temporelles TC vs TD sur les 3 intervalles ──
fig2, axes2 = plt.subplots(3, 1, figsize=(13, 10))
fig2.suptitle("V.1.3 – Chirp : comparaison TC / TD  (fe1 = 4000 Hz)")

labels_int = ['Intervalle a  [0 ; Dobs[',
              'Intervalle b  [Tobs/2-Dobs/2 ; Tobs/2+Dobs/2[',
              'Intervalle c  [Tobs-Dobs ; Tobs[']

for idx, (Na_tc, Ne_td, lbl) in enumerate(
        zip([Naa, Nab, Nac], [Ne1a, Ne1b, Ne1c], labels_int)):
    ax = axes2[idx]
    pas_aff = max(1, len(Na_tc) // 4000)
    ax.plot(tempsa[Na_tc[::pas_aff]], chirpa[Na_tc[::pas_aff]],
            'b-', lw=0.7, alpha=0.6, label='TC chirpa')
    ax.stem(tempe1[Ne_td], chirpe1[Ne_td],
            linefmt='r-', markerfmt='ro', basefmt=' ',
            label='TD chirpe1')
    ax.set_title(lbl, fontsize=9)
    ax.set_xlabel('Temps [s]'); ax.set_ylabel('Amplitude')
    ax.legend(fontsize=8); ax.grid(True)

plt.tight_layout()
plt.savefig(SAVE + 'fig2.png', dpi=150)

# ── Figure 3 : modules des TFD superposés ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(11, 4))
ax3.plot(Nu1a, abs(Chirpe1a), label='|Chirpe1a|  (intervalle a)')
ax3.plot(Nu1b, abs(Chirpe1b), label='|Chirpe1b|  (intervalle b)')
ax3.plot(Nu1c, abs(Chirpe1c), label='|Chirpe1c|  (intervalle c)')
ax3.set_title("V.1.3 – Modules des TFD des 3 intervalles (fe1 = 4000 Hz)")
ax3.set_xlabel('Frequence normalisee nu')
ax3.set_ylabel('Module TFD')
ax3.legend(); ax3.grid(True)
plt.tight_layout()
plt.savefig(SAVE + 'fig3.png', dpi=150)

# ── Figure 4 : spectrogramme ───────────────────────────────────────────────
nperseg = 256
f_sg, t_sg, Sxx = spgram(chirpe1, fs=fe1, nperseg=nperseg,
                           noverlap=nperseg // 2, nfft=nperseg, scaling='spectrum')

fig4, ax4 = plt.subplots(figsize=(11, 4))
pcm = ax4.pcolormesh(t_sg, f_sg, 10 * log10(Sxx + 1e-12),
                      shading='gouraud', cmap='inferno')
plt.colorbar(pcm, ax=ax4, label='PSD [dB]')
ax4.set_title(f"V.1.3 – Spectrogramme de chirpe1  (fe1 = {fe1:.0f} Hz)")
ax4.set_xlabel('Temps [s]'); ax4.set_ylabel('Frequence [Hz]')
plt.tight_layout()
plt.savefig(SAVE + 'fig4.png', dpi=150)

# ── Figure 5 : optionnel fe2 = 1600 Hz / fe3 = 2000 Hz ───────────────────
fig5, axes5 = plt.subplots(2, 3, figsize=(15, 8))
fig5.suptitle("V.1.4 – Optionnel : fe2 = 1600 Hz  /  fe3 = 2000 Hz")

for row, (fe_opt, lbl_fe) in enumerate([(1600.0, 'fe2 = 1600 Hz'),
                                         (2000.0, 'fe3 = 2000 Hz')]):
    c_opt, t_opt, Na_o, Nb_o, Nc_o = genere_chirp(fe_opt, fmin, fmax, Tobs, Dobs)
    for col, (Ne_i, lbl_i) in enumerate(zip([Na_o, Nb_o, Nc_o],
                                             ['Intervalle a', 'b', 'c'])):
        ax = axes5[row, col]
        ax.plot(t_opt[Ne_i], c_opt[Ne_i])
        ax.set_title(f"{lbl_fe} – {lbl_i}", fontsize=9)
        ax.set_xlabel('t [s]'); ax.set_ylabel('Amplitude')
        ax.grid(True)

plt.tight_layout()
plt.savefig(SAVE + 'fig5.png', dpi=150)


# =============================================================================
# V.2 – SIGNAL AUDIO COMPOSITE
# =============================================================================

audioSource_raw, audioSource_fe = sf.read(
    '/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Sources/Sources TP Sig3, 4, 5-20260415/rock1.wav',
    always_2d=True)
audioSource = audioSource_raw.T    # (voies x echantillons)
fe          = float(audioSource_fe)
print(f"[V.2] music.wav  –  fe = {fe} Hz  –  shape = {audioSource.shape}")

# Ecoute (decommenter)
# sd.play(audioSource.T, int(fe)); sd.wait()

nvoies, nech = audioSource.shape

# Fenetre Nobs : ~100 ms autour du max d'amplitude
Nobs_dur = int(0.1 * fe)
idx_max  = int(argmax(abs(audioSource[0])))
i_start  = max(0, idx_max - Nobs_dur // 2)
i_stop   = min(nech, i_start + Nobs_dur)
Nobs     = arange(i_start, i_stop)

t_all  = arange(nech) / fe
t_nobs = Nobs / fe

Audio_full, Nu_full, f_full = tfd(audioSource[0], fe)
Audio_nobs, Nu_nobs, f_nobs = tfd(audioSource[0, Nobs], fe)

# ── Figure 6 : sequence complete ──────────────────────────────────────────
fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(13, 7))
fig6.suptitle("V.2 – audioSource : representations temporelle et frequentielle (integralite)")

ax6a.plot(t_all, audioSource[0], lw=0.4)
ax6a.set_xlabel('Temps [s]'); ax6a.set_ylabel('Amplitude')
ax6a.set_title(f"Temporel – {nech} echantillons, fe = {fe:.0f} Hz")
ax6a.grid(True)

N2 = nech // 2
ax6b.plot(f_full[:N2], abs(Audio_full[:N2]))
ax6b.set_xlabel('Frequence [Hz]'); ax6b.set_ylabel('Module TFD')
ax6b.set_title("Spectre (module) – sequence complete")
ax6b.grid(True)

plt.tight_layout()
plt.savefig(SAVE + 'fig6.png', dpi=150)

# ── Figure 7 : fenetre Nobs ────────────────────────────────────────────────
N2_nobs = len(Nobs) // 2
db_nobs = 20 * log10(abs(Audio_nobs[:N2_nobs]) + 1e-10)

fig7, axes7 = plt.subplots(3, 1, figsize=(13, 10))
fig7.suptitle(f"V.2 – audioSource sur fenetre Nobs ~100 ms (ech. {i_start} a {i_stop})")

axes7[0].plot(t_nobs, audioSource[0, Nobs])
axes7[0].set_xlabel('Temps [s]'); axes7[0].set_ylabel('Amplitude')
axes7[0].set_title("Representation temporelle sur Nobs")
axes7[0].grid(True)

axes7[1].plot(f_nobs[:N2_nobs], abs(Audio_nobs[:N2_nobs]))
axes7[1].set_xlabel('Frequence [Hz]'); axes7[1].set_ylabel('Module TFD')
axes7[1].set_title("Spectre lineaire sur Nobs")
axes7[1].grid(True)

axes7[2].plot(f_nobs[:N2_nobs], db_nobs)
axes7[2].set_xlabel('Frequence [Hz]'); axes7[2].set_ylabel('Module [dB]')
axes7[2].set_title("Spectre en dB sur Nobs")
axes7[2].grid(True)

plt.tight_layout()
plt.savefig(SAVE + 'fig7.png', dpi=150)

print("=== SIG5 termine – fig1 a fig7 sauvegardes dans ResTP5/ ===")