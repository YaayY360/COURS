import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift, fftfreq
from tfd import tfd,tfd_noshift

#p1
fe=4000
fa=1053
Dobs=0.5
t=np.arange(0, Dobs, 1/fe)

#p2 creation du signal et calcul de la tfd
sig=np.sin(2 * np.pi * fa * t)
Sig,nu_norm=tfd(sig)
freqs=nu_norm *fe

#p3 trace des 3 graphiques:
fig1, axs1=plt.subplots(3, 1, figsize=(10, 10))
fig1.suptitle('Analyse du signal source et du signal filtré (Spectre centré)', fontsize=14)

axs1[0].plot(t[:50], sig[:50], label='sig (entree)', marker='.')
axs1[0].set_title('representation temporelle (partielle)')
axs1[0].set_xlabel('temps (s)')
axs1[0].set_ylabel('amplitude')

axs1[1].plot(freqs, np.abs(Sig), label='|Sig|')
axs1[1].set_title('module de la TFD (echelle lineaire)')
axs1[1].set_xlabel('frequence(Hz)')
axs1[1].set_ylabel('module')

axs1[2].plot(freqs, 20 * np.log10(np.abs(Sig) + 1e-10), label='|Sig| dB') #la valeur 1e-10 est la pour ne pas avoir logarithme de 0
axs1[2].set_title('module de la TFD (echelle dB)')
axs1[2].set_xlabel('frequence (Hz)')
axs1[2].set_ylabel('module (dB)')

# p4

b=[1, -2, 1]
a=[1]
sig2=signal.lfilter(b, a, sig)
Sig2, _ =tfd(sig2) #on ignore le 2ème argument car l'axe des frequences est le meme

axs1[0].plot(t[:50], sig2[:50], label='sig2 (sortie)', marker='x', linestyle='--')
axs1[1].plot(freqs, np.abs(Sig2), label='|Sig2|', linestyle='--')
axs1[2].plot(freqs, 20 * np.log10(np.abs(Sig2) + 1e-10), label='|Sig2| dB', linestyle='--')

for ax in axs1:
    ax.legend()
    ax.grid(True)
    # ax.set_xlim([-fe/2, fe/2])

plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig1.png')

# p5

fig2, axs2=plt.subplots(3, 1, figsize=(10, 10))
fig2.suptitle('analyse du filtre H(z)', fontsize=14)

h = np.array([1, -2, 1])
n_h = np.arange(len(h))
axs2[0].stem(n_h, h, basefmt=" ")
axs2[0].set_title('reponse impulsionnelle h[n]')
axs2[0].set_xlabel('echantillons n')

freqs_H, H_freq=signal.freqz(b, a, whole=True, fs=fe)
freqs_H_centree=freqs_H-fe/2
H_freq_centree=np.fft.fftshift(H_freq)

axs2[1].plot(freqs_H_centree, np.abs(H_freq_centree), color='green')
axs2[1].set_title('reponse en frequence : Module |H|')
axs2[1].set_xlabel('frequence(Hz)')
axs2[1].set_ylabel('gain')

axs2[2].plot(freqs_H_centree, np.angle(H_freq_centree), color='purple')
axs2[2].set_title('reponse en frequence : Phase')
axs2[2].set_xlabel('frequence (Hz)')
axs2[2].set_ylabel('phase (radians)')

for ax in axs2:
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig2.png')







#III.2.2
from scipy.io import wavfile
import sounddevice as sd

fe_audio, audio_brut = wavfile.read('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Sources/Sources TP Sig3, 4, 5-20260415/jazz.WAV')

#verif des dim et passage en Mono si nécessaire
print(f"Dimensions initiales de l'audio : {audio_brut.shape}")
if len(audio_brut.shape) > 1:
    audio=audio_brut[:, 0] #on ne garde que la première colonne (voie de gauche)
    print("Fichier stéréo détecté : conversion en mono.")
else:
    audio=audio_brut

print(f"nouvelles dims : {audio.shape}")
print(f"frequence d'echantillonnage fe = {fe_audio} Hz")

# Ecoute du signal

#sd.play(audio, fe_audio)
#sd.wait() #attendre la fin de la lecture pour continuer le script


Audio_TFD, nu_norm_audio=tfd(audio)
freqs_audio=nu_norm_audio*fe_audio

plt.figure(figsize=(10, 4))
plt.plot(freqs_audio, np.abs(Audio_TFD))
plt.title('spectre de la sequence audio (module tfd)')
plt.xlabel('frequence (Hz)')
plt.ylabel('amplitude')
plt.grid(True)
plt.xlim([-fe_audio/2, fe_audio/2])
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig3_spectre_audio.png')


indice_depart=2*fe_audio
nombre_echantillons=100

Nobs=np.arange(indice_depart, indice_depart +nombre_echantillons)

audio_zoom=audio[Nobs]
temps_zoom=Nobs/fe_audio
#
plt.figure(figsize=(10, 4))
plt.stem(temps_zoom, audio_zoom, basefmt=" ")
plt.title('Sequence audio echantillonnee (zoom en temps)')
plt.xlabel('temps (s)')
plt.ylabel('amplitude numerisee')
plt.grid(True)
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig4_zoom_temporel_stem.png')


#III.2.3

fe = fe_audio      #freq d'origine
fe2 = 2 * fe       #fréq double
#  L'ecoute a fe
#sd.play(audio, fe)
#sd.wait()
#  L'ecoute a fe2
#sd.play(audio, fe2)
#sd.wait()
N = len(audio)
#creation des vrais axes temporelsqui dépendent de la fréquence de lecture
temps_son1 = np.arange(N) / fe
temps_son2 = np.arange(N) / fe2  #signal2 qui finira deux fois plus tot
#on reutilise le spectre centre et les freqs normalisees (nu_norm_audio) calcules au III.2.2
#mais on adapte l'axe des freq physiques (en Hz)
freqs_son1 = nu_norm_audio * fe
freqs_son2 = nu_norm_audio * fe2 # Le spectre s'étale deux fois plus loin !
# traces sur une figure coupee en2
fig3, axs3 = plt.subplots(2, 2, figsize=(14, 8))
fig3.suptitle('influence de la freq de lecture sur la meme sequence audio', fontsize=14)
# temporel
axs3[0, 0].plot(temps_son1, audio, color='blue')
axs3[0, 0].set_title(f'signal temporel (lu a fe = {fe} Hz)')
axs3[0, 0].set_xlabel('temps (s)')
axs3[0, 0].set_ylabel('amplitude')
#frequentiel
axs3[1, 0].plot(freqs_son1, np.abs(Audio_TFD), color='blue')
axs3[1, 0].set_title('pectre (lu a fe)')
axs3[1, 0].set_xlabel('frequence (Hz)')
axs3[1, 0].set_ylabel('module')
axs3[1, 0].set_xlim([-fe/2, fe/2])
#temporel
axs3[0, 1].plot(temps_son2, audio, color='red')
axs3[0, 1].set_title(f'signal temporel (lu a fe2 = {fe2} Hz)')
axs3[0, 1].set_xlabel('temps (s)')
axs3[0, 1].set_ylabel('amplitude')
#frequentiel
axs3[1, 1].plot(freqs_son2, np.abs(Audio_TFD), color='red')
axs3[1, 1].set_title('spectre (lu a fe2)')
axs3[1, 1].set_xlabel('frequence (Hz)')
axs3[1, 1].set_ylabel('module')
axs3[1, 1].set_xlim([-fe2/2, fe2/2])

for ax in axs3.flat:
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig5_comparaison_fe.png')

#III.2.4
n0 = 5 #retard
x = np.array([1, 3, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0]) #sequence d'entree
n_x = np.arange(len(x))
#reponse impulsionnelle hr c'est un vecteur de zéros de taille n0+1, avec un '1' à la fin
hr = np.zeros(n0 + 1)
hr[n0] = 1
n_hr = np.arange(len(hr))

#sortie yr par filtrage
#b=hr (les coefficients du numerateur sont exactement la reponse impulsionnelle pour un filtre RIF)
#a=[1] (dénominateur a 1)
yr = signal.lfilter(hr, [1], x)
n_yr = np.arange(len(yr))
#trace des 3 chronogrammes alignes verticalement
fig4, axs4 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
fig4.suptitle(f'filtre retard pur (retard n0 = {n0} echantillons)', fontsize=14)
axs4[0].stem(n_x, x, basefmt=" ", linefmt="C0-", markerfmt="C0o")
axs4[0].set_title('sequence d\'entrée x[n]')
axs4[0].set_ylabel('amplitude')
#reponse impulsionnelle hr
axs4[1].stem(n_hr, hr, basefmt=" ", linefmt="C1-", markerfmt="C1o")
axs4[1].set_title('reponse impulsionnelle hr[n]')
axs4[1].set_ylabel('amplitude')
#yr
axs4[2].stem(n_yr, yr, basefmt=" ", linefmt="C2-", markerfmt="C2o")
axs4[2].set_title('sequence de sortie yr[n] (retardee)')
axs4[2].set_xlabel('echantillons n')
axs4[2].set_ylabel('amplitude')

for ax in axs4:
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig6.png')





#III.2.5

#test sur le petit signal x
n0=5
x=np.array([1, 3, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0])

#creation de he1 (1 à l'indice 0, 0.5 à l'indice n0)
he1 = np.zeros(n0 + 1)
he1[0] = 1
he1[n0] = 0.5

#filtrage pour obtenir ye1
ye1 = signal.lfilter(he1, [1], x)

#trace des 3 chronogrammes pour verifier he1
fig5, axs5 = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
fig5.suptitle("Principe de l'écho sur un signal test", fontsize=14)
axs5[0].stem(x, basefmt=" ", linefmt="C0-", markerfmt="C0o")
axs5[0].set_title("Entrée x[n]")
axs5[1].stem(he1, basefmt=" ", linefmt="C1-", markerfmt="C1o")
axs5[1].set_title("Filtre écho he1[n]")
axs5[2].stem(ye1, basefmt=" ", linefmt="C2-", markerfmt="C2o")
axs5[2].set_title("Sortie ye1[n] (On voit bien la répétition atténuée !)")
plt.tight_layout()

#L'echo sur la vraie musique (he2)
fe = 44100
retard_sec = 1.0
N_retard = int(retard_sec * fe) # Nombre d'échantillons pour 1 seconde

#creation du grand filtre he2
he2 = np.zeros(N_retard + 1)
he2[0] = 1       # Le son direct (pas pour la voie filtrée seule, mais utile si mixage)
he2[N_retard] = 0.5 # L'écho


#on va generer uniquement la voie "retardee" pour la stéréo parce que ca prends du temps.
filtre_echo_pur = np.zeros(N_retard + 1)
filtre_echo_pur[N_retard] = 0.5
#on suppose 'audio' defini dans les parties precedentes
audio_echo = signal.lfilter(filtre_echo_pur, [1], audio)

#creation de la piste Stereo hybride
#colonne 1 : audio d'origine (Gauche), colonne 2 : Audio avec Echo (Droite)
audioe = np.column_stack((audio, audio_echo))

#sauvegarde pour ecoute
wavfile.write("/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/musique_avec_echo_stereo.wav", fe, audioe.astype(np.int16))


#on extrait une infime fraction
zoom_indices = np.arange(40000, 40200) 
audio_extrait = audio[zoom_indices]
t_extrait = zoom_indices / fe

fig6, axs6 = plt.subplots(2, 1, figsize=(10, 8))
fig6.suptitle("tromper l'oeil : numerique vs analogique", fontsize=14)

#vrai signal numérique (Stem)
axs6[0].stem(t_extrait, audio_extrait, basefmt=" ")
axs6[0].set_title("realite : sequence numerique echantillonnee (stem)")
axs6[0].set_ylabel("amplitude")

#illusion analogique (Plot)
axs6[1].plot(t_extrait, audio_extrait, color='red')
axs6[1].set_title("L'illusion : representation de la source 'continue' (plot)")
axs6[1].set_xlabel("temps (s)")
axs6[1].set_ylabel("amplitude")

for ax in axs6:
    ax.grid(True)

plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/fig7.png')



