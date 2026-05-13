import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift, fftfreq

# 1- sous-echantillonnage

# p1
fe=1000
Dobs=0.5
t=np.arange(0, Dobs, 1/fe)
fa=10
s=np.sin(2 * np.pi * fa * t)
# p2
fe2=100
t2=np.arange(0, Dobs, 1/fe2)
s2=np.sin(2 * np.pi * fa * t2)
# p3
Nsous=np.arange(0, len(s), 10)
un=np.ones(len(Nsous))
# p4
s3=s[Nsous]
# p5.
plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(t,s,'b-',label='s (1kHz)')
plt.plot(t2,s2,'ro',label='s2 (100Hz)')
plt.title("signaux s et s2")
plt.legend()
plt.subplot(3, 1, 2)
plt.stem(t2, un, linefmt='k-', markerfmt='ko', basefmt='k-')
plt.title("impulsions d'echantillonnage (vecteur un)")
plt.subplot(3, 1, 3)
plt.stem(t2, s3, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title("sequence extraite s3")
plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig1.png')






# 2- chargement de la séquence et TFD de reference ---

# chargement du fichier .npz
path='/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Sources/sequence.npz'
data=np.load(path)
#print(data.files)
se=np.squeeze(data['se_amp'])
fe=float(data['se_fe'])
se_nobs=int(data['se_nobs'])

Dobs=se_nobs/fe
t=np.arange(se_nobs) / fe

SE=np.fft.fft(se)
f_axe=np.fft.fftfreq(se_nobs, d=1/fe)
module_SE=np.abs(SE)/se_nobs
f_axe_centre = np.fft.fftshift(f_axe)
module_SE_centre = np.fft.fftshift(module_SE)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, se, 'b')
plt.title('signal temporel se ')
plt.xlabel('temps (s)')
plt.ylabel('amplitude')
plt.xlim(0.15,0.25)
plt.subplot(2, 1, 2)
#plt.plot(f_axe[:se_nobs//2], module_SE[:se_nobs//2], 'r')
plt.plot(f_axe_centre, module_SE_centre, 'r')
plt.title('spectre (TFD) de la séquence se')
plt.xlabel('frequence (Hz)')
plt.ylabel('amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig2.png')



# 3- influence du nombre d'echantillons nobs


frac=4
sm=se[:se_nobs // frac]
nobs_m=len(sm)
Sm=np.fft.fft(sm)
f_axe_m= np.fft.fftshift(np.fft.fftfreq(nobs_m, d=1/fe))
module_Sm_centre = np.fft.fftshift(np.abs(Sm) / nobs_m)
plt.figure(figsize=(10, 5))
plt.plot(f_axe_centre, module_SE_centre, 'b', label='reference se (long)')
plt.plot(f_axe_m, module_Sm_centre, 'r--', label='sequence sm (court)')
plt.title('influence de nobs')
plt.legend()
plt.grid(True)
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig3.png')


# 4- Influence de l'ordre Nfft

Nfft = 8 * se_nobs
So = np.fft.fft(se, n=Nfft)

f_axe_o = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1/fe))
module_So_centre = np.fft.fftshift(np.abs(So) / se_nobs)
plt.figure(figsize=(10, 5))
plt.plot(f_axe_centre, module_SE_centre, 'bo', label='reference se')
plt.plot(f_axe_o, module_So_centre, 'r-', label='TFD so (ordre Nfft=8*nobs)')
plt.title(" influence de l'ordre Nfft")
plt.legend()
plt.grid(True)
plt.xlim(-300,300)
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig4.png')



# 5- influence du zero-padding ---

zeros_a_ajouter =np.zeros(7 * se_nobs)
szp=np.concatenate((se, zeros_a_ajouter))
Szp=np.fft.fft(szp)
f_axe_zp=np.fft.fftshift(np.fft.fftfreq(len(szp), d=1/fe))
module_Szp_centre=np.fft.fftshift(np.abs(Szp) / se_nobs)

plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
t_zp=np.arange(len(szp))/fe
plt.plot(t_zp,szp,'k')
plt.title('signal temporel szp (se + les zéros rajoutés à la fin)')

plt.subplot(2,1,2)
plt.plot(f_axe_centre, module_SE_centre, 'bo', label='reference se')
plt.plot(f_axe_zp, module_Szp_centre, 'g-', label='TFD Szp (zero-padding manuel)')
plt.title('influence du zero-padding')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(-300,300)
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig5.png')


# 6- influence de la fenetre de ponderation

hann=np.hanning(se_nobs)
sh =se*hann

Sh=np.fft.fft(sh)
module_Sh_centre = np.fft.fftshift(np.abs(Sh) / se_nobs)

plt.figure(figsize=(10, 5))
plt.plot(f_axe_centre, module_SE_centre, 'b', label='fenetre rectangulaire (coupure nette)')
plt.plot(f_axe_centre, module_Sh_centre, 'g', label='fenetre de Hanning')
plt.title('influence de la fenetre de ponderation')
plt.xlim(-300, 300)
plt.legend()
plt.grid(True)
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig6.png')




#  7- influence de la frequence d'echantillonnage fe

index_max=np.argmax(module_SE_centre[se_nobs//2:])
fa=f_axe_centre[se_nobs//2:][index_max]
print(f"frequence du signal détectée : {fa} Hz")


fef=5000.0
tf=np.arange(se_nobs)/fef
sf=np.sin(2 * np.pi * fa * tf) # Création du nouveau signal

Sf=np.fft.fft(sf)
f_axe_f=np.fft.fftshift(np.fft.fftfreq(se_nobs, d=1/fef))
module_Sf_centre=np.fft.fftshift(np.abs(Sf)/se_nobs)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(tf, sf, 'm-o')
plt.title(f'nouveau signal sf temporel fef={fef}Hz')

plt.subplot(2, 1, 2)
plt.plot(f_axe_centre, module_SE_centre, 'b', label=f'reference (fe={fe}Hz)')
plt.plot(f_axe_f, module_Sf_centre, 'm', label=f'haute frequence (fe={fef}Hz)')
plt.title('TFD avec une grande fe mais même nobs')
plt.xlim(-300, 300)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/fifty/PyCharmMiscProject/TNS_s6/COURS/Matieres/TNS/Resultats/Res TP4/fig7.png')


