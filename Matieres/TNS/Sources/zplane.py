# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:13:26 2024
@brief : diagramme des pôles et des zéros (DPZ) + réponse en fréquence (RF)
         (module linéaire) + RF (phase OU module en dB)
@param : G, b, a, whole, TFname, whole, phase :
         FT décrite par liste OU polynômes Num/Den OU formes sos (remplir
         avec [] pour les paramètres non-utilisés), TFname liste des intitulés
         des FT (indispensable !!!!), whole optionnel (False/True), phase
         optionnelle (True : phase / False : module en dB)
@return : Z, P, K, H_amp, nu : listes de zéros, pôles, gains statiques, RF,
         fréquence normalisée
@author : Nicolas SIMOND
exemple : Z1, P1, K1, H1_amp, nu1 = zplane(Hz1, [],[], [r"$H_{1_a}$", r"$H_{1_b}$", r"$H_{1_c}$"])
exemple : Z1, P1, K1, H1_amp, nu1 = zplane([], [b0, b1, b2],[a0, a1, a2], [r"$H_{1_a}$", r"$H_{1_b}$", r"$H_{1_c}$"])
nécessite dans le script principal : from zplane import zplane
passer True à False la ligne matplotlib.rcParams... si pas de compilateur Latex sur votre machine ...
"""
from matplotlib.pyplot import axis, Circle, figure, GridSpec, minorticks_on, show, tight_layout, xlim
from numpy import abs, angle, imag, log10, pi, poly, real, size
from scipy.signal import freqz, sos2zpk, tf2zpk
import matplotlib
matplotlib.rcParams["text.usetex"] = True   # insertion caractères Latex

def zplane(G=[], b=[], a=[], TFname="", whole=False, phase=True):
    indice = [r'$_a$', r'$_b$', r'$_c$', r'$_d$', r'$_e$', r'$_f$', r'$_g$', r'$_h$', r'$_i$', r'$_j$']

    # Crée la grille de sous-graphiques
    fig = figure(figsize=(12, 8))
    gs  = GridSpec(2, 2, figure=fig)

    # Trace le cercle unité pour référence
    ax1         = fig.add_subplot(gs[:, 0])
    unit_circle = Circle((0, 0), 1, color='gray', fill=False, linestyle='dotted')
    ax1.add_artist(unit_circle)
    axis([-1.1, 1.1, -1.1, 1.1])
    ax1.set_title('Diagramme des Pôles et des Zéros')
    ax1.set_xlabel('Partie Réelle')
    ax1.set_ylabel('Partie Imaginaire')

    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_aspect('equal', 'box')

    # Configure le tracé module de la réponse en fréquence
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Module des réponses en fréquence')
    ax2.set_xlabel(r"$\nu$")
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, which='major', linestyle='-', linewidth=0.5)
    minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', linewidth=0.5)

    # Configure le tracé de la phase OU du module_dB de la réponse en fréquence
    ax3 = fig.add_subplot(gs[1, 1])
    if phase==True :
        ax3.set_title('Phase des réponses en fréquence')
        ax3.set_ylabel('Phase [°]')
    else:
        ax3.set_title('Module des réponses en fréquence')
        ax3.set_ylabel('Amplitude [dB]')
    ax3.set_xlabel(r"$\nu$")
    ax3.grid(True, which='major', linestyle='-', linewidth=0.5)
    minorticks_on()
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.5)

    Z       = []
    P       = []
    K       = []
    H_amp   = []
    for i in range(len(TFname)):
        """Trace le diagramme des pôles et des zéros"""
        # Obtenir les pôles, zéros et gain du filtre
        if(len(G) > 0):
            zero, pole, gain = tf2zpk(G[i].num, G[i].den)
            nu, H = freqz(G[i].num, G[i].den, 1024, True, fs=1)
        elif (len(b) > 0):
            zero, pole, gain = tf2zpk(b[i], a[i])
            nu, H = freqz(b[i], a[i], 1024, True, fs=1)
        else:
            # if (size(sos[i]) == 6):
            #     tempo = sos[i].reshape(1, 6)
            #     zero, pole, gain = sos2zpk(tempo)
            # else:
            #     zero, pole, gain = sos2zpk(sos[i])
            bi      = poly(zero)
            ai      = poly(pole)
            nu, H   = freqz(bi, ai, 1024, True, fs=1)

        # normalisation du gain
        g0  = static_gain(H[:512])
        H   = H/g0

        Z.append(zero)
        P.append(pole)
        K.append(real(gain))
        H_amp.append(H)

        # Tracer les pôles et zéros
        ax1.scatter(real(zero), imag(zero), color='C' + f"{i+1}", facecolors='none', label=r"$z$" + indice[i])
        ax1.scatter(real(pole), imag(pole), color='C' + f"{i+1}", marker='x', label=r'$p$' + indice[i])
        ax1.legend()

        # Module de la réponse en fréquence
        ax2.plot(nu, abs(H), color='C' + f"{i+1}", label=r"$\vert$" + TFname[i] + r"$(\nu)\vert$")
        if whole==False :
            ax2.set_xlim((0, 0.5))
        ax2.legend()

        # Phase ou module en dB de la réponse en fréquence
        if phase == True:
            ax3.plot(nu, angle(H) * 180 / pi, color='C' + f"{i + 1}", label="$arg[$" + TFname[i] + r"$(\nu)]$")
        else:
            ax3.plot(nu, 20 * log10(abs(H) + 2 ** (-16)), color='C' + f"{i + 1}",
                     label=r"$\vert$" + TFname[i] + r"$_{dB}(\nu)$" + r"$\vert$")
        if whole==False :
            xlim((0, 0.5))
        ax3.legend()

    tight_layout()
    show()
    return Z, P, K, H_amp, nu

from numpy import abs
def static_gain(amp):
    gain    = abs(amp)
    maxi    = [gain[0], gain.max(), gain[-1]]

    if gain[0]==0 :
        maxi[0] = 10**-64
    if gain[2]==0 :
        maxi[2] = 10**-64
    comp    = maxi[0]/maxi[2]

    # passe-bande
    if (maxi[1] / maxi[0] > 100) and (maxi[1] / maxi[2] > 100) :
        g0 = maxi[1]
    # passe-bas
    elif (maxi[1]==maxi[0]) or (comp>2) :
        g0  = maxi[0]
    # passe-haut
    elif (maxi[1]==maxi[2]) or (comp<.5) :
        g0 = maxi[2]
    # coupe-bande
    else :
        g0 = maxi[0]

    return g0

