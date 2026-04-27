Ce document reprend les différents modèles présents dans ce dossier et en explique les contours.
A priori, tous les modèles sont entraînés avec l'ajout du background 'oiseau et vent' de -20 à 0 dB (répartition uniforme aléatoire).

Explication de tous les modèles dans ce dossier:

    -   model_Gunshot_v13_AF_final.pkl: Modèle crée le 25/04, spécialisé dans la détection de gunshot, avec un nombre limité de FP, et donc de fausse alarmes. Utilise toutes les fréquences (AF = all frequencies), probablement beaucoup plus sensible au bruit réel.
    -   model_Gunshot_v25_HF_final.pkl: Modèle crée le 25/04, spécialisé dans la détection de gunshot, avec un nombre limité de FP, et donc de fausse alarmes. Utilise seulement les fréquences dans la plage (5, 19), n'utilise pas les fréquences (0, 4) (HF = high frequencies), probablement moins sensible au bruit réel mais légèrement moins bon.
    -   model_fireworks_v25_AF_final.pkl: Modèle crée le 26/4, spécialisé dans fireworks avec nombre limité de FP, utilisant mean et std
    -   model_fireworks_v25_HF_final.pkl: Modèle crée le 26/4, spécialisé dans fireworks avec nombre limité de FP, utilisant mean et std, légèrement moins performant que la version AF