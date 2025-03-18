import streamlit as st

st.title("Questions-Réponses Lab4")

st.markdown("""
## Rappel des objectifs

- Comprendre comment les réseaux de neurones peuvent approximer l'inverse d'une fonction
- Comprendre les modèles de diffusion et leurs applications dans la génération d'images
- Créer un modèle de diffusion basique pour générer des images

---

## Réseaux Neurones et Inversion de Fonction

- **Les réseaux neuronaux peuvent-ils apprendre à estimer l'inverse de y=sin(x) ?**
  - Oui, en entraînant un modèle pour approximer x=arcsin(y), les réseaux neuronaux peuvent apprendre cet inverse. Cependant, cela ne fonctionne que dans la plage où la fonction arcsin est définie. 

- **Que se passe-t-il pour les valeurs en dehors de la plage [-1,1] ?**
  - Comme l'arcsin n'est défini que pour y dans [-1,1], tenter d'approximer l'inverse pour des valeurs en dehors de cette plage pourrait conduire à un comportement indéfini ou à des estimations incorrectes.

- **Implications pour des fonctions plus complexes :**
  - Approximativement les inverses pour des fonctions plus complexes peut nécessiter une attention particulière au domaine de la fonction et aux singularités potentielles ou au comportement non bijectif.

##### _Il faut donc retenir qu'il est important de faire attention aux propriétés des fonctions que l'on utilise dans l'architecture d'un réseau de neurones. Ici nous voulons que les fonctions utilisées soient inversibles_
---

## Modèles de Diffusion pour la Génération d'Images

- **Que se passe-t-il si nous changeons la fonction de bruitage ?**
  - La fonction de bruitage détermine le rythme et la manière dont le bruit est ajouté et supprimé. Le modifier peut affecter la convergence et la qualité des images générées ! Un ensemble de petits pats sont préférables à moins de grands sauts. 

- **Comment les modèles de diffusion se comparent-ils aux GANs ?**
  - Les modèles de diffusion affinent les images de manière itérative, ce qui peut offrir un entraînement plus stable, tandis que les GANs utilisent un entraînement adversarial qui peut être plus rapide mais plus sujet à l'instabilité.

##### _Les modèles de diffusion se reposent sur l'idée que chaque itération de débruitage ne dépend que de l'itération précédente. Ainsi, chaque itération affine progressivement l'image. Cela peut être plus stable que les GANs, qui peuvent être sujets à des problèmes de convergence._

---
            

## Réflexion Finale

- **Comment l'inversion de fonction est-elle liée aux modèles de diffusion ?**
  - Les deux impliquent des processus d'apprentissage où les réseaux neuronaux sont entraînés à approximer des transformations – les fonctions d'inversion reposent sur l'apprentissage d'un 'inverse', tandis que les modèles de diffusion apprennent des raffinements itératifs.

- **Comment la suppression itérative de bruit aide-t-elle ?**
  - La suppression itérative de bruit aide à générer des images en affinant progressivement une entrée bruyante en une représentation claire. On utilise la logique markovienne pour affirmer la convergence. Il est possible de faire un parrrallèle avec le Dynamic Programming, car ici également nous cherchons à diviser une image et ses caractérisqtiues en plusieurs problèmes séparés dans l'espace (grille) mais aussi dans le temps (steps).

- **Applications potentielles des modèles de diffusion :**
  - Génération d'images, de vidéos, données synthétiques, restauration d'images, etc
""")