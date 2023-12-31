# detect_user_building

Grace aux points d'accès d'un batiment, on souhaite qu'un usager puisse connaitre où il se trouve dans ce batiment. Ce projet vise à résoudre ce problème quelque soit la dimension avec laquelle on voit le batiment.

On a :

* `datas`: dossier des fichiers exemple de type `csv` utilisés
* `main.py`: code d'exemple
* `main.ipynb`: notebook de test des algorithmes `(Nous vous conseillons de le lire)`
* `requirements.txt`: fichier de librairies python à installer
* `src`: dossier contenant toutes les fonctions utiles au calcul de la position

On pourait calculer cette position en utilisant :

* les positions des points d'accès avec leurs signaux détectés :

![Test de la détection](images/test2.png "Premier test de détection")

* un ensemble de signaux collectés à des positions précisent du batiment

![Test de la détection](images/test2.png "Deuxième test de détection")
