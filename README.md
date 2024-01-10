MODE D’UTILISATION 
***********************************************************************************

Projet Python : my first Chatbot
Réalisé par : Elias Hafsia L1+
https://github.com/eliasleboss/Projet-python

************************************************************************************

Le projet comporte Deux fichiers  « . py » :
- Main.py :  programme principal (prétraitement du corpus) , appel du menu principal
- Functions.py : Toutes les fonctions sont codées dans « functions.py » ainsi que les trois menus (Menu principal, Menu partie I et Menu Partie II)

Il faut mettre les deux fichiers .py ainsi que le corpus (répertoire ./speeches) dans un même répertoire. puis lancer l'exécution du "main.py"

Lors du lancement du main.py les 8 fichiers se trouvant dans « ./speeches » seront nettoyés grâce à la fonction cleanText(contenu) se trouvant dans le fichier « functions.py ». Le résultat sera enregistré dans un nouveau répertoire : ./cleaned.  Le nom des fichiers cleaned est créé a partir du fichier initial au niveau du main.
L’exécution du « main.py » permet aussi de lancer le menu principal :

******** MENU PRINCIPAL ************ 
1. Accéder aux fonctionnalités de la partie I selon la demande de l’utilisateur
2. Accéder au mode Chatbot permettant à l’utilisateur de poser une question 
3. Quitter


Le Menu I :
******** MENU PARTIE 1 ************ 
1. Afficher la matrice TF_IDF et mots uniques
2. Afficher les mots les moins importants
3. Afficher le(s) mot(s) avec le score TF-IDF le plus élevé
4. Afficher le(s) mot(s) le(s) plus répété(s) par Chirac
5. Afficher le(s) nom(s) du (des) président(s) parlant de la « Nation »
6. Afficher le(s) nom(s) du (des) président(s) parlant du climat et/ou de l'écologie
7. Retour au menu principal
8. Quitter


Le Menu II : 
******** MENU PARTIE 2 ************ 
1. Les termes de la question présents dans le corpus 
2. TF-IDF Vector Question
3. Document qui répond à la question 
4. Le mot le plus pertinent 
5. Générer une réponse à une question 
6. Affiner la réponse à une question 
7. Retour au menu principal
8. Quitter

bugs connus:
mot le plus prononcé: ca dépend du pc, parfois il sort "veut" et d'autre fois "voudrais"
ecologie: quand on lui demande qui, a chaque fois il change qui, avant ca marchait mais pour des raison inconus, ca marche plus
