import os
import re
import math


def list_of_files(directory, extension):
    files_names = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files_names.append(filename)
    return files_names

def print_list(files_names):
    print(files_names)

def extractName(files_names):
    l= files_names
    l1= []
    for i in range(len(l)):
        f=l[i]
        f = f.split("_")[-1]
        f = f = f.split(".")[0]
        # Vérification si le dernier caractère est un chiffre
        if f[-1].isdigit():
            # Si c'est le cas, supprimez le dernier caractère
            f =f.split(f[-1])[0]
        l1.append(f)
    # supprimer les doublons
    l1= list(set(l1))
    return l1

def putSurname(l):

    for i in range (len(l)):
        if l[i]=="Chirac":
            l[i]= "Jacques Chirac"
        elif l[i]=="Macron":
            l[i] ='Emmanuel Macron'
        elif l[i]=="Hollande":
            l[i] ='François Hollande'
        elif l[i]=="Mitterrand":
            l[i] ='François Mitterrand'
        elif l[i]=="Sarkozy":
            l[i] ='Nicolas Sarkozy'
        elif l[i]=="Giscard dEstaing":
            l[i] ='Valery Giscard dEstaing'
    return l


def cleanText (contenu):
    #print(fileInit)
    #recupération du nom de fichier cleaned
    #f = fileInit.split("/")[-1]
    #f = f.split("_")[-1]

    #with open(fileInit, 'r', encoding='utf-8') as fileInit:
        #contenu = fileInit.read()

    # Remplacer les caractères spéciaux et les lettres avec accents par leurs équivalents sans accent
    contenu_interE=''.join(['e' if (c=="è" or c=="é" or c=="ê") else c.lower() for c in contenu])
    contenu_interA = ''.join(['a' if (c == "à" or c=="â") else c.lower() for c in contenu_interE])
    contenu_interC = ''.join(['c' if (c == "ç") else c.lower() for c in contenu_interA])
    contenu_interU = ''.join(['u' if (c == "ù") else c.lower() for c in contenu_interC])
    contenu_interpoint = ''.join(['' if (c == "," or c=="." or c=="!" or c=="?" or c=="- " or c=='"'or c==":") else c.lower() for c in contenu_interU])
    contenu_modifie = ''.join([' ' if not (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122 or 48 <=ord(c)<=57)
                                      and (c != '\n') or (c=="-") else c.lower() for c in contenu_interpoint])
    return contenu_modifie

    #chemin_sortie = "cleaned/Lower"+f
    #with open(chemin_sortie, 'w', encoding='utf-8') as fileClean:
        #fileClean.write(contenu_modifie)


# une fonction qui prend en paramètre une chaine de caractères et qui retourne un dictionnaire
# associant à chaque mot le nombre de fois qu’il apparait dans la chaine de caractères.
def calculer_tf(texte):
    # Utiliser une expression régulière pour extraire les mots du texte
    # et les mettre dans la liste mots
    mots = re.findall(r'\b\w+\b', texte)
    # Utiliser Counter pour compter le nombre d'occurrences de chaque mot
    #créer un dico vide
    dico_mot_occurence = {}
    # compter le nombre d'occurrences de chaque mot
    for mot in mots:
        if mot != "":
            if mot in dico_mot_occurence:
                dico_mot_occurence[mot] += 1
            else:
                dico_mot_occurence[mot] = 1
    return(dico_mot_occurence)

# une fonction qui prend en paramètre le répertoire où se trouve l’ensemble des fichiers du corpus
# et qui retourne un dictionnaire associant à chaque mot son score IDF (Inverse Document Frequency).

# Parcourir tous les fichiers du corpus.
# 1- Pour chaque fichier, utiliser la fonction précédente (calculer_tf) pour obtenir les occurrences de chaque mot dans le fichier.
# 2- Maintenir une liste des mots uniques rencontrés dans tous les fichiers.
# 3- Pour chaque mot unique, calculer son score IDF en utilisant la formule


def calculer_idf(repertoire_corpus):
    # Liste pour stocker le nombre de documents contenant chaque mot
    documents_contenant_mot = {}
    # Compteur pour le nombre total de documents
    nombre_documents_total = 0

    # Parcourir tous les fichiers du corpus
    for fichier in os.listdir(repertoire_corpus):
        chemin_fichier = os.path.join(repertoire_corpus, fichier)

        # Vérifier si le chemin correspond à un fichier (et non à un sous-répertoire, par exemple) et que  c'est un fichier texte
        if os.path.isfile(chemin_fichier) and fichier.endswith('.txt'):
            # Mettre à jour le nombre total de documents
            nombre_documents_total += 1

            # Lire le contenu du fichier
            with open(chemin_fichier, 'r', encoding='utf-8') as file:
                contenu_fichier = file.read()

                # Calculer les occurrences pour chaque mot dans le fichier
                dico_mot_occurence = calculer_tf(contenu_fichier)

                # Mettre à jour le nombre de documents contenant chaque mot

                for mot in dico_mot_occurence:
                    if mot in documents_contenant_mot:
                        documents_contenant_mot[mot] += 1
                    else:
                        documents_contenant_mot[mot] = 1
    #print(documents_contenant_mot)
    # Calculer le score IDF pour chaque mot
    idf_scores = {mot: math.log10(nombre_documents_total / dico_mot_occurence) for mot, dico_mot_occurence in documents_contenant_mot.items()}
    print(idf_scores)
    #print("\n")
    return idf_scores


#CALCULER TF_IDF
# Génération de la matrice TF-IDF à partir du corpus de documents :
#
# Utiliser la fonction calculer_tf pour obtenir les occurrences de chaque mot dans chaque document.
# Utiliser la fonction calculer_idf pour obtenir les scores IDF pour chaque mot.
# Calculer les scores TF-IDF en multipliant les scores TF par les scores IDF.
# Organiser ces scores dans une matrice où chaque ligne représente un mot et chaque colonne représente un document.


def calcul_tf_idf(repertoire):
    documents = []
    mots_uniques = set()

    for nom_fichier in os.listdir(repertoire):
        chemin_fichier = os.path.join(repertoire, nom_fichier)

        if os.path.isfile(chemin_fichier) and nom_fichier.endswith('.txt'):
            with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                contenu = fichier.read()
                documents.append(contenu)
                mots_uniques.update(set(re.findall(r'\b\w+\b', contenu)))

    idf_scores = calculer_idf(repertoire)

    matrice_tf_idf = []

    for mot in mots_uniques:
        vecteur_tf_idf = []

        for document in documents:
            dico_mot_occurence = calculer_tf(document)

            tf = dico_mot_occurence.get(mot, 0) / len(re.findall(r'\b\w+\b', document))
            tf_idf = tf * idf_scores.get(mot, 0)
            vecteur_tf_idf.append(tf_idf)

        matrice_tf_idf.append(vecteur_tf_idf)


    return matrice_tf_idf, mots_uniques

def calcul_transposee(matrice):
    nombre_lignes = len(matrice)
    nombre_colonnes = len(matrice[0])

    #remplir une matrice transposee de 0
    transposee = [[0] * nombre_lignes for _ in range(nombre_colonnes)]

    for i in range(nombre_lignes):
        for j in range(nombre_colonnes):
            transposee[j][i] = matrice[i][j]

    return transposee

def afficher_matrice_tf_idf(matrice_tf_idf, mots_uniques):
    for i, mot in enumerate(mots_uniques):
        print(f"{mot}: {matrice_tf_idf[i]}")

def mots_moins_importants(repertoire):
   matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
   mots_non_importants = []

   for i, mot in enumerate(mots_uniques):

       score_tfidf_moyen=0

       total=0
       #calcul de la moyenne de chaque ligne de la matrice tf_idf
       for j in range (len(matrice_tf_idf[i])) :
           #print(matrice_tf_idf[i][j])
           total=total+matrice_tf_idf[i][j]
           score_tfidf_moyen=total/len(matrice_tf_idf[i])
           #score_tfidf_moyen = sum(matrice_tf_idf[i][j]) /len(matrice_tf_idf[i])

       #print(score_tfidf_moyen)
       #si la moyenne est égale à 0
       if score_tfidf_moyen == 0:
           #ajouter le mot dans la liste
            mots_non_importants.append(mot)

   return mots_non_importants


def mot_max_tfidf(repertoire):
    matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
    mots_max_tf_idf = []
    list_score={}
    dico_max_trie={}
    for i, mot in enumerate(mots_uniques):
        total = 0
        # calcul de la somme de chaque ligne de la matrice tf_idf
        for j in range(len(matrice_tf_idf[i])):
            # print(matrice_tf_idf[i][j])
            total = total + matrice_tf_idf[i][j]
            # score_tfidf_moyen = sum(matrice_tf_idf[i][j]) /len(matrice_tf_idf[i])
        list_score[mot]=total
    # Trier le dictionnaire par les valeurs dans l'ordre décroissant
    dico_max_trie = dict(sorted(list_score.items(), key=lambda item: item[1], reverse=True))

    # Prendre les 10 premiers éléments du dictionnaire trié
    elements_max = list(dico_max_trie.items())[:10]

    # Afficher les 10 éléments avec les valeurs maximales
    for cle, valeur in elements_max:
        #print(f"{cle}: {valeur}")
        mots_max_tf_idf.append(cle)




    # Trouver la clé associée à la valeur maximale
    #cle_max = max(list_score, key=list_score.get)
    # Afficher la clé et la valeur maximale
    #print(f"Clé maximale : {cle_max}")
    #print(f"Valeur maximale : {list_score[cle_max]}")

        #if score_tfidf_moyen == 0:
            # ajouter le mot dans la liste
            #mots_non_importants.append(mot)
    return mots_max_tf_idf


# 3. Indiquer le(s) mot(s) le(s) plus répété(s) par le président Chirac hormis les mots dits « non importants »
def mots_plus_repetes_chirac(repertoire):

    mots_non_importants = mots_moins_importants(repertoire)
    matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)

    mots_uniques=list(mots_uniques)
    # Identifier le président Chirac dans le corpus
    chirac_index = 0

    # Trouver les mots les plus répétés par Chirac
    mots_plus_repetes = max(mots_uniques, key=lambda mot: matrice_tf_idf[mots_uniques.index(mot)][chirac_index] if mot not in mots_non_importants else 0)

    return mots_plus_repetes


def presidents_mentions_nation_repertoire(repertoire):
    # Calcul de la matrice TF-IDF
    matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)

    mots_uniques = list(mots_uniques)
    # extraire la iste des noms des président
    directory = "./speeches"
    files_names = list_of_files(directory, "txt")
    noms_presidents=extractName(files_names)
    #print(noms_presidents)

    # Index du mot "Nation" dans la liste des mots uniques
    index_nation = mots_uniques.index("nation")

    # Dictionnaire pour stocker le nombre de mentions de "Nation" par président
    mentions_nation = {}

    # Parcourir les colonnes de la matrice TF-IDF correspondant à chaque président
    for i, nom_president in enumerate(noms_presidents):
        # Récupérer le score TF-IDF pour le mot "Nation" et le président actuel
        score_tfidf_nation = matrice_tf_idf[index_nation][i]

        # Stocker le nombre de mentions de "nation" pour ce président
        mentions_nation[nom_president] = score_tfidf_nation

    # Trouver le(s) président(s) qui a(ont) parlé de "Nation" et le(s) plus répété(s)
    presidents_mentions_max = [nom for nom, mentions in mentions_nation.items() if
                               mentions == max(mentions_nation.values())]


    return presidents_mentions_max

def presidents_mentions_climat_ecologie(repertoire):
    # Calcul de la matrice TF-IDF
    matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
    print(matrice_tf_idf,mots_uniques)
    mots_uniques = list(mots_uniques)

    # extraire la iste des noms des président
    directory = "./speeches"
    files_names = list_of_files(directory, "txt")
    noms_presidents = extractName(files_names)
    # print(noms_presidents)

    # Mots clés liés au climat et à l'écologie
    mots_cle_climat_ecologie = ["climat","climatique","ecologique", "environnement", "developpement durable"]

    # Dictionnaire pour stocker le nombre de mentions de climat/écologie par président
    mentions_climat_ecologie = {nom_president: 0 for nom_president in noms_presidents}

    # Parcourir les colonnes de la matrice TF-IDF correspondant à chaque président
    for i, nom_president in enumerate(noms_presidents):
        score_tfidf_mots_cle = 0
        #print(nom_president)
        # Récupérer le score TF-IDF pour les mots clés liés au climat et à l'écologie
        for mot in mots_cle_climat_ecologie:
            score_mot = matrice_tf_idf[mots_uniques.index(mot)][i] if mot in mots_uniques else 0

            score_tfidf_mots_cle += score_mot
            #print(mot,score_tfidf_mots_cle)
            # Ajouter le score au dictionnaire des mentions
        mentions_climat_ecologie[nom_president] += score_tfidf_mots_cle

        # Filtrer les présidents ayant mentionné le climat et/ou l'écologie
    presidents_mentions_climat_ecologie = [nom for nom, mentions in mentions_climat_ecologie.items() if mentions > 0]


    return presidents_mentions_climat_ecologie

# les mots de la question
def tokenize_question(question):
    # Convertir la question en minuscules
    question = cleanText(question)

    # Supprimer la ponctuation
    #question = question.translate(str.maketrans("", "", string.punctuation))

    # Diviser la question en mots
    words = question.split()

    return words


# Fonction pour identifier les termes présents dans le corpus
def find_terms_in_corpus(question_terms, repertoire):
    # Initialiser une liste pour stocker les termes présents dans le corpus
    terms_in_corpus = []
    documents=[]
    for nom_fichier in os.listdir(repertoire):
        chemin_fichier = os.path.join(repertoire, nom_fichier)

        if os.path.isfile(chemin_fichier) and nom_fichier.endswith('.txt'):
            with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                contenu = fichier.read()
                documents.append(contenu)
                print(documents)
                print("************************")
    # Parcourir chaque terme de la question
    for term in question_terms:
        # Vérifier si le terme est présent dans au moins un document du corpus
        if any(term in document for document in documents):
            terms_in_corpus.append(term)

    return terms_in_corpus

def calculate_question_tfidf_vector(question_words, idf_scores, corpus_words):
    # Initialiser le vecteur TF-IDF de la question avec des zéros
    question_tfidf_vector = [0] * len(corpus_words)

    # Calculer le score TF pour chaque mot de la question
    tf_scores = {word: question_words.count(word) for word in question_words}

    # Remplir le vecteur TF-IDF en utilisant les scores TF, IDF et la liste de mots du corpus
    for word, tf_score in tf_scores.items():
        if word in idf_scores and word in corpus_words:
            # Index du mot dans le vecteur TF-IDF
            word_index = corpus_words.index(word)
            question_tfidf_vector[word_index] = tf_score * idf_scores[word]

    return question_tfidf_vector

"*****************************************************************************************"
#pour comparer le vecteur de la question avec chaque vecteur dans la matrice TF-IDF
# et trouver celui avec la plus haute similarité de cosinus.

def find_most_similar_document(question_tfidf_vector, tf_idf_matrix):
    # Comparer la similarité de la question avec chaque document dans la matrice TF-IDF
    max_similarity = -1
    most_similar_document = None

    for document_vector in tf_idf_matrix:
        similarity = cosine_similarity(question_tfidf_vector, document_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_document = document_vector


    return most_similar_document


# la norme d'un vecteur et la similarité cosinus entre deux vecteurs:

def dot_product(vector_a, vector_b):
    # Produit scalaire entre deux vecteurs
    return sum(a * b for a, b in zip(vector_a, vector_b))

def vector_norm(vector):
    # Norme d'un vecteur
    return math.sqrt(sum(a**2 for a in vector))

def cosine_similarity(vector_a, vector_b):
    # Similarité de cosinus entre deux vecteurs
    dot_prod = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0  # Éviter une division par zéro

    similarity = dot_prod / (norm_a * norm_b)
    return similarity




#******************************************************************************************
#*************************************  FONCION MENU   *************************************
def menu_principal():
    repertoire = "./cleaned"
    while True:
        print("\n ******** MENU PRINCIPAL ************ ")
        print("1. Afficher la matrice TF_IDF et mots uniques")
        print("2. Afficher les mots les moins importants")
        print("3. Afficher le(s) mot(s) avec le score TF-IDF le plus élevé")
        print("4. Afficher le(s) mot(s) le(s) plus répété(s) par Chirac")
        print("5. Afficher le(s) nom(s) du (des) président(s) parlant de la « Nation »")
        print("6. Afficher le(s) nom(s) du (des) président(s) parlant du climat et/ou de l'écologie")
        print("7. Mots de la question présents dans le corpus")
        print("8. TF-IDF Vector Question")
        print("10. SIMILARITY Question et corpus ")
        print("11. Quitter")

        choix = input("Choisissez une option (1-8): ")

        if choix == '1':

            print("****************************************************************************************")
            print("**************************LA MATRICE TF_IDF et Mot Unique EST: **************************")
            print("****************************************************************************************")
            matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
            afficher_matrice_tf_idf(matrice_tf_idf , mots_uniques)

        elif choix == '2':
            print("****************************************************************************************")
            print("********************** LES MOTS LES MOINS IMPORTANTS DANS LE CORPUS SONT : *************")
            print("****************************************************************************************")
            mots_non_importants=mots_moins_importants(repertoire)
            print(mots_non_importants)

        elif choix == '3':
            print("****************************************************************************************")
            print("********************** LES 10 MOTS AVEC MAX TF_IFD DANS LE CORPUS SONT : *************")
            print("****************************************************************************************")

            mots_max_tf_idf= mot_max_tfidf(repertoire)
            print(mots_max_tf_idf)

        elif choix == '4':
            print("****************************************************************************************")
            print("********************** LE MOT LE PLUS REPETE PAR CHIRAC: *************")
            print("****************************************************************************************")

            mots_plus_repetes=mots_plus_repetes_chirac(repertoire)
            print(f"Mot le plus répété par le président Chirac: {mots_plus_repetes}")

        elif choix == '5':
            print("****************************************************************************************")
            print("********************** LE PRESIDENT QUI MENTIONNE LE PLUS LE MOT NATION: *************")
            print("****************************************************************************************")

            presidents_mentions_max= presidents_mentions_nation_repertoire(repertoire)
            print(presidents_mentions_max)

        elif choix == '6':
            print("****************************************************************************************")
            print("********************** LE PRESIDENT QUI MENTIONNE climat_ecologie: *************")
            print("****************************************************************************************")

            liste_presidents_climat_ecologie= presidents_mentions_climat_ecologie(repertoire)
            print(liste_presidents_climat_ecologie)

        elif choix == '7':
            question = "Peux-tu me dire comment une nation peut-elle prendre soin du climat ?"
            # Tokenisation de la question
            question_terms = tokenize_question(question)

            # Identifier les termes présents dans le corpus
            terms_in_corpus = find_terms_in_corpus(question_terms, repertoire)

            print("Question terms:", question_terms)
            print("Terms in corpus:", terms_in_corpus)

        elif choix == '8':
            idf_scores=calculer_idf(repertoire)
            #print("idf_scores",idf_scores)
            matrice_tf_idf, corpus_words = calcul_tf_idf(repertoire)
            corpus_words=list(corpus_words)
            #print(len(corpus_words))
            question = "Peux_tu me dire comment une nation peut_elle prendre soin du climat ?"
            # Tokenisation de la question
            question_words=tokenize_question(question)
            print("question_words",question_words)
            question_tfidf_vector=calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            print(question_tfidf_vector)

        elif choix == '9':
            print("************** AU REVOIR **********!")
            break
        else:
            print("****** Option invalide. Veuillez choisir une option entre 1 et 7.*************")