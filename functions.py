#************************************************************************
# Auteur:  Elias HAFSIA
#  Fichier comportant l'ensemble des Fonctions dévéloppées pour le projet
#*************************************************************************




import os
import re
import math


# création d'une liste des noms de fichier
def list_of_files(directory, extension):
    files_names = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files_names.append(filename)
    return files_names

# afficher une liste
def print_list(files_names):
    print(files_names)

#extraire le nom du fichier
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

#mettre le prénom en fonction du nom
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

# prétraitement du texte
#mettre le texte en miniscule et remplacer les caractères spéciaux et la ponctuation
def cleanText (contenu):

    # Remplacer les caractères spéciaux et les lettres avec accents par leurs équivalents sans accent
    contenu_interE=''.join(['e' if (c=="è" or c=="é" or c=="ê") else c.lower() for c in contenu])
    contenu_interA = ''.join(['a' if (c == "à" or c=="â") else c.lower() for c in contenu_interE])
    contenu_interC = ''.join(['c' if (c == "ç") else c.lower() for c in contenu_interA])
    contenu_interU = ''.join(['u' if (c == "ù") else c.lower() for c in contenu_interC])
    contenu_interpoint = ''.join(['' if (c == "," or c=="." or c=="!" or c=="?" or c=="- " or c=='"'or c==":") else c.lower() for c in contenu_interU])
    contenu_modifie = ''.join([' ' if not (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122 or 48 <=ord(c)<=57)
                                      and (c != '\n') or (c=="-") else c.lower() for c in contenu_interpoint])
    return contenu_modifie



# une fonction qui prend en paramètre une chaine de caractères et qui retourne un dictionnaire
# associant à chaque mot le nombre de fois qu’il apparait dans la chaine de caractères.
def calculer_tf(texte):
    # Utiliser une expression régulière pour extraire les mots du texte
    # et les mettre dans la liste mots
    mots = re.findall(r'\b\w+\b', texte)
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
    #print(idf_scores)
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
    print( "*******Veuillez  attendre qulques secondes le programme calcule ...")
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

# calculer la transposée d'une matrice
def calcul_transposee(matrice):
    nombre_lignes = len(matrice)
    nombre_colonnes = len(matrice[0])

    #remplir une matrice transposee de 0
    transposee = [[0] * nombre_lignes for _ in range(nombre_colonnes)]

    for i in range(nombre_lignes):
        for j in range(nombre_colonnes):
            transposee[j][i] = matrice[i][j]

    return transposee

# afficher la matrice sous la forme de mot_unique vecteur Tf_Idf correspondant
def afficher_matrice_tf_idf(matrice_tf_idf, mots_uniques):
    for i, mot in enumerate(mots_uniques):
        print(f"{mot}: {matrice_tf_idf[i]}")

# liste des mots non importants apartir de la matrice tf_idf
def mots_moins_importants(repertoire,matrice_tf_idf,mots_uniques):
   #matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
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

# les 10 mots avec  le score tf_idf le plus grand
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

    return mots_max_tf_idf


# le mot le plus répété par le président Chirac hormis les mots dits « non importants »
def mots_plus_repetes_chirac(repertoire,mots_non_importants,matrice_tf_idf, mots_uniques):

    #mots_non_importants = mots_moins_importants(repertoire)
    #matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)

    mots_uniques=list(mots_uniques)
    # Identifier le président Chirac dans le corpus
    chirac_index = 0

    # Trouver les mots les plus répétés par Chirac
    mots_plus_repetes = max(mots_uniques, key=lambda mot: matrice_tf_idf[mots_uniques.index(mot)][chirac_index] if mot not in mots_non_importants else 0)

    return mots_plus_repetes

#le président qui a mentionné le plus  le mot nation
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

#les présidents qui ont parlé le plus du climat et écologie
def presidents_mentions_climat_ecologie(repertoire):
    # Calcul de la matrice TF-IDF
    matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
    #print(matrice_tf_idf,mots_uniques)
    mots_uniques = list(mots_uniques)

    # extraire la liste des noms des président
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

#### PARTIE II
# les mots de la question
def tokenize_question(question):
    # appeler la fonction Clean pour nettoyer la question
    #question = cleanText(question)
    # Diviser la question en mots
    words = question.split()
    return words

# Fonction pour identifier les termes de la question  présents dans le corpus
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
                #print(documents)
                #print("************************")
    # Parcourir chaque terme de la question
    for term in question_terms:
        # Vérifier si le terme est présent dans au moins un document du corpus
        if any(term in document for document in documents):
            terms_in_corpus.append(term)

    return terms_in_corpus

#calculer le vecteur tf_idf de la question
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
    #print(question_tfidf_vector)
    for document_vector in tf_idf_matrix:
        #print(document_vector)
        similarity = cosine_similarity(question_tfidf_vector, document_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_document = document_vector
    return most_similar_document


# la norme d'un vecteur et la similarité cosinus entre deux vecteurs:

#produit scalaire de deux vecteurs
def dot_product(vector_a, vector_b):
    # Produit scalaire entre deux vecteurs
    return sum(a * b for a, b in zip(vector_a, vector_b))

#la norme d'un vecteur '
def vector_norm(vector):
    # Norme d'un vecteur
    return math.sqrt(sum(a**2 for a in vector))

# le cosinus de similarité entre deux vecteurs
def cosine_similarity(vector_a, vector_b):
    # Similarité de cosinus entre deux vecteurs
    dot_prod = dot_product(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0  # Éviter une division par zéro

    similarity = dot_prod / (norm_a * norm_b)
    return similarity

#**********************************************
# le document du corpus qui répont le plus à la question
def find_most_relevant_document(tf_idf_matrix, question_tfidf_vector, file_names):
    max_similarity = -1
    most_relevant_document = None

    for i, document_vector in enumerate(tf_idf_matrix):
        similarity = cosine_similarity(question_tfidf_vector, document_vector)

        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_document = file_names[i]

    return most_relevant_document

# le mot du corpus  qui répond le plus à la question
def find_most_relevant_word(question_tfidf_vector, corpus_words):
    # Trouver l'index du mot ayant le score TF-IDF le plus élevé dans le vecteur de la question
    max_tfidf_index = question_tfidf_vector.index(max(question_tfidf_vector))
    #print ("question_tfidf_vector",max_tfidf_index)
    # Retourner le mot correspondant à cet index dans la liste des mots du corpus
    most_relevant_word = corpus_words[max_tfidf_index]

    return most_relevant_word

# la phrase qui représente la réponse à la question dans le document
def generate_response(most_relevant_document, most_relevant_word):
    speeches_path = "./speeches"  # Ajoutez le chemin du sous-répertoire
    mots_separe = re.findall('[A-Z][a-z]*', most_relevant_document)
    most_relevant_document="Nomination_"+mots_separe[-1]+".txt"
    document_path = os.path.join(speeches_path, most_relevant_document)

    # Lire le contenu du document pertinent
    with open(document_path, 'r', encoding='utf-8') as file:
        document_content = file.read()

    # Rechercher la première occurrence du mot dans le document
    word_index = document_content.find(most_relevant_word)

    # Trouver la phrase entourant le mot
    start_sentence = document_content.rfind('.', 0, word_index)
    end_sentence = document_content.find('.', word_index)

    # Extraire la phrase
    response = document_content[start_sentence + 1:end_sentence + 1]

    return response

# la phrase qui répond à la question avec l'ajout de l'expréssion du dictionnaire
def refine_response(question, generated_response):
    # Dictionnaire associant des formes de questions à des modèles de réponses
    question_starters = {
        "Comment": "Après analyse, {}.",
        "Pourquoi": "Car, {}.",
        "Peux-tu": "Oui, bien sûr! {}.",
        "Avez-vous":"Oui, bien sûr! {}.",
    }

    # Recherche de la forme de la question dans le dictionnaire
    for starter, response_pattern in question_starters.items():
        if question.startswith(starter):
            # Formater la réponse en utilisant le modèle associé
            formatted_response = response_pattern.format(generated_response)
            return formatted_response

    # Si la forme de la question n'est pas reconnue, retourner simplement la réponse générée
    return generated_response

#******************************************************************************************
#*************************************  FONCION MENU   *************************************

def menu_principal():
    repertoire = "./cleaned"
    while True:
        print("\n ******** MENU PRINCIPAL ************ ")
        print("1. Accéder aux fonctionnalités de la partie I selon la demande de l’utilisateur")
        print('2. Accéder au mode Chatbot permettant à l’utilisateur de poser une question ')
        print("3. Quitter")
        choix = input("Choisissez une option (1-3): ")

        if choix == '1':
            print("\n ******** MENU 1 ************ ")
            menu1()
        elif choix == '2':
            print("\n ******** MENU 2 ************ ")
            menu2()
        elif choix == "3":
            print("************** AU REVOIR **********!")
            break
        else:
            print("****** Option invalide. Veuillez choisir une option entre 1 et 3.*************")


def menu1():
    repertoire = "./cleaned"
    while True:
        print("\n ******** MENU PARTIE 1 ************ ")
        print("1. Afficher la matrice TF_IDF et mots uniques")
        print("2. Afficher les mots les moins importants")
        print("3. Afficher le(s) mot(s) avec le score TF-IDF le plus élevé")
        print("4. Afficher le(s) mot(s) le(s) plus répété(s) par Chirac")
        print("5. Afficher le(s) nom(s) du (des) président(s) parlant de la « Nation »")
        print("6. Afficher le(s) nom(s) du (des) président(s) parlant du climat et/ou de l'écologie")
        print("7. Retour au menu principal")
        print("8. Quitter")

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
            matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
            mots_non_importants=mots_moins_importants(repertoire,matrice_tf_idf,mots_uniques)
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
            matrice_tf_idf, mots_uniques = calcul_tf_idf(repertoire)
            mots_non_importants = mots_moins_importants(repertoire,matrice_tf_idf,mots_uniques)
            mots_plus_repetes=mots_plus_repetes_chirac(repertoire,mots_non_importants,matrice_tf_idf, mots_uniques)
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

            liste_presidents_climat_ecologie = presidents_mentions_climat_ecologie(repertoire)
            print(liste_presidents_climat_ecologie)
        elif choix == '7':
            menu_principal()
        elif choix == '8':
            print("************** AU REVOIR **********!")
            break
        else:
            print("****** Option invalide. Veuillez choisir une option entre 1 et 7.*************")

def menu2():
    repertoire = "./cleaned"
    while True:
        print("\n ******** MENU PARTIE 2 ************ ")
        print("1. Les termes de la question  présents dans le corpus ")
        print("2. TF-IDF Vector Question")
        print("3. Document qui répond à la question ")
        print("4. Le terme le plus pertinent du document qui répond à la question")
        print("5. Générer une réponse à une question ")
        print("6. Affiner la  réponse à une question ")
        print("7. Retour au menu principal")
        print("8. Quitter")

        choix = input("Choisissez une option (1-8): ")

        if choix == '1':
            question = "Peux-tu me dire comment une nation peut-elle prendre soin du climat ?"
            print("Question: ", question)
            # Tokenisation de la question
            question_terms = tokenize_question(question)

            # Identifier les termes présents dans le corpus
            terms_in_corpus = find_terms_in_corpus(question_terms, repertoire)

            print("Question terms:", question_terms)
            print("les mots de la question présents dans le corpus sont: ", terms_in_corpus)

        elif choix == '2':
            idf_scores=calculer_idf(repertoire)
            #print("idf_scores",idf_scores)
            matrice_tf_idf, corpus_words = calcul_tf_idf(repertoire)
            corpus_words=list(corpus_words)
            #print(len(corpus_words))
            question = "Peux_tu me dire comment une nation peut_elle prendre soin du climat ?"
            print("la question est: ",question)
            # Tokenisation de la question
            question_words=tokenize_question(question)
            print("question_words",question_words)
            question_tfidf_vector=calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            print("Le Vecteur TF_IDF de la Question",question_tfidf_vector)

        elif choix=="3":
            files_names = list_of_files(repertoire, "txt")
            idf_scores = calculer_idf(repertoire)
            question = "Peux_tu me dire comment une nation peut_elle prendre soin du climat ?"
            print ("la question: ",question)
            # Tokenisation de la question
            question_words = tokenize_question(question)
            print("question_words: ", question_words)
            tf_idf_matrix, corpus_words = calcul_tf_idf(repertoire)
            tf_idf_matrixT = calcul_transposee(tf_idf_matrix)
            corpus_words = list(corpus_words)
            question_tfidf_vector = calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            most_relevant_document = find_most_relevant_document(tf_idf_matrixT, question_tfidf_vector, files_names)
            print("Le document qui répond à la question est : ",most_relevant_document)

        elif choix == '4':
            files_names = list_of_files(repertoire, "txt")
            idf_scores = calculer_idf(repertoire)
            question = "Peux-tu me dire une nation peut_elle prendre soin du climat ?"
            # question="Comment une nation peut-elle prendre soin du climat ?"
            print("la question: ", question)
            question_words = tokenize_question(question)
            print("question_words: ", question_words)
            tf_idf_matrix, corpus_words = calcul_tf_idf(repertoire)
            tf_idf_matrixT = calcul_transposee(tf_idf_matrix)
            corpus_words = list(corpus_words)
            question_tfidf_vector = calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            most_relevant_document = find_most_relevant_document(tf_idf_matrixT, question_tfidf_vector, files_names)
            print("Le document qui répond à la question est : ", most_relevant_document)
            most_relevant_word = find_most_relevant_word(question_tfidf_vector, corpus_words)
            print("Le mot le plus pertinent est :", most_relevant_word)

        elif choix == '5':
            files_names = list_of_files(repertoire, "txt")
            idf_scores = calculer_idf(repertoire)
            question = "Peux-tu me dire une nation peut_elle prendre soin du climat ?"
            #question="Comment une nation peut-elle prendre soin du climat ?"
            print("la question: ", question)
            # Tokenisation de la question
            question_words = tokenize_question(question)
            print("question_words: ", question_words)
            tf_idf_matrix, corpus_words = calcul_tf_idf(repertoire)
            tf_idf_matrixT = calcul_transposee(tf_idf_matrix)
            corpus_words = list(corpus_words)
            question_tfidf_vector = calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            most_relevant_document = find_most_relevant_document(tf_idf_matrixT, question_tfidf_vector, files_names)
            print("Le document qui répond à la question est : ",most_relevant_document)
            most_relevant_word= find_most_relevant_word(question_tfidf_vector, corpus_words)
            print("Le mot le plus pertinent est :", most_relevant_word)
            response_generee = generate_response(most_relevant_document, most_relevant_word)
            print("Réponse générée :", response_generee)
            #reponse_rafine =refine_response(question, response_generee)
            #print( "la reponse raffinée est : ", reponse_rafine)

        elif choix == '6':
            files_names = list_of_files(repertoire, "txt")
            idf_scores = calculer_idf(repertoire)
            #question = "Peux-tu me dire une nation peut_elle prendre soin du climat ?"
            question="Comment une nation peut-elle prendre soin du climat ?"
            print("la question: ", question)
            # Tokenisation de la question
            question_words = tokenize_question(question)
            print("question_words: ", question_words)
            tf_idf_matrix, corpus_words = calcul_tf_idf(repertoire)
            tf_idf_matrixT = calcul_transposee(tf_idf_matrix)
            corpus_words = list(corpus_words)
            question_tfidf_vector = calculate_question_tfidf_vector(question_words, idf_scores, corpus_words)
            most_relevant_document = find_most_relevant_document(tf_idf_matrixT, question_tfidf_vector, files_names)
            print("Le document qui répond à la question est : ", most_relevant_document)
            most_relevant_word = find_most_relevant_word(question_tfidf_vector, corpus_words)
            print("Le mot le plus pertinent est :", most_relevant_word)
            response_generee = generate_response(most_relevant_document, most_relevant_word)
            #print("Réponse générée :", response_generee)
            reponse_rafine = refine_response(question, response_generee)
            print("la reponse raffinée est : ", reponse_rafine)
        elif choix == '7':
            menu_principal()

        elif choix == '8':
            print("************** AU REVOIR **********!")
            break

        else:
            print("****** Option invalide. Veuillez choisir une option entre 1 et 6.*************")


#*************************************  BONUS ***********************************************

# plus de prétraitement
def replace_special_characters(texte):
    replacements = {"l’": "le ", "la’": "la ", "qu’": "que ", "qui’": "qui "}
    for old, new in replacements.items():
        texte = texte.replace(old, new)
    return texte
