import os
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
    for i in range (len(l)):
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
    #ajoute le prenom au nom du president
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


def cleanText (fileInit):
    #recupération du nom de fichier cleaned
    f = fileInit.split("/")[-1]
    print(f)
    f = f.split("_")[-1]

    with open(fileInit, 'r', encoding='utf-8') as fileInit:
        contenu = fileInit.read()

    # Remplacer les caractères spéciaux et les lettres avec accents par leurs équivalents sans accent
        contenu_interE=''.join(['e' if (c=="è" or c=="é" or c=="ê") else c.lower() for c in contenu])
        contenu_interA = ''.join(['a' if (c == "à" or c=="â") else c.lower() for c in contenu_interE])
        contenu_interC = ''.join(['c' if (c == "ç") else c.lower() for c in contenu_interA])
        contenu_interU = ''.join(['u' if (c == "ù") else c.lower() for c in contenu_interC])
        contenu_interpoint = ''.join(['' if (c == "," or c=="." or c=="!" or c=="?" or c=="-" or c=='"' or c==":" or c=="'") else c.lower() for c in contenu_interU])
        contenu_modifie = ''.join([' ' if not (65 <= ord(c) <= 90 or 97 <= ord(c) <= 122 or 48 <=ord(c)<=57)
                                      and c != '\n'  else c.lower() for c in contenu_interpoint])

    #creation d'un nouveau fichier lower+nomdupresident.txt
    chemin_sortie = "cleaned/Lower"+f
    with open(chemin_sortie, 'w', encoding='utf-8') as fileClean:
        fileClean.write(contenu_modifie)