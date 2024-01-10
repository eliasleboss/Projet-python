#************************************************************************
# Auteur:  Elias HAFSIA

#  Fichier Main comportant l'ouverture des fichiers
# le prétaitemant des fichiers
# la création des nouveaux fichiers cleaned
# l'appel du Menu Principal
#*************************************************************************

from functions import *
import os

directory = "./speeches"
directory2="./cleaned"

#liste des fichier du repertoire initial
files_names = list_of_files(directory, "txt")

# les noms de fichiers
listName=extractName(files_names)

#mettre les prénoms
listSurname=putSurname(listName)

for i in range (len(files_names)):
      # concaténation du nom de repertoire et nom de fichier
      x = os.path.join(directory,files_names[i])
      # néttoyer les fichier et les mettre dans le repertoire cleaned
      # recupération du nom de fichier cleaned

      f = x.split("/")[-1]
      f = f.split("_")[-1]
      with open(x, 'r', encoding='utf-8') as fileInit:
          contenu = fileInit.read()
          contenu_modifie=cleanText(contenu)
          # prétraitement du contenu du fu fichier
          chemin_sortie = "cleaned/Lower" + f
          with open(chemin_sortie, 'w', encoding='utf-8') as fileClean:
              fileClean.write(contenu_modifie)




if __name__ == "__main__":
    # l'appel du menu principal
    menu_principal()



