from functions import *
import os

directory = "./speeches"
files_names = list_of_files(directory,"txt")
print_list(files_names)
listName=extractName(files_names)
listSurname=putSurname(listName)
print_list(listSurname)

for i in range (len(files_names)):
      # concaténation du nom de repertoire et nom de fichier
      x = os.path.join(directory,files_names[i])
      cleanText(x)




