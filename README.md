Une fois la lib ajoutée à votre projet il faut ajouter dans votre répertoire de travail un fichier "Path.ini" contenant le chemin relatif de la lib sous cette forme :

Path=<votre_chemin>

Pour le moment seules deux fonctions sont disponibles :

- learn() : Apprentissage, nécessite la présence d'un fichier "Corpus.txt" dans le répertoire de la lib contenant une grande quantité de texte
- prompt() : Pour le moment cette fonction effectue un test d'encodage du prompt en identifiants de tokens puis le décodage
