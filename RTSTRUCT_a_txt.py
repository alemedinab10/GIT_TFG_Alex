import os
import shutil

bdd = os.path.join("F:", "TFG_AlejandraMedinaBenito", "CosasUtiles", "BasedeDatos")

for f in os.listdir(bdd):
    dir_RTSTRUCT = os.path.join(bdd, f, r"I1")
    file_name, file_ext = os.path.splitext(dir_RTSTRUCT)
    # Cambiar la extensión a .txt
    new_file_name = "RTSTRUCT" + '.txt'
    # Construir la nueva ruta del archivo con la extensión cambiada
    new_file_path = os.path.join(bdd, f, new_file_name)
    # Renombrar el archivo con la nueva extensión
    shutil.copyfile(dir_RTSTRUCT, new_file_path)

    print(dir_RTSTRUCT)