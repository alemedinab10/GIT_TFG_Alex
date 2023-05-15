import os
import pydicom


class Paciente:

    def __init__(self, paciente):
        self.paciente = paciente
        self.direccionBaseDatos = os.path.join(os.getcwd(), "PacienteEjemplo")
        self.files, self.damagedFiles, self.RTSTRUCT = self.importarDatos()
        


    def importarDatos(self):
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        dir_PET = os.path.join(self.direccionBaseDatos, str(self.paciente), r"PET\\")

        files = {"CT": [], "PET": []}
        damaged_files = {"CT": [], "PET": []}
        boolDamagedFiles = True

        # Leer imágenes CT y PET
        for f in os.listdir(dir_CT):
            try:
                ds = pydicom.dcmread(os.path.join(dir_CT, f))
                if ds.pixel_array is None:
                    raise ValueError("Pixel data is missing or empty.")
                files["CT"].append(ds)
            except:
                damaged_files["CT"].append(f)

        for f in os.listdir(dir_PET):
            try:
                ds = pydicom.dcmread(os.path.join(dir_PET, f))
                if ds.pixel_array is None:
                    raise ValueError("Pixel data is missing or empty.")
                files["PET"].append(ds)
            except:
                damaged_files["PET"].append(f)

        damaged_files["CT"] = sorted(damaged_files["CT"])
        damaged_files["PET"] = sorted(damaged_files["PET"])

        if ((len(damaged_files["PET"]) != 0 ) | (len(damaged_files["CT"]) != 0 )): print("Damaged files")
        else: 
            print ("No damaged files")
            boolDamagedFiles = False
        if (len(damaged_files["CT"]) != 0 ): print(f"\nCT Error: \n{damaged_files['CT']}")
        if (len(damaged_files["PET"]) != 0 ): print(f"\nPET Error: \n{damaged_files['PET']}")

        # Leer RTSTRUCT

        dir_RTSTRUCT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"I1")

        # Lee el archivo RTSTUCT que s e supone que es un dcm
        rtstruct = pydicom.dcmread(dir_RTSTRUCT)

        if (boolDamagedFiles):
            return files, damaged_files, rtstruct
        else:
            return files, None, rtstruct



    def metodo2(self):
        # código del método 2
        pass



o = Paciente(29)

print(o.RTSTRUCT)