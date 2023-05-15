import os
import pydicom
import re
import cv2
import numpy as np


class Paciente:

    def __init__(self, paciente):
        self.paciente = paciente
        self.direccionBaseDatos = os.path.join(os.getcwd(), "PacienteEjemplo")
        self.files, self.damagedFiles, self.RTSTRUCT = self.importarDatos()
        self.rois = self.obtener_ROI_RTSTRUCT()
        self.nombres_ROI = self.obtenerNombresROI()
        self.UI_Contornos = self.importarContornosRTSTRUCT()


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


    def extraerROIName(self, roi):
        # Define el patrón para encontrar las líneas que empiezan por "(0008, 1155)"
        patron = re.compile(r"\((3006, 0084)\)\s+Referenced ROI Number\s+IS:\s+'?(\d+)'?")
        name = patron.search(str(roi)).group(2)
        return name


    def extraerUI(self, ds):
        # Buscar el patrón deseado usando expresiones regulares
        patron = re.compile(r'\(0008, 1155\) Referenced SOP Instance UID\s+UI:\s+(.*)\n')
        resultados = patron.findall(str(ds))
        return resultados


    def obtener_Coordenadas(self, ct):
        patron = re.compile(r"\((0020, 0032)\)\s+Image Position \(Patient\)\s+DS:\s+(.*)")
        coordenadas = patron.search(str(ct)).group(2)
        coordenadas = coordenadas.strip("[]").split(",")
        coordenadas = [float(coord.strip()) for coord in coordenadas]
        return coordenadas


    def obtener_ROIs_y_posicion_del_Dicom (self, numero_en_archivo_dicom):
        r = []
        pos = []
        dicom = pydicom.dcmread(f"../PacienteEjemplo/{self.paciente}/CT/DICOM_{str(numero_en_archivo_dicom).zfill(3)}.dcm")
        plano = self.obtener_Coordenadas(dicom)[2]
        for clave_externa in self.UI_Contornos:
            for i, cont in enumerate(self.UI_Contornos[clave_externa], 0):
                if (plano == cont[0][2]):
                    r.append(clave_externa)
                    pos.append(i)
        return r, pos


    def obtenerMascaraROIEspecifica(self, ROI, numero_dicom):
        dicom = pydicom.dcmread(f"../PacienteEjemplo/{self.paciente}/CT/DICOM_{str(numero_dicom).zfill(3)}.dcm")
        coordenadas = self.obtener_Coordenadas(dicom)
        x_imagen = coordenadas[0]
        y_imagen = coordenadas[1]
        # Definir las coordenadas de los contornos
        r, pos = self.obtener_ROIs_y_posicion_del_Dicom(numero_dicom, self.UI_Contornos)
        posicion = None;
        for i, roi in enumerate(r):
            if (ROI == roi):
                posicion = i
        if (posicion == None):
            print ("El ROI especificado no se encuentra en el DICOM")
            return
        mRois = np.zeros((512, 512), dtype=np.int32)
        contornos = []
        contornos.append(self.UI_Contornos[ROI][pos[posicion]][:, :-1])
        scaled_contornos = []
        for contorno in contornos:
            for x, y in contorno:
                x = (x - x_imagen)
                y = (y - y_imagen)
                scaled_contornos.append((x, y))
        cv2.drawContours(mRois, [np.array(scaled_contornos, dtype=np.int32)], 0, 1, -1)
        return mRois

    def obtener_ROI_RTSTRUCT (self):
        r = []
        for roi in self.RTSTRUCT.ROIContourSequence:
            r.append(roi)
        return r


    def obtenerNombresROI (self):
        nR = []
        for roi in self.rois:
            ROI_name = int(self.extraerROIName(roi))
            nR.append(ROI_name)
        return nR

    def importarContornosRTSTRUCT(self):
        UI_Contornos = {}

        # Recorre la lista de 'rois'
        for roi in self.rois:
            contour_data_patron = re.compile(r'Contour Data\s+DS:\s+(.*)').findall(str(roi))
            ROI_name = int(self.extraerROIName(roi))
            UI_Contornos[ROI_name] = []
            posicion = 0

            for cont in contour_data_patron:
                if cont.split()[0] == "Array":
                    UI_Contornos[ROI_name].append([-1])
                    # Paso a la siguiente posición de UI
                    posicion+=1
                else:
                    # Busca todos los números en la cadena y conviértelos a flotantes
                    cont_values = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', cont)]
                    # Divide la lista en sub-listas de 3 elementos cada una
                    cont_matrix = [cont_values[i:i+3] for i in range(0, len(cont_values), 3)]
                    # Imprime la lista resultante
                    cont_matrix = np.array(cont_matrix)
                    UI_Contornos[int(self.extraerROIName(roi))].append(cont_matrix)
                    posicion+=1

        return UI_Contornos

o = Paciente(29)

print(o.UI_Contornos)