import os
import pydicom
import re
import cv2
import numpy as np
import shutil
import SimpleITK as sitk
import nrrd
import pandas as pd
#from radiomics import featureextractor
import time
from tqdm import tqdm


class Paciente:

    def __init__(self, paciente, ROI_con_mayor_suvmax):
        self.paciente = paciente
        self.direccionBaseDatos = os.path.join(os.path.dirname(os.getcwd()), "PacienteEjemplo")
        self.altura = self.extraerAlturaCT()
        self.peso = self.extraerPesoCT()
        self.spacing = self.extraerSpacing()
        self.files, self.damagedFiles, self.RTSTRUCT = self.importarDatos()
        self.rois = self.obtener_ROI_RTSTRUCT()
        self.nombres_ROI = self.obtenerNombresROI()
        self.UI_Contornos = self.obtenerUI_Contornos()
        self.ROI_con_mayor_suvmax = ROI_con_mayor_suvmax
        self.dicom_roi_map = self.obtener_mapa_dicom_ROI()
        #self.df_Paciente = self.extract_Pyradiomics_data()
        #self.mascaraGeneral = self.obtenerMascarasPaciente()


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


    def extraerAlturaCT(self):
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        ds = pydicom.dcmread(os.path.join(dir_CT, os.listdir(dir_CT)[0]))
        # Buscar el patrón deseado usando expresiones regulares
        patron = re.compile(r'\(0018, 1130\) Table Height\s+DS:\s+(.*)\n')
        resultados = patron.findall(str(ds))
        return float(resultados[0].replace("'", ""))


    def extraerPesoCT(self):
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        ds = pydicom.dcmread(os.path.join(dir_CT, os.listdir(dir_CT)[0]))
        # Buscar el patrón deseado usando expresiones regulares
        patron = re.compile(r'\(0010, 1030\) Patient\'s Weight\s+DS:\s+(.*)\n')
        resultados = patron.findall(str(ds))
        return float(resultados[0].replace("'", ""))


    def extraerSpacing(self):
        dir_PET = os.path.join(self.direccionBaseDatos, str(self.paciente), r"PET\\")
        ds = pydicom.dcmread(os.path.join(dir_PET, os.listdir(dir_PET)[0]))
        # Buscar el patrón deseado usando expresiones regulares
        patron = re.compile(r'\(0028, 0030\) Pixel Spacing\s+DS:\s+(.*)\n')
        patronZ = re.compile(r'\(0018, 0050\) Slice Thickness\s+DS:\s+(.*)\n')
        coordenadaXY = patron.findall(str(ds))[0].replace("'", "")
        coordenadaZ = patronZ.findall(str(ds))[0].replace("'", "")
        resultado = eval(coordenadaXY)
        resultado.append(float(coordenadaZ))
        return resultado


    def obtener_Coordenadas(self, ct):
        patron = re.compile(r"\((0020, 0032)\)\s+Image Position \(Patient\)\s+DS:\s+(.*)")
        coordenadas = patron.search(str(ct)).group(2)
        coordenadas = coordenadas.strip("[]").split(",")
        coordenadas = [float(coord.strip()) for coord in coordenadas]
        return coordenadas


    def obtener_ROIs_y_posicion_del_Dicom (self, numero_en_archivo_dicom):
        r = []
        pos = []
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente),"CT", f"DICOM_{str(numero_en_archivo_dicom).zfill(3)}.dcm"))
        plano = self.obtener_Coordenadas(dicom)[2]
        for clave_externa in self.UI_Contornos:
            for i, cont in enumerate(self.UI_Contornos[clave_externa], 0):
                if (plano == cont[0][2]):
                    r.append(clave_externa)
                    pos.append(i)
        return r, pos


    def obtenerMascara(self, numero_dicom, label=None, ROI=None):
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente), "CT", f"DICOM_{str(numero_dicom).zfill(3)}.dcm"))
        coordenadas = self.obtener_Coordenadas(dicom)
        x_imagen = coordenadas[0]
        y_imagen = coordenadas[1]
        # Definir las coordenadas de los contornos
        r, pos = self.obtener_ROIs_y_posicion_del_Dicom(numero_dicom)
        
        # Se calcula la mascara completa

        mRois = {}
        for i in range(len(r)):
            mRois[i] = np.zeros((512, 512), dtype=np.int32)
        contornos = []
        for i in range(len(r)):
            contornos.append(self.UI_Contornos[r[i]][pos[i]][:, :-1])
        scaled_contornos = [[] for _ in range(len(contornos))]
        for i, contorno in enumerate(contornos):
            for x, y in contorno:
                x = (x - x_imagen)
                y = (y - y_imagen)
                scaled_contornos[i].append((x, y))
        for i in range(len(r)):
            # Dibujar el contorno en la matriz de ceros
            cv2.drawContours(mRois[i], [np.array(scaled_contornos[i], dtype=np.int32)], 0, r[i], -1)
        mascara = np.zeros((512, 512), dtype=object)  # crear matriz mascara con ceros
        for k in range(len(mRois)):  # recorrer todas las matrices mRois[i]s
                for i in range(512):  # recorrer filas
                    for j in range(512):  # recorrer columnas
                        if (mRois[k][i][j] != 0):
                            if (mascara[i][j] == 0):
                                mascara[i][j] = mRois[k][i][j]
                            elif (mascara[i][j] != 0):
                                mascara[i][j] = [mascara[i][j]]
                                mascara[i][j].append(mRois[k][i][j])
                            else:
                                mascara[i][j].append(mRois[k][i][j])

        if label is None:
            if ROI is None:
                # Sin label ni ROI
                return mascara
            else:
                # Sin label con ROI
                if not (isinstance(ROI, list)):
                    mascaraResultado = np.zeros((512, 512), dtype=int)
                    for i in range(512):
                        for j in range(512):
                            try :
                                len(mascara[i][j])
                                existeElemento = False
                                for elemento in mascara[i][j]:
                                    if (elemento == ROI):
                                        existeElemento = True
                                if existeElemento:
                                    mascaraResultado[i][j] = ROI
                            except :
                                if (mascara[i][j] == ROI):
                                    mascaraResultado[i][j] = ROI
                    return mascaraResultado
                else:
                    mascaraResultado = np.zeros((512, 512), dtype=int)
                    for i in range(512):
                        for j in range(512):
                            try :
                                len(mascara[i][j])
                                existeElemento = []
                                for elemento in ROI:
                                    if (elemento in mascara[i][j]):
                                        existeElemento.append(elemento)
                                if len(existeElemento) > 0:
                                    mascaraResultado[i][j] = existeElemento
                            except :
                                if (mascara[i][j] in ROI):
                                    mascaraResultado[i][j] = mascara[i][j]
                    return mascaraResultado
        else:
            if ROI is None:
                # Con label sin ROI
                mascaraResultado = np.zeros((512, 512), dtype=int)
                for i in range(512):
                    for j in range(512):
                        try :
                            len(mascara[i][j])
                            mascaraResultado[i][j] = label
                        except :
                            if (mascara[i][j] != 0):
                                mascaraResultado[i][j] = label
                return mascaraResultado
            else:
                # Con label con ROI
                if not (isinstance(ROI, list)):
                    mascaraResultado = np.zeros((512, 512), dtype=int)
                    for i in range(512):
                        for j in range(512):
                            try :
                                len(mascara[i][j])
                                existeElemento = False
                                for elemento in mascara[i][j]:
                                    if (elemento in ROI):
                                        existeElemento = True
                                if existeElemento:
                                    mascaraResultado[i][j] = label
                            except :
                                if (mascara[i][j] in ROI):
                                    mascaraResultado[i][j] = label
                    return mascaraResultado
                else:
                    mascaraResultado = np.zeros((512, 512), dtype=int)
                    for i in range(512):
                        for j in range(512):
                            try :
                                len(mascara[i][j])
                                existeElemento = False
                                for elemento in mascara[i][j]:
                                    if (elemento == ROI):
                                        existeElemento = True
                                if existeElemento:
                                    mascaraResultado[i][j] = label
                            except :
                                if (mascara[i][j] in ROI):
                                    mascaraResultado[i][j] = label
                    return mascaraResultado


    def obtenerMascarasPaciente(self):
        mascaras = []
        total = len(self.obtener_mapa_dicom_Paciente())
        progress_bar = tqdm(total=total, desc='Construyendo Mascaras')

        for p in self.obtener_mapa_dicom_Paciente():
            mascaras.append(self._obtenerMascaraOptimizadoParaConstructor(p))
            progress_bar.update(1)
            time.sleep(0.001)  # Simulación de un tiempo de procesamiento

        progress_bar.close()

        for i in range (len(mascaras)):
            mascaras[i] = mascaras[i].astype(int)
            
        return mascaras


    def mascaraToString(self, mascara):
        output = ""
        for i in range(len(mascara)):
            for j in range(len(mascara[0])):
                if (mascara[i][j] == 0):
                    output += ". "
                else:
                    output += str(mascara[i][j]) + " "
            output += "\n"
        return output


    def _obtenerMascaraROIEspecificaOptimizadoNRRD(self, ROI, numero_dicom):
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente),"CT", f"DICOM_{str(numero_dicom).zfill(3)}.dcm"))
        coordenadas = self.obtener_Coordenadas(dicom)
        x_imagen = coordenadas[0]
        y_imagen = coordenadas[1]
        # Definir las coordenadas de los contornos
        r, pos = self.obtener_ROIs_y_posicion_del_Dicom(numero_dicom)
        posicion = None;
        for i, roi in enumerate(r):
            if (ROI == roi):
                posicion = i
        if (posicion == None):
            print (f"ERROR: _obtenerMascaraROIEspecificaOptimizadoNRRD(ROI=>{ROI}) El ROI especificado no se encuentra en el DICOM")
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


    def _obtenerMascaraOptimizadoParaConstructor(self, numero_dicom):
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente),"CT", f"DICOM_{str(numero_dicom).zfill(3)}.dcm"))
        coordenadas = self.obtener_Coordenadas(dicom)
        x_imagen = coordenadas[0]
        y_imagen = coordenadas[1]
        # Definir las coordenadas de los contornos
        r, pos = self.obtener_ROIs_y_posicion_del_Dicom(numero_dicom)
        mRois = {}
        for i in range(len(r)):
            mRois[i] = np.zeros((512, 512), dtype=np.int32)
        contornos = []

        for i in range(len(r)):
            contornos.append(self.UI_Contornos[r[i]][pos[i]][:, :-1])

        scaled_contornos = [[] for _ in range(len(contornos))]
        for i, contorno in enumerate(contornos):
            for x, y in contorno:
                x = (x - x_imagen)
                y = (y - y_imagen)
                scaled_contornos[i].append((x, y))

        for i in range(len(r)):
            # Dibujar el contorno en la matriz de ceros
            cv2.drawContours(mRois[i], [np.array(scaled_contornos[i], dtype=np.int32)], 0, 1, -1)

        mascara = np.zeros((512,512), dtype=object)  # crear matriz mascara con ceros

        for k in range(len(mRois)):  # recorrer todas las matrices mRois[i]s
            for i in range(512):  # recorrer filas
                for j in range(512):  # recorrer columnas
                    if (mRois[k][i][j] != 0):
                        mascara[i][j] = 1

        return mascara


    def obtener_ROI_RTSTRUCT(self):
        r = []
        for roi in self.RTSTRUCT.ROIContourSequence:
            r.append(roi)
        return r


    def obtenerNombresROI(self):
        nR = []
        for roi in self.rois:
            ROI_name = int(self.extraerROIName(roi))
            nR.append(ROI_name)
        return nR


    def _importarContornosRTSTRUCT(self):
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


    def _procesar_RTSTRUCT_como_txt(self):
        # Abrir archivo y leer su contenido
        with open(os.path.join(self.direccionBaseDatos, str(self.paciente), "RTSTRUCT.txt"), encoding='utf-8') as archivo:
            texto = archivo.read()

        # Eliminar caracteres raros
        texto_limpio = re.sub('[^A-Za-z0-9().-_,/\n]+', ' ', texto)
        
        # Sobrescribir el archivo original con el texto limpio
        with open(os.path.join(self.direccionBaseDatos, str(self.paciente), "RTSTRUCT.txt"), 'w', encoding='utf-8') as archivo:
            archivo.write(texto_limpio)
        
        # Leer el archivo limpio
        with open(os.path.join(self.direccionBaseDatos, str(self.paciente), "RTSTRUCT.txt"), 'r') as archivo_limpio:
            # Buscar la línea que contiene 'ManualOverrideMaxThresholdS'
            lineas_restantes = archivo_limpio.readlines()
            indice = lineas_restantes.index(next(linea for linea in lineas_restantes if 'ManualOverrideMaxThresholdS' in linea))
            # Guardar las líneas restantes en un archivo nuevo
            with open('archivo_nuevo.txt', 'w') as archivo_nuevo:
                archivo_nuevo.writelines(lineas_restantes[indice+1:])
        
        # Eliminar las líneas que contienen 'ManualOverrideMaxThresholdS' del archivo nuevo
        with open('archivo_nuevo.txt', "r") as f:
            lineas = f.readlines()
        new_lines = [line for line in lineas if 'ManualOverrideMaxThresholdS' not in line]
        with open('archivo_limpio_sin_lineas.txt', "w") as f:
            f.writelines(new_lines)
        os.remove('archivo_nuevo.txt')

        # Separa por ROIS
        search_string = "0 IS"
        input_file = "archivo_limpio_sin_lineas.txt"
        output_folder = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt")
        os.makedirs(output_folder, exist_ok=True)
        current_roi_position = 0
        current_output_file = os.path.join(output_folder, f"{self.nombres_ROI[current_roi_position]}.txt")
        current_group_text = ""
        
        with open(input_file, "r") as f:
            for i, line in enumerate(f, 1):  # Agrega el argumento "1" al enumerate para empezar en la línea 1
                if (search_string in line):
                    if current_group_text:
                        with open(current_output_file, "w") as f_out:
                            f_out.write(current_group_text)

                        current_group_text = ""
                        current_roi_position += 1
                        current_output_file = os.path.join(output_folder, f"{self.nombres_ROI[current_roi_position]}.txt")
                else:
                    current_group_text += line
        if current_group_text:
            with open(current_output_file, "w") as f_out:
                f_out.write(current_group_text)

        # Guardar el ultimo ROI
        search_string = f"0 IS {str(self.nombres_ROI[len(self.nombres_ROI)-2])}"
        input_file = "archivo_limpio_sin_lineas.txt"
        output_file = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt", "tempLast.txt")
        start_writing = False
        with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
            for line in f_in:
                if search_string in line:
                    start_writing = True
                    continue
                if start_writing:
                    f_out.write(line)
        with open(output_file, "r") as f, open(os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt", f"{str(self.nombres_ROI[len(self.nombres_ROI)-1])}.txt"), "w") as output:
            for line in f:
                if len(line.split("\\")) > 1 and any(char.isdigit() for char in line.split("\\")[1]) and "UI" not in line:
                    # línea contiene números entre \\ y no contiene "UI", agregar al archivo de salida
                    output.write(line)
        os.remove(output_file)

        # Filtrar los saltos de linea no esperados
        input_dir = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt")
        output_dir = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt")
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                in_F = os.path.join(input_dir, filename)
                out_F = os.path.join(output_dir, f"ROI_{filename}")
                with open(in_F, "r") as archivo:
                    lineas = archivo.readlines()
                with open(out_F, "w") as archivo_modificado:
                    for linea in lineas:
                        if re.search(r"0P\s\S*\n", linea):
                            linea = linea.replace("\n", " ")
                        archivo_modificado.write(linea)
                os.remove(in_F)

        # Dejar solo las lineas de los Contours
        input_dir = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt")
        output_dir = os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt")
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, f"f{filename}")
                with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
                    filtered_lines = filter(lambda line: line.find("CLOSED_PLANAR") != -1, f_in)
                    f_out.writelines(filtered_lines)
                with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
                    for line in f_in:
                        words = line.split()
                        try:
                            index_0p = words.index("0P")
                            new_line = " ".join(words[index_0p+2:])
                        except ValueError:
                            new_line = line
                        f_out.write(new_line)
                os.remove(input_file)
        os.remove("archivo_limpio_sin_lineas.txt")


    def obtenerUI_Contornos(self):
        UI_Contornos = self._importarContornosRTSTRUCT()
        self._procesar_RTSTRUCT_como_txt()

        for clave_externa in UI_Contornos:
            with open(os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt", f"fROI_{clave_externa}.txt"), "r") as archivo:
                lineas = archivo.readlines()
            for i, cont in enumerate(UI_Contornos[clave_externa], 0):
                if (len(cont) == 1):
                    try:
                        numeros = re.findall(r'\d+(?:\.\d+)?', lineas[i][:lineas[i].index("0 SQ")])
                    except ValueError:
                        numeros = re.findall(r'\d{1,3}(?:\.\d{1,3})?', lineas[i])
                    except IndexError:
                        pass
                    numeros = np.array(numeros)
                    numeros = np.negative(numeros.astype(np.float64))
                    n_pad = 3 - len(numeros) % 3
                    if n_pad < 3:
                        numeros = np.pad(numeros, (0, n_pad), mode='constant', constant_values=0)
                    numeros = numeros.reshape(-1, 3)
                    UI_Contornos[clave_externa][i] = numeros
        # Borro los archivos de texto sobrantes
        shutil.rmtree(os.path.join(self.direccionBaseDatos, str(self.paciente), "ROI_txt"))
        return UI_Contornos


    def obtener_mapa_dicom_ROI(self):
        # mapa de arcivos dicom por cada roi
        drm = {}
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        for r in self.nombres_ROI:
            drm[str(r)] = []
        for f in os.listdir(dir_CT):
            numeroMascara = f.split("_")[1].split(".")[0]
            # Obtener archivos de cada ROI
            rois_por_archivo = self.obtener_ROIs_y_posicion_del_Dicom(numeroMascara)[0]
            if (len(rois_por_archivo) > 0):
                for r in rois_por_archivo:
                    if (numeroMascara not in drm[str(r)]):
                        drm[str(r)].append(numeroMascara)
        return drm


    def obtener_mapa_dicom_Paciente(self):
        # mapa de arcivos dicom por cada roi
        drm = {}
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        for f in os.listdir(dir_CT):
            numeroMascara = f.split("_")[1].split(".")[0]
            ds = pydicom.dcmread(os.path.join(dir_CT, f))
            drm[str(numeroMascara)] = self.obtener_Coordenadas(ds)[2]

        drm = dict(sorted(drm.items(), key=lambda x: x[1], reverse=True))
        return drm


    def _ct_to_NRRD (self):
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        # Crear la carpeta para los archivos DICOM
        if not os.path.exists(os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}")):
            os.makedirs(os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "CT"))
            os.makedirs(os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD"))
            
        for f in os.listdir(dir_CT):
            numeroDcm = f.split("_")[1].split(".")[0]

            if (numeroDcm in self.dicom_roi_map[self.ROI_con_mayor_suvmax]):
                filename = f"DICOM_{numeroDcm}.dcm"
                source_path = os.path.join(dir_CT, filename)
                dest_path = os.path.join(os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "CT"), filename)
                shutil.copyfile(source_path, dest_path)

        reader = sitk.ImageSeriesReader()
        dicomReader = reader.GetGDCMSeriesFileNames(os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "CT"))
        reader.SetFileNames(dicomReader)
        dicoms = reader.Execute()
        sitk.WriteImage(dicoms, os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "image.nrrd"))


    def _mascara_to_NRRD (self):
        dir_CT = os.path.join(self.direccionBaseDatos, str(self.paciente), r"CT\\")
        mascaras_ROI_mayorSUVMAX = [] 

        for f in os.listdir(dir_CT):
            numeroDcm = f.split("_")[1].split(".")[0]
            if numeroDcm in self.dicom_roi_map[self.ROI_con_mayor_suvmax]:
                mascaras_ROI_mayorSUVMAX.append(self._obtenerMascaraROIEspecificaOptimizadoNRRD(int(self.ROI_con_mayor_suvmax), numeroDcm))

        mascaras_ROI_mayorSUVMAX = np.array([mascaras_ROI_mayorSUVMAX])
        mascaras_ROI_mayorSUVMAX = mascaras_ROI_mayorSUVMAX.reshape((3, 512, 512))

        mascaras_ROI_mayorSUVMAX = mascaras_ROI_mayorSUVMAX.astype(int)
        mask_image = sitk.GetImageFromArray(mascaras_ROI_mayorSUVMAX)

        sitk.WriteImage(mask_image, os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "mask.nrrd"))

        # Lee el archivo NRRD de la imagen para aplicarlo a la mascara
        file_path_image = os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "image.nrrd")
        data_Image, header_Image = nrrd.read(file_path_image)
        # Lee el archivo NRRD y obtén los datos y el encabezado
        file_path = os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "mask.nrrd")
        data_Mask, header_Mask = nrrd.read(file_path)

        # Modifica los valores del encabezado según tus necesidades
        header_Mask['type'] = header_Image['type']
        header_Mask['space directions'] = header_Image['space directions']
        header_Mask['space origin'] = header_Image['space origin']

        # Guarda los datos modificados y el encabezado en un nuevo archivo NRRD
        new_file_path = os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "mask.nrrd")
        nrrd.write(new_file_path, data_Mask, header_Mask)


    """ def extract_Pyradiomics_data (self):
        self._ct_to_NRRD()
        self._mascara_to_NRRD()

        warnings.filterwarnings('ignore')
        data = pd.DataFrame()
        imagePath = os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "image.nrrd")
        maskPath = os.path.join(self.direccionBaseDatos, str(self.paciente), f"CT_ROI_{self.ROI_con_mayor_suvmax}", "NRRD", "mask.nrrd")

        # Instantiate the default extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()

        result_pyradiomics = extractor.execute(imagePath, maskPath)

        data['Paciente'] = [self.paciente]
        data = data.set_index('Paciente')
        data['ROI con mayor SUVmax'] = [self.ROI_con_mayor_suvmax]
        for i, (key, value) in enumerate(result_pyradiomics.items()):
            if i > 10 :
                data[key] = [value]
        return data """