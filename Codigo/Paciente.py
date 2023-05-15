import os
import pydicom
import re
import cv2
import numpy as np
import shutil


class Paciente:

    def __init__(self, paciente):
        self.paciente = paciente
        self.direccionBaseDatos = os.path.join(os.getcwd(), "PacienteEjemplo")
        self.files, self.damagedFiles, self.RTSTRUCT = self.importarDatos()
        self.rois = self.obtener_ROI_RTSTRUCT()
        self.nombres_ROI = self.obtenerNombresROI()
        self.UI_Contornos = self.obtenerUI_Contornos()


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
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente),"CT", f"DICOM_{str(numero_en_archivo_dicom).zfill(3)}.dcm"))
        plano = self.obtener_Coordenadas(dicom)[2]
        for clave_externa in self.UI_Contornos:
            for i, cont in enumerate(self.UI_Contornos[clave_externa], 0):
                if (plano == cont[0][2]):
                    r.append(clave_externa)
                    pos.append(i)
        return r, pos


    def obtenerMascaraROIEspecifica(self, ROI, numero_dicom):
        dicom = pydicom.dcmread(os.path.join(self.direccionBaseDatos, str(self.paciente),"CT", f"DICOM_{str(numero_dicom).zfill(3)}.dcm"))
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
            print (f"ERROR: obtenerMascaraROIEspecifica(ROI=>{ROI}) El ROI especificado no se encuentra en el DICOM")
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



o = Paciente(29)

print(o.UI_Contornos)