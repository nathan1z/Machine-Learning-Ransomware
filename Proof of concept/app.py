from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import pefile
from sklearn.impute import SimpleImputer
import time

app = Flask(__name__)

# Cargar el modelo y los transformadores
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

@app.route('/')
def index():
    return render_template('index.html')

 #Ruta para manejar la sumisión de comentarios
@app.route('/submit_comment', methods=['POST'])
def submit_comment():
    name = request.form['name']
    comment = request.form['comment']
    # Aquí puedes manejar el comentario, como guardarlo en una base de datos
    return redirect(url_for('upload_file'))

# Ruta para manejar los mensajes de contacto
@app.route('/contact', methods=['POST'])
def contact():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    # Aquí puedes manejar el mensaje de contacto, como enviarlo por correo electrónico
    return redirect(url_for('upload_file'))

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()  # Iniciar el temporizador
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and (file.filename.endswith('.exe') or file.filename.endswith('.dll')):
        file_content = file.read()
        features = extract_features(file_content)
        
        # Asegurarse de que el número de características sea 54
        if len(features) > 54:
            features = features[:54]
        elif len(features) < 54:
            features.extend([0] * (54 - len(features)))
        
        # Convertir None a NaN para usar SimpleImputer
        features = np.array(features, dtype=object)
        features[features == None] = np.nan

        # Rellenar los valores NaN con la media de cada característica
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        features = imputer.fit_transform(features.reshape(1, -1))

        # Aplicar escalador y PCA
        features = scaler.transform(features)
        features = pca.transform(features)
        
        # Realizar la predicción
        prediction = model.predict(features)
        
        result = 'El archivo es ransomware.' if prediction[0] == 1 else 'El archivo es benigno.'
        
        end_time = time.time()  # Finalizar el temporizador
        processing_time = round(end_time - start_time, 2)
        
        # Recopilar detalles adicionales
        details = {
            "file_size": f"{len(file_content)} bytes",
            "file_type": file.mimetype,
            "reason": "Detected as ransomware due to suspicious behavior patterns." if prediction[0] == 1 else "No suspicious behavior detected.",
            "features": {f"Feature {i+1}": val for i, val in enumerate(features.flatten())}
        }
        
        # Debugging prints
        print("Result:", result)
        print("Processing Time:", processing_time)
        print("Details:", details)
        
        return render_template('result.html', result=result, details=details, processing_time=processing_time)
    else:
        return redirect(request.url)

    start_time = time.time()  # Iniciar el temporizador
    
    if 'file' not in request.files:
        print("No file part in the request")  # Debugging print
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        print("No selected file")  # Debugging print
        return redirect(request.url)
    
    if file and (file.filename.endswith('.exe') or file.filename.endswith('.dll')):
        file_content = file.read()
        features = extract_features(file_content)
        
        # Asegurarse de que el número de características sea 54
        if len(features) > 54:
            features = features[:54]
        elif len(features) < 54:
            features.extend([0] * (54 - len(features)))
        
        # Convertir None a NaN para usar SimpleImputer
        features = np.array(features, dtype=object)
        features[features == None] = np.nan

        # Rellenar los valores NaN con la media de cada característica
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        features = imputer.fit_transform(features.reshape(1, -1))

        # Aplicar escalador y PCA
        features = scaler.transform(features)
        features = pca.transform(features)
        
        # Realizar la predicción
        prediction = model.predict(features)
        
        result = 'El archivo es ransomware.' if prediction[0] == 1 else 'El archivo es benigno.'
        
        end_time = time.time()  # Finalizar el temporizador
        processing_time = round(end_time - start_time, 2)
        
        # Recopilar detalles adicionales
        details = {
            "file_size": f"{len(file_content)} bytes",
            "file_type": file.mimetype,
            "reason": "Detected as ransomware due to suspicious behavior patterns." if prediction[0] == 1 else "No suspicious behavior detected."
        }
        
        # Debugging prints
        print("Result:", result)
        print("Processing Time:", processing_time)
        print("Details:", details)
        
        return render_template('result.html', result=result, details=details, processing_time=processing_time)
    else:
        print("File type not supported")  # Debugging print
        return redirect(request.url)



def extract_features(file_content):
    try:
        # Cargar el archivo PE usando pefile
        pe = pefile.PE(data=file_content)

        # Características de las secciones
        section_entropies = [section.get_entropy() for section in pe.sections]
        section_raw_sizes = [section.SizeOfRawData for section in pe.sections]
        section_virtual_sizes = [section.Misc_VirtualSize for section in pe.sections]
        resource_sizes = [
            entry.data.struct.Size for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(entry, 'data')
        ]

        # Extraer características
        features = [
            0,  # Placeholder para 'Name'
            0,  # Placeholder para 'md5'
            pe.FILE_HEADER.Machine,
            pe.FILE_HEADER.SizeOfOptionalHeader,
            pe.FILE_HEADER.Characteristics,
            pe.OPTIONAL_HEADER.MajorLinkerVersion,
            pe.OPTIONAL_HEADER.MinorLinkerVersion,
            pe.OPTIONAL_HEADER.SizeOfCode,
            pe.OPTIONAL_HEADER.SizeOfInitializedData,
            pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            pe.OPTIONAL_HEADER.BaseOfCode,
            pe.OPTIONAL_HEADER.BaseOfData if hasattr(pe.OPTIONAL_HEADER, 'BaseOfData') else 0,
            pe.OPTIONAL_HEADER.ImageBase,
            pe.OPTIONAL_HEADER.SectionAlignment,
            pe.OPTIONAL_HEADER.FileAlignment,
            pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            pe.OPTIONAL_HEADER.MajorImageVersion,
            pe.OPTIONAL_HEADER.MinorImageVersion,
            pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            pe.OPTIONAL_HEADER.SizeOfImage,
            pe.OPTIONAL_HEADER.SizeOfHeaders,
            pe.OPTIONAL_HEADER.CheckSum,
            pe.OPTIONAL_HEADER.Subsystem,
            pe.OPTIONAL_HEADER.DllCharacteristics,
            pe.OPTIONAL_HEADER.SizeOfStackReserve,
            pe.OPTIONAL_HEADER.SizeOfStackCommit,
            pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            pe.OPTIONAL_HEADER.LoaderFlags,
            pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
            len(pe.sections),  # SectionsNb
            sum(section_entropies) / len(section_entropies) if section_entropies else 0,  # SectionsMeanEntropy
            min(section_entropies) if section_entropies else 0,  # SectionsMinEntropy
            max(section_entropies) if section_entropies else 0,  # SectionsMaxEntropy
            sum(section_raw_sizes) / len(section_raw_sizes) if section_raw_sizes else 0,  # SectionsMeanRawsize
            min(section_raw_sizes) if section_raw_sizes else 0,  # SectionsMinRawsize
            max(section_raw_sizes) if section_raw_sizes else 0,  # SectionMaxRawsize
            sum(section_virtual_sizes) / len(section_virtual_sizes) if section_virtual_sizes else 0,  # SectionsMeanVirtualsize
            min(section_virtual_sizes) if section_virtual_sizes else 0,  # SectionsMinVirtualsize
            max(section_virtual_sizes) if section_virtual_sizes else 0,  # SectionMaxVirtualsize
            len(pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,  # ImportsNbDLL
            sum([len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT]) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,  # ImportsNb
            sum([1 for entry in pe.DIRECTORY_ENTRY_IMPORT if hasattr(entry, 'ordinal')]) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0,  # ImportsNbOrdinal
            len(pe.DIRECTORY_ENTRY_EXPORT.symbols) if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT') else 0,  # ExportNb
            len(pe.DIRECTORY_ENTRY_RESOURCE.entries) if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE') else 0,  # ResourcesNb
            sum([entry.get_entropy() for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(entry, 'data')]) / len(resource_sizes) if resource_sizes else 0,  # ResourcesMeanEntropy
            min([entry.get_entropy() for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(entry, 'data')]) if resource_sizes else 0,  # ResourcesMinEntropy
            max([entry.get_entropy() for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries if hasattr(entry, 'data')]) if resource_sizes else 0,  # ResourcesMaxEntropy
            sum(resource_sizes) / len(resource_sizes) if resource_sizes else 0,  # ResourcesMeanSize
            min(resource_sizes) if resource_sizes else 0,  # ResourcesMinSize
            max(resource_sizes) if resource_sizes else 0,  # ResourcesMaxSize
            pe.OPTIONAL_HEADER.DATA_DIRECTORY[10].Size if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 10 else 0,  # LoadConfigurationSize
            len(pe.FileInfo) if hasattr(pe, 'FileInfo') else 0,  # VersionInformationSize
        ]

    except pefile.PEFormatError:
        features = [0, 0] + [0] * 52  # Placeholder para 'Name' y 'md5', 0 para otras características en caso de error

    return features

if __name__ == '__main__':
    app.run(debug=True)
