import tkinter as tk  # Tkinter: Für die Erstellung der GUI.
from tkinter import filedialog, messagebox  # Dialoge für Datei-Auswahl und Nachrichtenboxen.
from tkinter import Toplevel  # Toplevel: Für das Erstellen neuer Fenster in der GUI.

from PIL import Image, ImageTk  # PIL: Zum Öffnen, Bearbeiten und Anzeigen von Bildern in der GUI.

import pydicom  # pydicom: Zum Arbeiten mit DICOM-Dateien.
from pydicom.dataset import FileMetaDataset, FileDataset  # DICOM-Datasets zum Erstellen/Bearbeiten von DICOM-Dateien.
from pydicom.uid import generate_uid, SecondaryCaptureImageStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian  # Zum Generieren von UIDs und Festlegen von DICOM-Einstellungen.

import os  # os: Zum Arbeiten mit Dateipfaden und Dateisystemoperationen.

import numpy as np  # NumPy: Für Array-Operationen, speziell für Bild- und Voxel-Daten.

from datetime import datetime  # datetime: Zum Arbeiten mit Datums- und Zeitstempeln, z.B. in DICOM-Dateien.

import matplotlib.pyplot as plt  # Matplotlib: Für das Plotten von 2D- und 3D-Daten.
from mpl_toolkits.mplot3d import Axes3D  # 3D-Plotting-Tool von Matplotlib.

from mayavi import mlab  # Mayavi: Für die Visualisierung von 3D-Daten.

from tkinter import ttk  # Für den Ladebalken
import threading  # Für die parallele Verarbeitung


def loadVOX(fname):
    vol = []
    try:
        f = open(fname, "rb")
        size = os.path.getsize(fname)
        f.seek(size - 512 * 512 * 512 * 2)
        vol = np.reshape(np.frombuffer(f.read(512 * 512 * 512 * 2), dtype='ushort'), (512, 512, 512), order='F').swapaxes(0, 1)
    finally:
        f.close()
    return vol

def create_dicom_from_vox(image_array, output_filename):
    image_array = np.swapaxes(image_array, 0, 2)
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    normalized_data = (image_array - min_val) / (max_val - min_val)
    scaled_data = (normalized_data * 255).astype(np.uint8)
    
    multi_frame_pixel_data = b''.join(frame.tobytes() for frame in scaled_data)

    ds = FileDataset(output_filename, {}, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = "SampleName1: Robin"
    ds.PatientID = "VolumeFileName: CT_20231116_112805.VOX"
    ds.StudyDescription = "SeriesComment: Robin1"
    ds.SeriesDescription = "Robin1"
    ds.PatientComments = "KV:70 UA:114 FOV:60 VoxelSize:120 ScanTime:4min XrayFilterAI 0,5mm"
    ds.Rows = image_array.shape[0]
    ds.Columns = image_array.shape[1]
    ds.NumberOfFrames = len(image_array)
    ds.ImagePositionPatient = r"0\1\0"
    ds.ImageOrientationPatient = r"0\1\0\0\1\0"
    ds.PixelSpacing = r"1\1"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = multi_frame_pixel_data
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    ds.save_as(output_filename)

def show_vox_image_mayavi(vox_path):
    image_array = loadVOX(vox_path)
    
    # Downsampling: Verkleinere die Auflösung um den Faktor 4
    downsample_factor = 2
    image_array = image_array[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Normierung des Volumens für die Anzeige
    image_array = image_array.astype(np.float32) / np.max(image_array)
    
    # Visualisierung des Volumens mit Mayavi
    mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    
    # Erzeuge eine Skalarfeldquelle für das Volumen
    src = mlab.pipeline.scalar_field(image_array)
    
    # Volumenrendering
    vol = mlab.pipeline.volume(src, vmin=0.2, vmax=0.8) # vmin=0.2, vmax=0.8
    src.scene.light_manager.light_mode = 'vtk'
    src.scene.light_manager.intensity = 3.0
    vol._volume_property.scalar_opacity_unit_distance = 0.1 # besser
    vol.update()
    
    mlab.show()


def show_vox_with_mayavi2(vox_data, centroid, normal_vector):
    # Normiere das Volumen für die Anzeige
    vox_data = vox_data.astype(np.float32) / np.max(vox_data)
    
    # Visualisierung des Volumens mit Mayavi
    mlab.figure(size=(800, 800), bgcolor=(0, 0, 0))
    src = mlab.pipeline.scalar_field(vox_data)
    vol = mlab.pipeline.volume(src, vmin=0.2, vmax=0.8)
    vol._volume_property.scalar_opacity_unit_distance = 0.1

    # Berechne Flächennormalenvektoren und den Mittelpunkt der Ebene
    normal_vector = np.array([0, 0, 1])  # Beispiel: Z-Achse als Normalenvektor
    center = np.array(vox_data.shape) / 2

   
    # Zeige den Normalenvektor und den Schwerpunkt an
    mlab.quiver3d(centroid[0], centroid[1], centroid[2], normal_vector[0], normal_vector[1], normal_vector[2], scale_factor=50)
    mlab.points3d(centroid[0], centroid[1], centroid[2], color=(1, 0, 0), scale_factor=10)
    
    
    mlab.show(stop=True)


def show_vox_image(vox_path):    #wird nicht benutzt 
    image_array = loadVOX(vox_path)
    
    # Downsampling: Verkleinere die Auflösung um den Faktor 4
    # nur jedes  4. voxel (wenn downsamplefaktor 4)
    downsample_factor = 1
    image_array = image_array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Erhöhe den Threshold für die Anzeige, um weniger Daten zu visualisieren
    threshold = np.percentile(image_array, 99)  # Nur die hellsten 1% anzeigen
    x, y, z = np.where(image_array > threshold)

    ax.scatter(x, y, z, c=z, cmap='gray', marker='o', s=1)
    ax.set_title("3D Voxel Image (Downsampled)")
    
    plt.show()

def select_vox():
    vox_path = filedialog.askopenfilename(title="Select VOX File", filetypes=[("VOX Files", "*.VOX")])
    vox_label.config(text=vox_path)
    
      # 3D-Bild anzeigen
    if vox_path:
        show_vox_image_mayavi(vox_path)
        
    return vox_path

def select_tiff():
    tiff_path = filedialog.askopenfilename(title="Select TIFF File", filetypes=[("TIFF Files", "*.tif")])
    tiff_label.config(text=tiff_path)
    
    # TIFF-Bild anzeigen
    show_tiff_image(tiff_path)
    
    return tiff_path

def select_output():
    output_path = filedialog.asksaveasfilename(defaultextension=".dcm", filetypes=[("DICOM Files", "*.dcm")])
    output_label.config(text=output_path)
    return output_path

def show_tiff_image(tiff_path):
    tiff_image = Image.open(tiff_path)
    tiff_image.thumbnail((400, 400))  # Größe des Bildes anpassen für die Anzeige

    top = Toplevel()
    top.title("Histogram Image")

    img = ImageTk.PhotoImage(tiff_image)
    panel = tk.Label(top, image=img)
    panel.image = img  # Referenz speichern, um Garbage Collection zu verhindern
    panel.pack()

def create_dicom_from_tiff(tiff_path):
    tiff_image = Image.open(tiff_path)
    
    if tiff_image.mode != 'RGB':
        raise ValueError("Das Bild ist kein RGB-Bild")

    r, g, b = tiff_image.split()
    
    target_size = (512, 512)
    r = r.resize(target_size, Image.LANCZOS)
    g = g.resize(target_size, Image.LANCZOS)
    b = b.resize(target_size, Image.LANCZOS)
    
    r.save('C:/Users/robin/Pictures/image_R_channel.tif')
    g.save('C:/Users/robin/Pictures/image_G_channel.tif')
    b.save('C:/Users/robin/Pictures/image_B_channel.tif')

    grayscale_tiffs = [
        'C:/Users/robin/Pictures/image_R_channel.tif',
        'C:/Users/robin/Pictures/image_G_channel.tif',
        'C:/Users/robin/Pictures/image_B_channel.tif'
    ]

    dicom_files = []

    for tiff_file in grayscale_tiffs:
        tiff_image = Image.open(tiff_file)
        tiff_array = np.array(tiff_image)

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()

        ds = FileDataset(tiff_file, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'OT'
        ds.ContentDate = datetime.now().strftime('%Y%m%d')
        ds.ContentTime = datetime.now().strftime('%H%M%S')
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = tiff_array.shape
        ds.PixelSpacing = [1, 1]
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = tiff_array.tobytes()

        dicom_output_filename = tiff_file.replace('.tif', '.dcm')
        ds.save_as(dicom_output_filename)
        dicom_files.append(dicom_output_filename)

        print(f"Grayscale TIFF converted to DICOM and saved as {dicom_output_filename}")

    return dicom_files

def combine_dicom(dicom_files, output_filename):
    base_dicom = pydicom.dcmread(dicom_files[0])

    if not hasattr(base_dicom, 'SOPClassUID'):
        base_dicom.SOPClassUID = generate_uid()

    base_dicom.SOPInstanceUID = generate_uid()

    all_pixel_data = []
    frame_sequences = []

    for filename in dicom_files:
        dicom_data = pydicom.dcmread(filename)

        if hasattr(dicom_data, 'PixelData'):
            if not hasattr(dicom_data.file_meta, 'TransferSyntaxUID'):
                dicom_data.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

            frames = dicom_data.NumberOfFrames if 'NumberOfFrames' in dicom_data else 1

            for i in range(frames):
                pixel_array = dicom_data.pixel_array
                if frames > 1:
                    all_pixel_data.append(pixel_array[i].tobytes())
                else:
                    all_pixel_data.append(pixel_array.tobytes())

                if 'PerFrameFunctionalGroupsSequence' in dicom_data:
                    frame_sequences.append(dicom_data.PerFrameFunctionalGroupsSequence[i])

    base_dicom.NumberOfFrames = len(all_pixel_data)
    base_dicom.PixelData = b''.join(all_pixel_data)

    if frame_sequences:
        base_dicom.PerFrameFunctionalGroupsSequence = frame_sequences

    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = base_dicom.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = base_dicom.SOPInstanceUID
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(output_filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.update(base_dicom)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(output_filename)
    print(f"Combined DICOM file saved as {output_filename}")

def process_files():
    vox_path = vox_label.cget("text")
    tiff_path = tiff_label.cget("text")
    output_path = output_label.cget("text")
    
    if not vox_path or not tiff_path or not output_path:
        messagebox.showerror("Error", "Please select VOX, TIFF files and specify output path.")
        return
    
    try:
        image_array = loadVOX(vox_path)
        create_dicom_from_vox(image_array, "C:/Users/robin/Pictures/test.dcm")
        dicom_files = create_dicom_from_tiff(tiff_path)
        dicom_files.insert(0, "C:/Users/robin/Pictures/test.dcm")
        combine_dicom(dicom_files, output_path)
        messagebox.showinfo("Success", "DICOM files processed and combined successfully.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def loadVOXFromDICOM(dicom_path):
    # Lade die DICOM-Datei
    ds = pydicom.dcmread(dicom_path)
    # Extrahiere das 3D-VOX-Bild aus den ersten 512 Frames
    vox_data = ds.pixel_array[:512]
    return vox_data

def extractTIFFFromDICOM(dicom_path):
    # Lade die DICOM-Datei
    ds = pydicom.dcmread(dicom_path)
    # Extrahiere die letzten drei Frames für das TIFF-Bild
    r = ds.pixel_array[-3]
    g = ds.pixel_array[-2]
    b = ds.pixel_array[-1]
    # Kombiniere die Kanäle wieder zu einem RGB-Bild
    tiff_image = np.stack((r, g, b), axis=-1)
    return tiff_image

def show_tiff_image(tiff_image):
    # Zeige das TIFF-Bild an
    plt.imshow(tiff_image)
    plt.title("Reconstructed TIFF Image")
    plt.show(block=False)

def calculate_centroid_and_normal(vox_data):
    """
    Berechnet den Schwerpunkt (Centroid) und den Normalenvektor des 3D-Volumens. 
    """
    # Binarisiere das Volumen
    ones = np.where(vox_data > 0)
    ones_array = np.array([ones[0], ones[1], ones[2]]).T
    
    # Berechne den Schwerpunkt (Centroid)
    centroid = np.mean(ones_array, axis=0)
    
    # Berechne die Kovarianzmatrix und Eigenvektoren (Hauptträgheitsachsen)
    S = np.cov((ones_array - centroid).T)
    w, v = np.linalg.eig(S)
    normal_vector = v[:, np.argmin(w)]  # Der Vektor mit dem kleinsten Eigenwert ist der Normalenvektor

    return centroid, normal_vector
def process_combined_dicom(progress_var):
    dicom_path = filedialog.askopenfilename(title="Select Combined DICOM File", filetypes=[("DICOM Files", "*.dcm")])
    
    if not dicom_path:
        messagebox.showerror("Error", "No DICOM file selected.")
        return

    try:
        progress_var.set(0)  # Ladebalken auf 0% setzen
        
        # VOX-Bild anzeigen
        vox_data = loadVOXFromDICOM(dicom_path)
        centroid, normal_vector = calculate_centroid_and_normal(vox_data)
        progress_var.set(50)  # Ladebalken auf 50% nach Berechnung
        
        show_vox_with_mayavi2(vox_data, centroid, normal_vector)
        progress_var.set(80)  # Ladebalken auf 80% nach der Mayavi-Visualisierung

        # TIFF-Bild anzeigen
        tiff_image = extractTIFFFromDICOM(dicom_path)
        show_tiff_image(tiff_image)
        
        progress_var.set(100)  # Ladebalken auf 100% nach der vollständigen Verarbeitung
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

def start_processing_combined_dicom():
    # Starte die Verarbeitung in einem separaten Thread, um die GUI nicht zu blockieren
    threading.Thread(target=process_combined_dicom, args=(progress_var,)).start()

# GUI erstellen
root = tk.Tk()
root.title("DICOM VOX-TIFF Processor")

# Dark Mode Farben
dark_bg = "#2e2e2e"
dark_fg = "#ffffff"
dark_btn_bg = "#555555"
dark_btn_fg = "#ffffff"

root.configure(bg=dark_bg)

# VOX-Datei auswählen
vox_button = tk.Button(root, text="Select VOX File", command=select_vox, bg=dark_btn_bg, fg=dark_btn_fg)
vox_button.pack(pady=10)
vox_label = tk.Label(root, text="No VOX file selected", wraplength=300, bg=dark_bg, fg=dark_fg)
vox_label.pack()

# TIFF-Datei auswählen
tiff_button = tk.Button(root, text="Select TIFF File", command=select_tiff, bg=dark_btn_bg, fg=dark_btn_fg)
tiff_button.pack(pady=10)
tiff_label = tk.Label(root, text="No TIFF file selected", wraplength=300, bg=dark_bg, fg=dark_fg)
tiff_label.pack()

# Output-Pfad auswählen
output_button = tk.Button(root, text="Select Output File", command=select_output, bg=dark_btn_bg, fg=dark_btn_fg)
output_button.pack(pady=10)
output_label = tk.Label(root, text="No output file selected", wraplength=300, bg=dark_bg, fg=dark_fg)
output_label.pack()

# Button zum Starten des Prozesses
process_button = tk.Button(root, text="Combine VOX and TIFF", command=process_files, bg=dark_btn_bg, fg=dark_btn_fg)
process_button.pack(pady=20)

# Ladebalken hinzufügen
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=20, padx=20, fill=tk.X)

# Button zur Auswahl und Verarbeitung der kombinierten DICOM-Datei
next_button = tk.Button(root, text="Select DICOM File", command=start_processing_combined_dicom, bg=dark_btn_fg, fg=dark_btn_bg)
next_button.pack(pady=20)

root.mainloop()