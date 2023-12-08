import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import random_split


clases = dict({0: "Gato", 1: "No es gato"})

# directorio de las imagenes
dataset_dir = "Gatos"
# ajustamos las imagenes
transformadas = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

# cargamos el dataset de imagenes
dataset = datasets.ImageFolder(dataset_dir, transform=transformadas)

# dividimos el dataset en conjuntos de entrenamiento, validación y prueba

# tamaño del dataset
len_dataset = len(dataset)
# definimos el tamaño de los datos de entrenamiento (70% del dataset)
len_entre = int(len_dataset * 0.7)
# definimos el tamaño de los datos de prueba
len_prueba = int((len_dataset - len_entre)/2)
# tamaño del set de validación
len_val = len_dataset - len_entre - len_prueba
# creamos los conjuntos de entrenamiento y prueba
entrenamiento, validacion, prueba = random_split(dataset, (len_entre, len_val, len_prueba))

# definimos el dataloader con los datos de entrenamiento
loader_entrenamiento = torch.utils.data.DataLoader(entrenamiento, batch_size=32, shuffle=True)
# definimos el dataloader con los datos de validacion
loader_validacion = torch.utils.data.DataLoader(validacion, batch_size=32, shuffle=True)
# definimos el dataloader con los datos de prueba
loader_prueba = torch.utils.data.DataLoader(prueba, batch_size=32, shuffle=True)


# creamos nuestro modelo personalizado
class CNN(torch.nn.Module):

    # constructor
    def __init__(self):
        super().__init__()
        # parte convolutiva
        self.convolutiva = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.AdaptiveAvgPool2d(output_size=(6, 6))
        )
        # clasificador
        self.clasificador = nn.Sequential(
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.convolutiva(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.clasificador(x)
        return x


def evaluar(modelo, prueba, disp, criterio, mostrar=False):

    # ponemos el modelo en modo de evaluacion
    modelo.eval()
    # inicializamos variables
    correctas = 0
    total = 0
    perdida_val = 0.0
    # obtenemos los datos del data loader
    for imagenes, etiquetas in iter(prueba):
        # copiamos los datos al dispositivo seleccionado (gpu o cpu)
        imagenes = imagenes.to(disp)
        etiquetas = etiquetas.to(disp)

        # obtenemos las salidas del modelo
        salidas = modelo(imagenes)

        # obtenemos las predicciones seleccionando la clase con la
        # mayor probabilidad
        predichos = torch.max(salidas.data, 1)[1]

        # acumulamos el número de datos
        total += len(etiquetas)

        # pérdida
        perdida_val += criterio(salidas, etiquetas).data
        # acumulamos las predicciones correctas
        correctas += (predichos == etiquetas).sum()

    perdida_val = float(perdida_val/float(total))
    perdida_val = round(perdida_val, 4)
    # calculamos la precisión
    precision = 100 * correctas / float(total)
    precision = round(float(precision), 4)

    if mostrar:
        print("Precisión: {}%".format(precision))
    else:
        return precision, perdida_val


def entrenar_modelo(modelo, entrenamiento, validacion, criterio, optim, disp, pre_d=100, epocas=100, mostrar=20):

    # listas donde guardaremos los datos para graficar
    l_perdida = []
    l_precision = []
    l_perd_val = []

    # vamos recorriendo las épocas una a una
    for epoca in range(1, epocas + 1):

        perdida_data = 0.0
        total = 0
        # configuramos el modelo en modo de entrenamiento
        modelo.train()
        # obtenemos las imagenes y las etiquetas
        for imagenes, etiquetas in iter(entrenamiento):
            # copiamos los datos al dispositivo seleccionado
            imagenes = imagenes.to(disp)
            etiquetas = etiquetas.to(disp)

            # establecemos el gradiente en 0
            optim.zero_grad()

            # obtenemos la salida del modelo
            salidas = modelo(imagenes)

            # acumulamos el número de datos
            total += len(etiquetas)

            # calculamos la perdida (valor esperado - valor obtenido)
            perdida = criterio(salidas, etiquetas)
            perdida_data += perdida.data

            # calculamos el gradiente usando la perdida estimada
            perdida.backward()

            # actualizamos los pesos
            optim.step()

        # calculamos la perdida
        perdida_data = float(perdida_data/float(total))
        perdida_data = round(perdida_data, 4)

        # calculamos la precisión
        preci, perd_val = evaluar(modelo, validacion, disp, criterio)

        # guardamos los datos para graficar posteriormente
        l_perdida.append(perdida_data)
        l_precision.append(preci)
        l_perd_val.append(perd_val)

        # si se llega a la perdida aceptada y se pierde precisión,
        # se detiene el entrenamiento
        if pre_d <= preci:
            # imprimimos el progreso
            print("Época: {}\tPérdida ent: {}\tPerdida val: {}\tPrecisión: {}".format(
                epoca, perdida_data, perd_val, preci
            ))
            break

        # mostramos el progreso de acuerdo a 'mostrar'
        if epoca % mostrar == 0:
            # imprimimos el progreso
            print("Época: {}\tPérdida ent: {}\tPerdida val: {}\tPrecisión: {}".format(
                epoca, perdida_data, perd_val, preci
            ))

    # convertimos las listas a arrays de numpy
    l_perdida = np.asarray(l_perdida)
    l_precision = np.asarray(l_precision)
    l_perd_val = np.asarray(l_perd_val)
    l_epoca = np.arange(epoca)

    return l_perdida, l_perd_val, l_precision, l_epoca


# usamos una gpu si es que hay alguna disponible
disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instanciamos nuestra
cnn = CNN().to(disp)
# funcion de perdida
# criterio = nn.NLLLoss()
criterio = nn.CrossEntropyLoss()
# optimizador
optimizador = torch.optim.Adam(cnn.parameters(), lr=0.00001)


perdida, perd_val, precision, epoca = entrenar_modelo(
    cnn, loader_entrenamiento, loader_validacion, criterio,
    optimizador, disp, pre_d=80, mostrar=1
     )

# función de pérdida
plt.plot(epoca, perdida, color='r')
plt.plot(epoca, perd_val, color='b')
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Pérdida")
plt.legend(["Entrenamiento", "Validación"])
plt.show()

# precisión
plt.plot(epoca, precision, color='g')
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.title("Precisión")
plt.show()
