import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

A = pd.read_csv('Hg.xlsx - Hoja1.csv')

Data = [A['Voltage U1'], A['Corriente IA'], A['Corriente IA.1'], A['Corriente IA.2'], A['Corriente IA.3'], A['Corriente IA.4'], A['Corriente IA.5'], A['Corriente IA.6'], A['Corriente IA.7'], A['Corriente IA.8'], A['Corriente IA.9'], A['Corriente IA.10'], A['Corriente IA.11']]

def MaximosHg(n, Data, numberData):
    Imax = []
    V = []
    IA = np.zeros((8, n))
    for j in range(8):
        for i in range(n):
            IA[j][i] = float(Data[numberData][i + 800 + 200*j])
        Imax.append(max(IA[j]))
        V.append(float(Data[0][list(Data[numberData]).index(str(max(IA[j])))]))
    return V, Imax

I = np.zeros((len(Data) - 1, 8))
V = np.zeros((len(Data) - 1, 8))

for i in range(len(Data) - 1):
    vmax, imax = MaximosHg(200, Data, i + 1)
    I[i] = imax
    V[i] = vmax

for j in range(len(Data)):
    for i in range(3):
        Data[j].pop(i)

Letras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']


Vdif = np.zeros((len(Data) - 1, 7))

for i in range(len(Data) - 1):
    for j in range(len(V[1]) - 1):
        Vdif[i][j] = np.around(np.subtract(V[i][j + 1], V[i][j]), 3)


print('Las corrientes maximas son (fila: Maximos de grafica #, columna: Maximo #...)')
print(I)
print('')
print('Los voltajes correspondientes son (fila: Voltajes de grafica #, columna: Voltaje #...)')
print(V)
print('')
print('Las diferencias calculadas entra voltajes sucesivos son (fila: Diferencias de voltaje de grafica #...)')
print(Vdif)
print('')
print('La media de las diferencias entre voltajes es:', np.mean(Vdif))
print('La varianza de las diferencias entre voltajes es:', np.std(Vdif))


for i in range(len(Data) - 1):
    plt.figure(figsize = (12, 7))
    plt.plot(Data[0], Data[i + 1], 'b-')
    plt.plot(V[i], I[i], 'ro')
    plt.grid()
    plt.xlabel('Voltaje (V)', fontsize = 15)
    plt.ylabel('Corriente (nA)', fontsize = 15)
    plt.legend(['Datos obtenidos', 'Maximos'], fontsize = 15)
    plt.title(f'Datos para Hg{Letras[i]}', fontsize = 15)
    plt.tick_params(labelsize = 12.5)

plt.figure(figsize = (12, 7))
for i in range(len(Data) - 1):
    plt.plot(Data[0], Data[i + 1], '-', lw = 0.9)
plt.grid()
plt.xlabel('Voltaje (V)', fontsize = 15)
plt.ylabel('Corriente (nA)', fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.legend(['Hg{}'.format(i) for i in Letras], fontsize = 15)
plt.title(f'Datos para Hg', fontsize = 15)

plt.show()

