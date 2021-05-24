import numpy as np
import matplotlib.pyplot as plt

Q = np.loadtxt('Ne1.txt', skiprows = 3)
A = np.loadtxt('Ne1A.txt', skiprows = 3)
B = np.loadtxt('Ne1B.txt', skiprows = 3)
C = np.loadtxt('Ne1C.txt', skiprows = 3)
D = np.loadtxt('Ne1D.txt', skiprows = 3)
E = np.loadtxt('Ne1E.txt', skiprows = 3)
F = np.loadtxt('Ne1F.txt', skiprows = 3)
G = np.loadtxt('Ne1G.txt', skiprows = 3)
H = np.loadtxt('Ne1H.txt', skiprows = 3)
I = np.loadtxt('Ne1I.txt', skiprows = 3)
J = np.loadtxt('Ne1J.txt', skiprows = 3)
K = np.loadtxt('Ne1K.txt', skiprows = 3)

Voltage = [[], [], [], [], [], [], [], [], [], [], [], []]
Intensity = [[], [], [], [], [], [], [], [], [], [], [], []]

for i in range(len(A)):
    Voltage[0].append(Q[i][0])
    Voltage[1].append(A[i][0])
    Voltage[2].append(B[i][0])
    Voltage[3].append(C[i][0])
    Voltage[5].append(E[i][0])
    Voltage[6].append(F[i][0])
    Voltage[7].append(G[i][0])
    Voltage[8].append(H[i][0])
    Voltage[9].append(I[i][0])
    Voltage[10].append(J[i][0])
    Voltage[11].append(K[i][0])


    Intensity[0].append(Q[i][1])
    Intensity[1].append(A[i][1])
    Intensity[2].append(B[i][1])
    Intensity[3].append(C[i][1])
    Intensity[5].append(E[i][1])
    Intensity[6].append(F[i][1])
    Intensity[7].append(G[i][1])
    Intensity[8].append(H[i][1])
    Intensity[9].append(I[i][1])
    Intensity[10].append(J[i][1])
    Intensity[11].append(K[i][1])

for i in range(len(D)):
    Voltage[4].append(D[i][0])
    Intensity[4].append(D[i][1])

def MaximosNe(n, V, I):
    Imax = []
    Vmax = []
    IA = np.zeros((3, n))
    for j in range(3):
        for i in range(n):
            IA[j][i] = I[i + 600 + 625*(j + 1)]
        Imax.append(max(IA[j]))
        Vmax.append(V[I.index(max(IA[j]))])
    return Vmax, Imax

Val_Vmax = np.zeros((len(Voltage), 3))
Val_Imax = np.zeros((len(Voltage), 3))


for i in range(len(Intensity)):
    vmax, imax = MaximosNe(500, Voltage[i], Intensity[i])
    Val_Vmax[i] = vmax
    Val_Imax[i] = imax

Vdif = np.zeros((len(Voltage), 2))

for i in range(len(Voltage)):
    for j in range(len(Val_Vmax[1]) - 1):
        Vdif[i][j] = np.around(np.subtract(Val_Vmax[i][j + 1], Val_Vmax[i][j]), 3)

print('Las corrientes maximas son (fila: Maximos de grafica #, columna: Maximo #...)')
print(Val_Imax)
print('')
print('Los voltajes correspondientes son (fila: Voltajes de grafica #, columna: Voltaje #...)')
print(Val_Vmax)
print('')
print('Las diferencias calculadas entra voltajes sucesivos son (fila: Diferencias de voltaje de grafica #...)')
print(Vdif)
print('')
print('La media de las diferencias entre voltajes es:', np.mean(Vdif))
print('La varianza de las diferencias entre voltajes es:', np.std(Vdif))
      
Letras = ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

for i in range(len(Intensity)):
    plt.figure(figsize = (12, 7))
    plt.plot(Voltage[i], Intensity[i], 'b-')
    plt.plot(Val_Vmax[i], Val_Imax[i], 'ro')
    plt.grid()
    plt.xlabel('Voltaje (V)', fontsize = 15)
    plt.ylabel('Corriente (nA)', fontsize = 15)
    plt.legend(['Datos obtenidos', 'Maximos'], fontsize = 15)
    plt.title(f'Datos para Ne1{Letras[i]}', fontsize = 15)
    plt.tick_params(labelsize = 12.5)

plt.figure(figsize = (12, 7))
for i in range(len(Intensity)):
    plt.plot(Voltage[i], Intensity[i], '-', lw = 0.9)
plt.grid()
plt.xlabel('Voltaje (V)', fontsize = 15)
plt.ylabel('Corriente (nA)', fontsize = 15)
plt.legend(['Ne1{}'.format(i) for i in Letras], fontsize = 15)
plt.title('Datos para Ne', fontsize = 15)
plt.tick_params(labelsize = 12.5)

plt.show()