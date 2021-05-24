import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import meshgrid
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D


def Termopar(x):
    return -0.0005*x**3 + 0.0068*x**2 + 18.366*x + 1.968

ValoresT1 = np.loadtxt('Conductividad_termica_T1.txt', skiprows = 12, usecols = (1, 2, 3, 6, 7))
ValoresT2 = np.loadtxt('Conductividad_termica_T2.txt', skiprows = 1, usecols = (0, 1, 2))
ValoresT3 = np.loadtxt('Conductividad_termica_T3_15.txt', skiprows = 1, usecols = (1, 2, 3))
ValoresT4_1 = np.loadtxt('Conductividad_termica_T3_30.txt', skiprows = 6, usecols = (1, 2, 3), max_rows = 141)
ValoresT4_2 = np.loadtxt('Conductividad_termica_T3_30.txt', skiprows = 6, usecols = (5, 6, 7))

D1, T1 = np.zeros((5, len(ValoresT1))), np.zeros((4, len(ValoresT1)))
D2, T2 = np.zeros((3, len(ValoresT2))), np.zeros((2, len(ValoresT2)))
D3, T3 = np.zeros((3, len(ValoresT3))), np.zeros((2, len(ValoresT3)))
D4_1, T4_1 = np.zeros((3, len(ValoresT4_1))), np.zeros((2, len(ValoresT4_1)))
D4_2, T4_2 = np.zeros((3, len(ValoresT4_2))), np.zeros((2, len(ValoresT4_2)))

for i in range(len(ValoresT1)):
    for j in range(5):
        D1[j][i] = ValoresT1[i][j]
    for k in range(4):
        T1[k][i] = Termopar(D1[k + 1][i])

for i in range(len(ValoresT2)):
    for j in range(3):
        D2[j][i] = ValoresT2[i][j]
    for k in range(2):
        T2[k][i] = Termopar(D2[k + 1][i])

for i in range(len(ValoresT3)):
    for j in range(3):
        D3[j][i] = ValoresT3[i][j]
    for k in range(2):
        T3[k][i] = Termopar(D3[k + 1][i])

for i in range(len(ValoresT4_1)):
    for j in range(3):
        D4_1[j][i] = ValoresT4_1[i][j]
    for k in range(2):
        T4_1[k][i] = Termopar(D4_1[k + 1][i])

for i in range(len(ValoresT4_2)):
    for j in range(3):
        D4_2[j][i] = ValoresT4_2[i][j]
    for k in range(2):
        T4_2[k][i] = Termopar(D4_2[k + 1][i])

def MaximosT(D, T):
    PosT = find_peaks(T, distance = 10)[0]
    return D[PosT], T[PosT]

tiempo1_1, temp1_1 = MaximosT(D1[0], T1[0])
tiempo1_2, temp1_2 = MaximosT(D1[0], T1[1])
tiempo1_3, temp1_3 = MaximosT(D1[0], T1[2])
tiempo1_4, temp1_4 = MaximosT(D1[0], T1[3])

tiempo2_1, temp2_1 = MaximosT(D2[0], T2[0])
tiempo2_2, temp2_2 = MaximosT(D2[0], T2[1])

tiempo3_1, temp3_1 = MaximosT(D3[0], T3[0])
tiempo3_2, temp3_2 = MaximosT(D3[0], T3[1])

tiempo4_1_1, temp4_1_1 = MaximosT(D4_1[0], T4_1[0])
tiempo4_1_2, temp4_1_2 = MaximosT(D4_1[0], T4_1[1])

tiempo4_2_1, temp4_2_1 = MaximosT(D4_2[0], T4_2[0])
tiempo4_2_2, temp4_2_2 = MaximosT(D4_2[0], T4_2[1])

def RestaMax(p1, p2):
    resta = abs(np.subtract(p1, p2))
    return resta, np.mean(resta)

R1, mediaR1 = RestaMax(np.insert(tiempo1_1, 0, 0), tiempo1_2)
R2, mediaR2 = RestaMax(tiempo1_3, tiempo1_4)
R3, mediaR3 = RestaMax(tiempo2_1, tiempo2_2)
R4, mediaR4 = RestaMax(tiempo3_1, tiempo3_2)
R5, mediaR5 = RestaMax(tiempo4_1_1, tiempo4_1_2)
R6, mediaR6 = RestaMax(tiempo4_2_1, tiempo4_2_2)

def IncertidumbreSuma(Ival1, Ival2):
    return np.around(np.sqrt(Ival1**2 + Ival2**2), 2)

def IncertidumbreMul(val1, val2, Ival1, Ival2, exp1, exp2):
    return np.around(val1*val2*np.sqrt((exp1*Ival1/val1)**2 + (exp2*Ival2/val2)**2), 2)

def IncertidumbreDiv(val1, val2, Ival1, Ival2, exp1, exp2):
    return np.around(val1/val2*np.sqrt((exp1*Ival1/val1)**2 + (exp2*Ival2/val2)**2), 6)

d_termocupla = 9
I_d_termocupla = 0.05

I_tiempo = 0.01

print('Los tiempos que corresponden a los picos de T para la primera medida de los datos 1 son:')
print(np.insert(tiempo1_1, 0, 0)*60)
print(tiempo1_2*60)
print('Los tiempos que corresponden a los picos de T para la segunda medida de los datos 1 son:')
print(tiempo1_3*60)
print(tiempo1_4*60)
print('Los tiempos que corresponden a los picos de T para la medida de los datos 2 son:')
print(tiempo2_1*60)
print(tiempo2_2*60)
print('Los tiempos que corresponden a los picos de T para la medida de los datos 3 son:')
print(tiempo3_1*60)
print(tiempo3_2*60)
print('Los tiempos que corresponden a los picos de T para la medida de los datos 5 son:')
print(tiempo4_1_1*60)
print(tiempo4_1_2*60)
print('Los tiempos que corresponden a los picos de T para la medida de los datos 6 son:')
print(tiempo4_2_1*60)
print(tiempo4_2_2*60)
print('Con incertidumbre de +/-',I_tiempo)
print('')
print('El desfasaje de los picos de T para la primera medida de los datos 1 son:')
print(R1*60)
print('El desfasaje de los picos de T para la segunda medida de los datos 1 son:')
print(R2*60)
print('El desfasaje de los picos de T para la medida de los datos 2 son:')
print(R3*60)
print('El desfasaje de los picos de T para la medida de los datos 3 son:')
print(R4*60)
print('El desfasaje de los picos de T para la primera medida de los datos 4 son:')
print(R5*60)
print('El desfasaje de los picos de T para la segunda medida de los datos 4 son:')
print(R6*60)
print('Con incertidumbre de +/-',IncertidumbreSuma(I_tiempo, I_tiempo))
print('')

print('Las velocidades de fase para la primera medida de los datos 1 son:')
print(d_termocupla/(R1*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R1*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV1 = d_termocupla/(mediaR1*60)
I_promV1 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R1*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R1)
print(promV1,'+/-',I_promV1)

print('Las velocidades de fase para la segunda medida de los datos 1 son:')
print(d_termocupla/(R2*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R2*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV2 = d_termocupla/(mediaR2*60)
I_promV2 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R2*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R2)
print(promV2,'+/-',I_promV2)

print('Las velocidades de fase para la medida de los datos 2 son:')
print(d_termocupla/(R3*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R3*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV3 = d_termocupla/(mediaR3*60)
I_promV3 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R3*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R3)
print(promV3,'+/-',I_promV3)

print('Las velocidades de fase para la medida de los datos 3 son:')
print(d_termocupla/(R4*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R4*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV4 = d_termocupla/(mediaR4*60)
I_promV4 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R4*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R4)
print(promV4,'+/-',I_promV4)

print('Las velocidades de fase para la primera medida de los datos 4 son:')
print(d_termocupla/(R5*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R5*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV5 = d_termocupla/(mediaR5*60)
I_promV5 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R5*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R5)
print(promV5,'+/-',I_promV5)

print('Las velocidades de fase para la segunda medida de los datos 4 son:')
print(d_termocupla/(R6*60))
print('Con incertidumbres')
print(IncertidumbreDiv(d_termocupla, R6*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1))
print('y promedio')
promV6 = d_termocupla/(mediaR6*60)
I_promV6 = np.sqrt(sum(np.power(IncertidumbreDiv(d_termocupla, R6*60, I_d_termocupla, IncertidumbreSuma(I_tiempo, I_tiempo), 1, 1), 2)))/len(R6)
print(promV6,'+/-',I_promV6)
print('')

promV = np.mean(np.array([promV1, promV2, promV3, promV4, promV5, promV6]))
A = np.array([I_promV1, I_promV2, I_promV3, I_promV4, I_promV5, I_promV6])
I_promV = np.sqrt(sum(np.power(A, 2)))/len(A)

print('La velocidad promedio da fase de todos los datos es:')
print(promV,'+/-',I_promV)

den = 7.8
cal = 0.12
t = 5*60

D = promV**2*t/(4*np.pi)
I_D = IncertidumbreMul(promV, t, I_promV, I_tiempo, 2, 1)/(4*np.pi)

K = den*cal*D
I_K = den*cal*I_D

print('')
print('El valor calculado de D es')
print(D,'+/-',I_D)
print('El valor calculado de K es')
print(K,'+/-',I_K)

plt.figure(figsize = (8, 8))
plt.plot(D1[0], T1[0], 'k--')
plt.plot(D1[0], T1[1], 'k-.')
plt.plot(D1[0][0], T1[0][0], 'ko')
plt.plot(tiempo1_1, temp1_1, 'ko')
plt.plot(tiempo1_2, temp1_2, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(D1[0], T1[2], 'k--')
plt.plot(D1[0], T1[3], 'k-.')
plt.plot(tiempo1_3, temp1_3, 'ko')
plt.plot(tiempo1_4, temp1_4, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(D2[0], T2[0], 'k--')
plt.plot(D2[0], T2[1], 'k-.')
plt.plot(tiempo2_1, temp2_1, 'ko')
plt.plot(tiempo2_2, temp2_2, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5) 
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(D3[0], T3[0], 'k--')
plt.plot(D3[0], T3[1], 'k-.')
plt.plot(tiempo3_1, temp3_1, 'ko')
plt.plot(tiempo3_2, temp3_2, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(D4_1[0], T4_1[0], 'k--')
plt.plot(D4_1[0], T4_1[1], 'k-.')
plt.plot(tiempo4_1_1, temp4_1_1, 'ko')
plt.plot(tiempo4_1_2, temp4_1_2, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(D4_2[0], T4_2[0], 'k--')
plt.plot(D4_2[0], T4_2[1], 'k-.')
plt.plot(tiempo4_2_1, temp4_2_1, 'ko')
plt.plot(tiempo4_2_2, temp4_2_2, 'ko')
plt.xlabel(r'$t$ (min)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.legend(['1ra termocupla', '2da termocupla', 'Maximos'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

def Solucion1(x, t):
    return 100*np.exp(-np.sqrt(w/(2*67))*x)*(np.cos(w*t - np.sqrt(w/(2*D))*x)) + 150 - 0.01*x

tau = 5*60

w = 2*np.pi/tau

valorx = np.linspace(0, 250, 1000)

plt.figure(figsize=(8, 8))
plt.plot(valorx/60, Solucion1(valorx, 2), 'k--')
plt.xlabel(r'$x$ (m)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

valorx = np.linspace(0, 100, 1000)

fig1 = plt.figure(figsize = (8, 8))
ax = Axes3D(fig1)
X = valorx
Y = valorx
X, Y = np.meshgrid(X, Y)
Z = 100*np.exp(-np.sqrt(w/(2*67))*X)*(np.cos(w*Y - np.sqrt(w/(2*D))*X)) + 150 - 0.01*X
ax.plot_surface(X, Y, Z)
ax.set_xlabel(r'$x$ (cm)', fontsize = 15)
ax.set_ylabel(r'$t$ (min)', fontsize = 15)
ax.set_zlabel(r'$T$ (°C)', fontsize = 15)
ax.grid()


plt.show()
