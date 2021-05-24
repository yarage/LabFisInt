import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('Libro1.txt', skiprows = 1)

AA = np.zeros((4, len(A)))
for i in range(len(A)):
    for j in range(4):
        AA[j][i] = A[i][j]

def RegresionPolinomial(puntosx, puntosy, orden, M):
    valor_x = np.linspace(np.amin(puntosx), np.amax(puntosx), M)
    graf = []
    n = len(puntosx)
    matriz_regresion = np.zeros((orden + 1, orden + 1))
    vector_regresion = np.zeros((orden + 1, 1))
    for i in range (0, orden + 1):
        for j in range (0, orden + 1):
            if i + j == 0:
                matriz_regresion[i][j] = n
            matriz_regresion[i][j] = sum(np.power(puntosx, i + j))
        vector_regresion[i][0] = sum(np.multiply(np.power(puntosx, i), np.power(puntosy, 1)))
    x_RP = np.linalg.solve(matriz_regresion, vector_regresion)
    for x in valor_x:
        y = 0
        for i in range (0, orden + 1):
            y = y + x_RP[i]*x**i
        graf.append(y)
    return x_RP, valor_x, graf

def Termopar(x):
    return -0.0005*x**3 + 0.0068*x**2 + 18.366*x + 1.968

orden = 3

puntosx1 = AA[0]
puntosx2 = AA[1]
puntosx3 = AA[2]
puntosx4 = AA[3]
puntosx5 = np.linspace(0, 26.831, 100)
puntosy = np.linspace(0, 490, 50)


coef1, vx1, graf1 = RegresionPolinomial(puntosx1, puntosy, orden, 1000)
coef2, vx2, graf2 = RegresionPolinomial(puntosx2, puntosy, orden, 1000)
coef3, vx3, graf3 = RegresionPolinomial(puntosx3, puntosy, orden, 1000)
coef4, vx4, graf4 = RegresionPolinomial(puntosx4, puntosy, orden, 1000)

print('Los coeficientes de la regresion 1 son:')
for i in range (0, orden + 1):
    print(f'a{i} = {np.around(coef1[i], 4)}')
print('')
print('Los coeficientes de la regresion 1 son:')
for i in range (0, orden + 1):
    print(f'a{i} = {np.around(coef2[i], 4)}')
print('')
print('Los coeficientes de la regresion 1 son:')
for i in range (0, orden + 1):
    print(f'a{i} = {np.around(coef3[i], 4)}')
print('')
print('Los coeficientes de la regresion 1 son:')
for i in range (0, orden + 1):
    print(f'a{i} = {np.around(coef4[i], 4)}')

    
plt.figure(figsize=(8,8))
plt.plot(vx1, graf1)
plt.plot(vx2, graf2)
plt.plot(vx3, graf3)
plt.plot(vx4, graf4)
plt.plot(puntosx5, Termopar(puntosx5))
plt.legend(['Tipo E', 'Tipo K', 'Tipo R', 'Tipo S', 'Tipo J'], fontsize=15)
plt.xlabel(r'$V$ (mV)', fontsize = 15)
plt.ylabel(r'$T$ (°C)', fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()


plt.show()