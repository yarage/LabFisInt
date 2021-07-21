import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
import scipy.constants as c

def Regresion(x, y, f, beta0):
    model = odr.Model(f)
    data = odr.RealData(x, y)
    myodr = odr.ODR(data, model, beta0)
    out = myodr.run()
    popt = out.beta
    perr = out.sd_beta
    x_fit = np.linspace(min(x), max(x), 5000)
    fit = f(popt, x_fit)
    return popt, perr, x_fit, fit

def MinimosCuadrados(puntosx, puntosy):
    n = len(puntosx)
    graf = []
    valor_x = np.linspace(np.amin(puntosx), np.amax(puntosx), 1000) 
    a1 = (n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy))/(n*sum(np.power(puntosx, 2)) - (sum(puntosx))**2)
    a0 = np.mean(puntosy) - a1*np.mean(puntosx)
    r1 = n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy)
    r2 = np.sqrt(n*sum(np.power(puntosx, 2)) - sum(puntosx)**2)*np.sqrt(n*sum(np.power(puntosy, 2)) - sum(puntosy)**2)
    r = r1/r2
    for x in valor_x:
        y = a0 + a1*x
        graf.append(y)
    return r**2

def FuncionPrueba(B, x):
    return B[0] + B[1]*x + B[2]*x**2

def FuncionPrueba1(B, x):
    return B[0] + B[1]/x**B[2]

def Funcion2(B, x):
    return B[0]/x**B[1]

def Funcion3(B, x):
    return B[0] + B[1]*x

def Funcion4(B, x):
    return B[0] + B[1]*x

def IncertidumbreSuma(Ival):
    y = 0
    for i in range(len(Ival)):
        y += Ival[i]**2
    return np.sqrt(y)

def IncertidumbreMul(val, Ival, exp):
    y1, y2 = 1, 0
    for i in range(len(val)):
        y1 *= val[i]
        y2 += (exp[i]*Ival[i]/val[i])**2
    return y1*np.sqrt(y2)

def IncertidumbreDiv(val, Ival, exp):
    y2 = 0
    for i in range(len(val)):
        y2 += (exp[i]*Ival[i]/val[i])**2
    return val[0]/val[1]*np.sqrt(y2)
    
Data_exp1 = np.loadtxt('Datos.txt', skiprows = 12, max_rows = 4, usecols = (1, 2, 3, 4)).T
Data_exp2_1 = np.loadtxt('Datos.txt', skiprows = 30, max_rows = 10).T
Data_exp2_2 = np.loadtxt('Datos.txt', skiprows = 42, max_rows = 29).T
Data_exp3 = np.loadtxt('Datos.txt', skiprows = 81, max_rows = 10).T
Data_exp4 = np.loadtxt('Datos.txt', skiprows = 100, max_rows = 11).T

Medidas = np.zeros((len(Data_exp1), len(Data_exp1)))
Medias = np.zeros((len(Data_exp1), 2))

for i in range(len(Data_exp1)):
    for j in range(len(Data_exp1)):
        Medidas[i][j] = Data_exp1[i][j]/Data_exp1[i][0]

for i in range(len(Data_exp1)):
    Medias[i][0] = np.mean(Medidas.T[i])
    Medias[i][1] = np.std(Medidas.T[i])

print('EXP 1')
print('Medidas relativas')
print(Medidas)
print('Medias con sus incertidumbres')
print(Medias)

print('')
print('EXP 2')
Data_exp2_prueba = np.loadtxt('Datos.txt', skiprows = 55, max_rows = 16).T
Rad_prom = np.mean(Data_exp2_1[1])
exp2, I_exp2, fitx2, fity2 = Regresion(Data_exp2_2[0]/100, Data_exp2_2[1] - Rad_prom, Funcion2, [1, 2])
exp21, I_exp21, fitx21, fity21 = Regresion(1/((Data_exp2_prueba[0]/100)**2), Data_exp2_prueba[1] - Rad_prom, Funcion3, [1, 1])
r2_2 = MinimosCuadrados(1/((Data_exp2_prueba[0]/100)**2), Data_exp2_prueba[1] - Rad_prom)

print(f'Radiacion promedio del ambiente: {Rad_prom}')
print('Curva para la ley de inversos cuadrados')
print(f'a0 = {exp2[0]} +/- {I_exp2[0]}')
print(f'a1 = {exp2[1]} +/- {I_exp2[1]}')
print(f'r^2 = 0.987')
print('Recta para la ley de inversos cuadrados')
print(f'a0 = {exp21[0]} +/- {I_exp21[0]}')
print(f'a1 = {exp21[1]} +/- {I_exp21[1]}')
print(f'r^2 = {r2_2}')
print('')
print('EXP 3')
I_Res = np.zeros(len(Data_exp3[0]))
I_Res_rel = np.zeros(len(Data_exp3[0]))
for i in range(len(Data_exp3[0])):
    I_Res[i] = IncertidumbreDiv([Data_exp3[0][i], Data_exp3[1][i]], [0.01, 0.01], [1, 1])
Res_Amb, I_Res_Amb = 0.6, 0.01
Res = Data_exp3[0]/Data_exp3[1]
for i in range(len(Data_exp3[0])):
    I_Res_rel[i] = IncertidumbreDiv([Res[i], Res_Amb], [I_Res[i], I_Res_Amb], [1, 1])
Res_rel = Res/Res_Amb
R = np.array([1.0, 1.43, 1.87, 2.34, 2.85, 3.36, 3.88, 4.41, 4.95, 5.48, 6.03, 6.58, 7.14, 7.71, 8.28, 8.86, 9.44, 10.03])
T = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
T_prueba = np.array([1701.47003348, 1796.16856756, 1887.47150532, 1967.74881812])
Rad_prueba = np.array([6.7, 8.4, 10.1, 12.0])
exp_p, I_exp_p, fitx_p, fity_p = Regresion(R, T, FuncionPrueba, [1, 1, 1])

def f(x):
    return exp_p[0] + exp_p[1]*x + exp_p[2]*x**2

T3 = f(Res_rel)
exp3, I_exp3, fitx3, fity3 = Regresion(np.log(T3), np.log(Data_exp3[2]), Funcion3, [1, 4])
exp3_d, I_exp3_d, fitx3_d, fity3_d = Regresion(np.log(T_prueba), np.log(Rad_prueba), Funcion3, [1, 4])
r3_1 = MinimosCuadrados(np.log(T3), np.log(Data_exp3[2]))
r3_2 = MinimosCuadrados(np.log(T_prueba), np.log(Rad_prueba))
print('Resistencias del filamento')
print(Res)
print('Incertidumbres de las resistencias del filamento')
print(I_Res)
print('Resistencias relativas para la data')
print(Res_rel)
print('Incertidumbres de las resistencias relativas para la data')
print(I_Res_rel)
print('Temperaturas')
print(T3)
print('Temperaturas a la cuarta')
print(T3**4)
print('Curva Ley SB altas T (todos los datos)')
print(f'a0 = {exp3[0]} +/- {I_exp3[0]}')
print(f'a1 = {exp3[1]} +/- {I_exp3[1]}')
print(f'r^2 = {r3_1}')
print('Curva Ley SB altas T (ultimos datos)')
print(f'a0 = {exp3_d[0]} +/- {I_exp3_d[0]}')
print(f'a1 = {exp3_d[1]} +/- {I_exp3_d[1]}')
print(f'r^2 = {r3_2}')

print('')
print('EXP 4')

T1 = np.array([26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129])
R1 = np.array([95.447, 91.126, 87.022, 83.124, 79.422, 75.903, 72.560, 69.380, 66.356, 63.480, 60.743, 58.138, 55.658, 53.297, 51.048, 48.905, 46.863, 44.917, 43.062,41.292, 39.605, 37.995, 36.458, 34.991, 33.591, 32.253, 30.976, 29.756, 28.590, 27.475, 26.409, 25.390, 24.415, 23.483, 22.590, 21.736, 20.919, 20.136, 19.386, 18.668, 17.980, 17.321, 16.689, 16.083, 15.502, 14.945, 14.410, 13.897, 13.405, 12.932, 12.479, 12.043, 11.625, 11.223, 10.837, 10.467, 10.110, 9.7672, 9.4377, 9.1208, 8.8160, 8.5227, 8.2406, 7.9691, 7.7077, 7.4562, 7.2140, 6.9806, 6.7559, 6.5394, 6.3308, 6.1298, 5.9361, 5.7493, 5.5693, 5.3956, 5.2281, 5.0666, 4.9107, 4.7603, 4.6151, 4.4750, 4.3397, 4.2091, 4.0829, 3.9611, 3.8434, 3.7297, 3.6198, 3.5136, 3.4110, 3.3118, 3.2158, 3.1230, 3.0333, 2.9465, 2.8625, 2.7813, 2.7027, 2.6266, 2.5530, 2.4817, 2.4126, 2.3458])
exp_p1, I_exp_p1, fitx_p1, fity_p1 = Regresion(R1, T1, FuncionPrueba1, [-100, 300, .1])

def f1(x):
    return exp_p1[0] + exp_p1[1]/x**exp_p1[2]
print(exp_p1)

Tamb = 23 + 273
Ramb = 87.2
Tk = f1(Data_exp4[0]) + 273
exp4, I_exp4, fitx4, fity4 = Regresion(Tk**4 - Tamb**4, Data_exp4[1], Funcion4, [1, 1])
r4 = MinimosCuadrados(Tk**4 - Tamb**4, Data_exp4[1])

print('Temperaturas a partir de R')
print(Tk)
print('Diferencia de temperaturas a la cuarta')
print(Tk**4 - Tamb**4)
print('Curva Ley SB bajas T')
print(f'a0 = {exp4[0]} +/- {I_exp4[0]}')
print(f'a1 = {exp4[1]} +/- {I_exp4[1]}')
print(f'r^2 = {r4}')

plt.figure(figsize = (8, 8))
plt.plot(fitx2, fity2, 'k--')
plt.plot(Data_exp2_2[0]/100, Data_exp2_2[1] - Rad_prom, 'ko')
plt.xlabel(r'$x$ (m)', fontsize = 18)
plt.ylabel(r'Rad (mV)', fontsize = 18)
plt.legend(['Ajuste', 'Datos'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(fitx21, fity21, 'k--')
plt.plot(1/((Data_exp2_2[0]/100)**2), Data_exp2_2[1] - Rad_prom, 'ko')
plt.xlabel(r'$1/x^2$ (m$^{-2}$)', fontsize = 18)
plt.ylabel(r'Rad (mV)', fontsize = 18)
plt.legend(['Ajuste', 'Datos'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(np.exp(fitx3), np.exp(fity3), 'k--')
plt.plot((T3), (Data_exp3[2]), 'ko')
plt.xlabel(r'$T$ (K)', fontsize = 18)
plt.ylabel(r'Rad (mV)', fontsize = 18)
plt.legend(['Ajuste 1', 'Datos'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(np.exp(fitx3), np.exp(fity3), 'k--')
plt.plot(np.exp(fitx3_d), np.exp(fity3_d), 'k:')
plt.plot((T3), (Data_exp3[2]), 'ko')
plt.xlabel(r'$T$ (K)', fontsize = 18)
plt.ylabel(r'Rad (mV)', fontsize = 18)
plt.legend(['Ajuste 1', 'Ajuste 2', 'Datos'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(fitx4, fity4, 'k--')
plt.plot(Tk**4 - Tamb**4, Data_exp4[1], 'ko')
plt.xlabel(r'$T_k^4 - T_{amb}^4$ (K$^4$)', fontsize = 18)
plt.ylabel(r'Rad (mV)', fontsize = 18)
plt.legend(['Ajuste', 'Datos'], fontsize = 18)
plt.tick_params(labelsize = 15)
plt.grid()
plt.show()

