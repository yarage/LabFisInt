import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

ValoresT1 = np.loadtxt('PAR.txt', skiprows = 8, usecols = (2, 3), max_rows = 7).T
ValoresT2 = np.loadtxt('PAR.txt', skiprows = 18, usecols = (2, 3), max_rows = 5).T
ValoresT3 = np.loadtxt('PAR.txt', skiprows = 35, usecols = (2, 3, 4), max_rows = 8).T
ValoresT4 = np.loadtxt('PAR.txt', skiprows = 46, usecols = (2, 3, 4), max_rows = 5).T

def MinimosCuadrados(puntosx, puntosy):
    n = len(puntosx)
    graf = []
    valor_x = np.linspace(np.amin(puntosx), np.amax(puntosx), 1000) 
    a1 = (n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy))/(n*sum(np.power(puntosx, 2)) - (sum(puntosx))**2)
    a0 = np.mean(puntosy) - a1*np.mean(puntosx)
    r1 = n*sum(np.multiply(puntosx, puntosy)) - sum(puntosx)*sum(puntosy)
    r2 = np.sqrt(n*sum(np.power(puntosx, 2)) - sum(puntosx)**2)*np.sqrt(n*sum(np.power(puntosy, 2)) - sum(puntosy)**2)
    r = r1/r2
    sigma = np.sqrt(sum((puntosy - a1*puntosx - a0)**2)/(n - 2))
    E_a1 = (np.sqrt(n)*sigma)/np.sqrt(n*sum(np.power(puntosx, 2)) - sum(puntosx)**2)
    for x in valor_x:
        y = a0 + a1*x
        graf.append(y)
    return a0, a1, valor_x, graf, np.around(r**2, 3), E_a1

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

a0_1, a1_1, valorx_1, graf1, r1, Ea1_1 = MinimosCuadrados(ValoresT1[0]*1e3, ValoresT1[1])
a0_2, a1_2, valorx_2, graf2, r2, Ea1_2 = MinimosCuadrados(100/ValoresT2[0], ValoresT2[1])
a0_3_1, a1_3_1, valorx_3_1, graf3_1, r3_1, Ea1_3_1 = MinimosCuadrados(ValoresT3[0]*1e3, ValoresT3[1])
a0_3_2, a1_3_2, valorx_3_2, graf3_2, r3_2, Ea1_3_2 = MinimosCuadrados(ValoresT3[0]*1e3, ValoresT3[2])
a0_4_1, a1_4_1, valorx_4_1, graf4_1, r4_1, Ea1_4_1 = MinimosCuadrados(ValoresT4[0]*1e3, ValoresT4[1])
a0_4_2, a1_4_2, valorx_4_2, graf4_2, r4_2, Ea1_4_2 = MinimosCuadrados(ValoresT4[0]*1e3, ValoresT4[2])

C, I_C = 220, 20
Uc2, I_Uc2 = 1500, 100
dplate, I_dplate = 0.26, 0.0005

VC1, I_VC1 = a1_1*C, IncertidumbreMul([a1_1, C], [Ea1_1, I_C], [1, 1])

Aplate, I_Aplate = np.pi*dplate**2/4, np.pi*IncertidumbreMul([dplate], [I_dplate], [2])/4
qd, I_qd = C*a1_2, IncertidumbreMul([C, a1_2], [I_C, Ea1_2], [1, 1]) 
dummy1, I_dummy1 = Uc2*Aplate, IncertidumbreMul([Uc2, Aplate], [I_Uc2, I_Aplate], [1, 1])
e0, I_e0 = qd/dummy1, IncertidumbreDiv([qd, dummy1], [I_qd, I_dummy1], [1, 1])

d3, I_d3 = 0.0098, 0.0005
qUc1_1, I_qUc1_1 = C*a1_3_1, IncertidumbreMul([C, a1_3_1], [I_C, Ea1_3_1], [1, 1])
qUc1_2, I_qUc1_2 = C*a1_3_2, IncertidumbreMul([C, a1_3_2], [I_C, Ea1_3_2], [1, 1])
dummy2_1, I_dummy2_1 = qUc1_1*d3, IncertidumbreMul([qUc1_1, d3], [I_qUc1_1, I_d3], [1, 1])
dummy2_2, I_dummy2_2 = qUc1_2*d3, IncertidumbreMul([qUc1_2, d3], [I_qUc1_2, I_d3], [1, 1])
Vplastico, I_Vplastico = dummy2_1/Aplate, IncertidumbreDiv([dummy2_1, Aplate], [I_dummy2_1, I_Aplate], [1, 1])
Vaire_1, I_Vaire_1 = dummy2_2/Aplate, IncertidumbreDiv([dummy2_2, Aplate], [I_dummy2_2, I_Aplate], [1, 1])
ep, I_ep = qUc1_1/qUc1_2, IncertidumbreDiv([qUc1_1, qUc1_2], [I_qUc1_1, I_qUc1_2], [1, 1])

d4, I_d4 = 0.003, 0.0005
qUc2_1, I_qUc2_1 = C*a1_4_1, IncertidumbreMul([C, a1_4_1], [I_C, Ea1_4_1], [1, 1])
qUc2_2, I_qUc2_2 = C*a1_4_2, IncertidumbreMul([C, a1_4_2], [I_C, Ea1_4_2], [1, 1])
dummy3_1, I_dummy3_1 = qUc2_1*d4, IncertidumbreMul([qUc2_1, d4], [I_qUc2_1, I_d4], [1, 1])
dummy3_2, I_dummy3_2 = qUc2_2*d4, IncertidumbreMul([qUc2_2, d4], [I_qUc2_2, I_d4], [1, 1])
Vvidrio, I_Vvidrio = dummy3_1/Aplate, IncertidumbreDiv([dummy3_1, Aplate], [I_dummy3_1, I_Aplate], [1, 1])
Vaire_2, I_Vaire_2 = dummy3_2/Aplate, IncertidumbreDiv([dummy3_2, Aplate], [I_dummy3_2, I_Aplate], [1, 1])
ev, I_ev = qUc2_1/qUc2_2, IncertidumbreDiv([qUc2_1, qUc2_2], [I_qUc2_1, I_qUc2_2], [1, 1])

print('Tabla 1')
print(f'El valor de V/Uc es ({a1_1/1e-3} +/- {Ea1_1/1e-3})*10^3, r = {r1}')
print(f'El valor de la capacitancia de las placas con d = 0.003 m es {VC1} +/- {I_VC1} nF')
print('')
print('Tabla 2')
print(f'El valor de V*d es ({a1_2/1e-3} +/- {Ea1_2/1e-3})*10^-3 V*m, r = {r2}')
print(f'El valor de Q*d es {qd} +/- {I_qd} nC*m')
print(f'El area es ({Aplate/1e-3} +/- {I_Aplate/1e-3})*10^-3 m^2')
print(f'El valor de Uc es {Uc2} +/- {I_Uc2} V')
print(f'El valor de la permitividad del aire es ({e0*1e3} +/- {I_e0*1e3})*10^-12 C/N*m')
print('')
print('Tabla 3')
print(f'El valor de V/Uc para el aire es ({a1_3_2/1e-3} +/- {Ea1_3_2/1e-3})*10^-3, r = {r3_1}')
print(f'El valor de Q/Uc para el aire es {qUc1_2} +/- {I_qUc1_2} nF')
print(f'El valor de la permitividad del aire es ({Vaire_1*1e3} +/- {I_Vaire_1*1e3})*10^-12 C/N*m')
print('')
print(f'El valor de V/Uc para el plastico es ({a1_3_1/1e-3} +/- {Ea1_3_1/1e-3})*10^-3, r = {r3_2}')
print(f'El valor de Q/Uc para el plastico es {qUc1_1} +/- {I_qUc1_1} nF')
print(f'El valor de la permitividad del plastico es ({Vplastico*1e3} +/- {I_Vplastico*1e3})*10^-12 C/N*m')
print('')
print(f'El valor de e_plastico es {ep} +/- {I_ep}')
print('')
print('Tabla 4')
print(f'El valor de V/Uc para el aire es ({a1_4_2/1e-3} +/- {Ea1_4_2/1e-3})*10^-3, r = {r4_2}')
print(f'El valor de Q/Uc para el aire es {qUc2_2} +/- {I_qUc2_2} nF')
print(f'El valor de la permitividad del aire es ({Vaire_2*1e3} +/- {I_Vaire_2*1e3})*10^-12 C/N*m')
print('')
print(f'El valor de V/Uc para el vidrio es ({a1_4_1/1e-3} +/- {Ea1_4_1/1e-3})*10^-3, r = {r4_1}')
print(f'El valor de Q/Uc para el vidrio es {qUc2_1} +/- {I_qUc2_1} nF')
print(f'El valor de la permitividad del vidrio es ({Vvidrio*1e3} +/- {I_Vvidrio*1e3})*10^-12 C/N*m')
print('')
print(f'El valor de e_vidrio es {ev} +/- {I_ev}')
print('')

plt.figure(figsize=(8, 8))
plt.plot(ValoresT1[0]*1e3, ValoresT1[1], 'ko')
plt.plot(valorx_1, graf1, 'k--')
plt.xlabel(r'$U_c$ (V)', fontsize = 15)
plt.ylabel(r'$V$ (V)', fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize=(8, 8))
plt.plot(1/ValoresT2[0]*100, ValoresT2[1], 'ko')
plt.plot(valorx_2, graf2, 'k--')
plt.xlabel(r'$1/d$ ($\mathrm{m}^{-1}$)', fontsize = 15)
plt.ylabel(r'$V$ (V)', fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(valorx_3_1, graf3_1, 'k-.')
plt.plot(valorx_3_2, graf3_2, 'k--')
plt.plot(ValoresT3[0]*1e3, ValoresT3[1], 'ko')
plt.plot(ValoresT3[0]*1e3, ValoresT3[2], 'k+', ms = 15, mew = 2)
plt.xlabel(r'$U_c$ (V)', fontsize = 15)
plt.ylabel(r'$V$ (V)', fontsize = 15)
plt.legend(['Plastico', 'Aire', 'Datos plastico', 'Datos aire'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.figure(figsize = (8, 8))
plt.plot(valorx_4_1, graf4_1, 'k-.')
plt.plot(valorx_4_2, graf4_2, 'k--')
plt.plot(ValoresT4[0]*1e3, ValoresT4[1], 'ko')
plt.plot(ValoresT4[0]*1e3, ValoresT4[2], 'k+', ms = 15, mew = 2)
plt.xlabel(r'$U_c$ (V)', fontsize = 15)
plt.ylabel(r'$V$ (V)', fontsize = 15)
plt.legend(['Vidrio', 'Aire', 'Datos vidrio', 'Datos aire'], fontsize = 15)
plt.tick_params(labelsize = 12.5)
plt.grid()

plt.show()