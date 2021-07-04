import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
import scipy.constants as c

Data1 = np.loadtxt('Datos-GAP-de-energia-Si-_-Ge.txt', skiprows = 16, max_rows = 19).T
Data2 = np.loadtxt('Datos-GAP-de-energia-Si-_-Ge.txt', skiprows = 42, max_rows = 16).T
Data3 = np.loadtxt('Datos-GAP-de-energia-Si-_-Ge.txt', skiprows = 67, max_rows = 24).T
Data4 = np.loadtxt('Datos-GAP-de-energia-Si-_-Ge.txt', skiprows = 95, max_rows = 24).T

def Regresion(x, y, f, beta0):
    model = odr.Model(f)
    data = odr.RealData(x, y)
    myodr = odr.ODR(data, model, beta0)
    out = myodr.run()
    popt = out.beta
    perr = out.sd_beta
    x_fit = np.linspace(min(x), max(x), 100)
    fit = f(popt, x_fit)
    return popt, perr, x_fit, fit

def Determinacion(x, y):
    n = len(x)
    r1 = n*sum(np.multiply(x, y)) - sum(x)*sum(y)
    r2 = np.sqrt(n*sum(np.power(x, 2)) - sum(x)**2)*np.sqrt(n*sum(np.power(y, 2)) - sum(y)**2)
    r = (r1/r2)**2
    return np.around(r, 3)

def f1(B, x):
    return B[0] + B[1]*x

def f2(B, x):
    return B[0] - B[1]*x

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
    
Si1_coef1, I_Si1_coef1, Si1_x1, Si1_fit1 = Regresion(Data1[1], np.log(Data1[0]/1e3), f1, [1, 1])
Si1_coef2, I_Si1_coef2, Si1_x2, Si1_fit2 = Regresion(Data1[3], np.log(Data1[2]/1e3), f1, [1, 1])
Ge1_coef1, I_Ge1_coef1, Ge1_x1, Ge1_fit1 = Regresion(Data2[1], np.log(Data2[0]/1e3), f1, [1, 1])
Ge1_coef2, I_Ge1_coef2, Ge1_x2, Ge1_fit2 = Regresion(Data2[3], np.log(Data2[2]/1e3), f1, [1, 1])

Si2_coef1, I_Si2_coef1, Si2_x1, Si2_fit1 = Regresion(1/(Data3[1] + 273), np.log(Data3[0]/1e6), f2, [1, 0])
Si2_coef2, I_Si2_coef2, Si2_x2, Si2_fit2 = Regresion(1/(Data3[3] + 273), np.log(Data3[2]/1e6), f2, [1, 0])
Ge2_coef1, I_Ge2_coef1, Ge2_x1, Ge2_fit1 = Regresion(1/(Data4[1] + 273), np.log(Data4[0]/1e6), f2, [1, 0])
Ge2_coef2, I_Ge2_coef2, Ge2_x2, Ge2_fit2 = Regresion(1/(Data4[3] + 273), np.log(Data4[2]/1e6), f2, [1, 0])

Si_n1, I_Si_n1 = c.e/(Si1_coef1[1]*c.k*293), I_Si1_coef1[1]*c.e/(c.k*293*Si1_coef1[1]**2)
Si_n2, I_Si_n2 = c.e/(Si1_coef2[1]*c.k*293), I_Si1_coef2[1]*c.e/(c.k*293*Si1_coef2[1]**2)
Ge_n1, I_Ge_n1 = c.e/(Ge1_coef1[1]*c.k*293), I_Ge1_coef1[1]*c.e/(c.k*293*Ge1_coef1[1]**2)
Ge_n2, I_Ge_n2 = c.e/(Ge1_coef2[1]*c.k*293), I_Ge1_coef2[1]*c.e/(c.k*293*Ge1_coef2[1]**2)


Si_gap1, I_Si_gap1 = Si2_coef1[1]*Si_n1*c.k/c.e, IncertidumbreMul([Si2_coef1[1], Si_n1], [I_Si2_coef1[1], I_Si_n1], [1, 1])*c.k/c.e
Si_gap2, I_Si_gap2 = Si2_coef2[1]*Si_n2*c.k/c.e, IncertidumbreMul([Si2_coef2[1], Si_n2], [I_Si2_coef2[1], I_Si_n2], [1, 1])*c.k/c.e
Ge_gap1, I_Ge_gap1 = Ge2_coef1[1]*Ge_n1*c.k/c.e, IncertidumbreMul([Ge2_coef1[1], Ge_n1], [I_Ge2_coef1[1], I_Ge_n1], [1, 1])*c.k/c.e
Ge_gap2, I_Ge_gap2 = Ge2_coef2[1]*Ge_n2*c.k/c.e, IncertidumbreMul([Ge2_coef2[1], Ge_n2], [I_Ge2_coef2[1], I_Ge_n2], [1, 1])*c.k/c.e

Si_n, I_Si_n = np.mean([Si_n1, Si_n2]), np.mean([I_Si_n1, I_Si_n2])
Ge_n, I_Ge_n = np.mean([Ge_n1, Ge_n2]), np.mean([I_Ge_n1, I_Ge_n2])
Si_gap, I_Si_gap = np.mean([Si_gap1, Si_gap2]), np.mean([I_Si_gap1, I_Si_gap2])
Ge_gap, I_Ge_gap = np.mean([Ge_gap1, Ge_gap2]), np.mean([I_Ge_gap1, I_Ge_gap2])

print('Medida de n')
print('Medidas 1 silicio')
print(f'a0 = {Si1_coef1[0]}+/-{I_Si1_coef1[0]}')
print(f'a1 = {Si1_coef1[1]}+/-{I_Si1_coef1[1]}')
print(f'n = {Si_n1}+/-{I_Si_n1}')
print('Medidas 2 silicio')
print(f'a0 = {Si1_coef2[0]}+/-{I_Si1_coef2[0]}')
print(f'a1 = {Si1_coef2[1]}+/-{I_Si1_coef2[1]}')
print(f'n = {Si_n2}+/-{I_Si_n2}')
print('')
print('Medidas 1 germanio')
print(f'a0 = {Ge1_coef1[0]}+/-{I_Ge1_coef1[0]}')
print(f'a1 = {Ge1_coef1[1]}+/-{I_Ge1_coef1[1]}')
print(f'n = {Ge_n1}+/-{I_Ge_n1}')
print('Medidas 2 germanio')
print(f'a0 = {Ge1_coef2[0]}+/-{I_Ge1_coef2[0]}')
print(f'a1 = {Ge1_coef2[1]}+/-{I_Ge1_coef2[1]}')
print(f'n = {Ge_n2}+/-{I_Ge_n2}')
print('')
print('Medida del Gap')
print('Medida 1 silicio')
print(f'a0 = {Si2_coef1[0]}+/-{I_Si2_coef1[0]}')
print(f'a1 = {Si2_coef1[1]}+/-{I_Si2_coef1[1]}')
print(f'Eg = {Si_gap1}+/-{I_Si_gap1}')
print('Medida 2 silicio')
print(f'a0 = {Si2_coef2[0]}+/-{I_Si2_coef2[0]}')
print(f'a1 = {Si2_coef2[1]}+/-{I_Si2_coef2[1]}')
print(f'Eg = {Si_gap2}+/-{I_Si_gap2}')
print('')
print('Medida 1 germanio')
print(f'a0 = {Ge2_coef1[0]}+/-{I_Ge2_coef1[0]}')
print(f'a1 = {Ge2_coef1[1]}+/-{I_Ge2_coef1[1]}')
print(f'Eg = {Ge_gap1}+/-{I_Ge_gap1}')
print('Medida 2 germanio')
print(f'a0 = {Ge2_coef2[0]}+/-{I_Ge2_coef2[0]}')
print(f'a1 = {Ge2_coef2[1]}+/-{I_Ge2_coef2[1]}')
print(f'Eg = {Ge_gap2}+/-{I_Ge_gap2}')
print('')
print('Resultados')
print('Medida n')
print(f'Silicio: n = {Si_n}+/-{I_Si_n}')
print(f'Germanio: n = {Ge_n}+/-{I_Ge_n}')
print('Medida gap')
print(f'Silicio: Eg = {Si_gap}+/-{I_Si_gap}')
print(f'Germanio: Eg = {Ge_gap}+/-{I_Ge_gap}')

plt.figure(figsize = (8, 8))
plt.errorbar(Data1[1], np.log(Data1[0]/1e3), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Si1_x1, Si1_fit1, 'k--', label = 'Ajuste')
plt.legend(loc="upper left",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$V$ (V)', fontsize=15)
plt.ylabel(r'ln($I$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(Data1[3], np.log(Data1[2]/1e3), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Si1_x2, Si1_fit2, 'k--', label = 'Ajuste')
plt.legend(loc="upper left",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$V$ (V)', fontsize=15)
plt.ylabel(r'ln($I$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(Data2[1], np.log(Data2[0]/1e3), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Ge1_x1, Ge1_fit1, 'k--', label = 'Ajuste')
plt.legend(loc="upper left",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$V$ (V)', fontsize=15)
plt.ylabel(r'ln($I$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(Data2[3], np.log(Data2[2]/1e3), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Ge1_x2, Ge1_fit2, 'k--', label = 'Ajuste')
plt.legend(loc="upper left",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$V$ (V)', fontsize=15)
plt.ylabel(r'ln($I$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(1/(Data3[1] + 273), np.log(Data3[0]/1e6), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Si2_x1, Si2_fit1, 'k--', label = 'Ajuste')
plt.legend(loc="upper right",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$1/T$ (1/K)', fontsize=15)
plt.ylabel(r'ln($I_0$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(1/(Data3[3] + 273), np.log(Data3[2]/1e6), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Si2_x2, (Si2_fit2), 'k--', label = 'Ajuste')
plt.legend(loc="upper right",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$1/T$ (1/K)', fontsize=15)
plt.ylabel(r'ln($I_0$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(1/(Data4[1] + 273), np.log(Data4[0]/1e6), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Ge2_x1, Ge2_fit1, 'k--', label = 'Ajuste')
plt.legend(loc="upper right",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$1/T$ (1/K)', fontsize=15)
plt.ylabel(r'ln($I_0$)', fontsize=15)
plt.grid()

plt.figure(figsize = (8, 8))
plt.errorbar(1/(Data4[3] + 273), np.log(Data4[2]/1e6), color = 'k', capsize=2, marker = 'o', ls = 'None', label = 'Datos')
plt.plot(Ge2_x2, Ge2_fit2, 'k--', label = 'Ajuste')
plt.legend(loc="upper right",fontsize=15)
plt.tick_params(labelsize = 12.5)
plt.xlabel(r'$1/T$ (1/K)', fontsize=15)
plt.ylabel(r'ln($I_0$)', fontsize=15)
plt.grid()
plt.show()
