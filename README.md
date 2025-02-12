# LAB-2:  Convolución, Correlación y Transformación

## Justificación
### ¿Qué es una electromiografía?

Un electromiograma (EMG) es una prueba clínica común que se utiliza para evaluar la función de los músculos y los nervios que los controlan. Los estudios EMG se utilizan para ayudar en el diagnóstico y tratamiento de trastornos como las distrofias musculares y las neuropatías. Los estudios de conducción nerviosa que miden qué tan bien y qué tan rápido los nervios conducen los impulsos a menudo se realizan junto con estudios EMG.

En este caso se muestra un hombre de 62 años con dolor lumbar crónico y neuropatía debido a una radiculopatía L5 derecha.
### Metodo de adquisición

Se colocó un electrodo de aguja concéntrica de 25 mm en el músculo tibial anterior de cada sujeto. Luego se pidió al paciente que flexionara suavemente el pie contra resistencia. El electrodo de aguja se reposicionó hasta que se identificaron potenciales de la unidad motora con un tiempo de aumento rápido. Luego se recogieron datos durante varios segundos, momento en el que se pidió al paciente que se relajara y se retiró la aguja.
## Convolución
Se escogio como el sistema *h[n]* cada dígito del código de cada estudiante y para la señal *x[n]* cada dígito de su cedula, con el fin de encontrar la señal resultante *y[n]* de la convolución. Se encontraron los valores y la grafica de *y[n]* a mano, al igual que con la interfaz de Python.

### Valores De La Convolución y Grafica (a Mano)
Para el calculo se multiplicaron cada una de las filas por las columnas y posteriormente se sumaron los datos en diagonales, obteniendo al final 15 valores para la convolucion *y[n]*. Al elaborar la grafica se tomo como eje x la cantidad de datos obtenidos, es decir, 15 datos y como eje x los valores de la señal *y[n]*.

**Convolución de Samuel E. Velandia**


![Imagen de WhatsApp 2025-02-05 a las 11 44 38_65214d87](https://github.com/user-attachments/assets/01c5b06f-6195-40f2-8c51-f8ac9c71529e)

**Convolución de Santiago E. Diaz**

![image](https://github.com/user-attachments/assets/cb7af520-97fd-4925-8fe7-8069e3184d38)
![image](https://github.com/user-attachments/assets/5f6c43dd-1f94-4bef-978e-5eb9b4f80491)

**Convolución de Salome Ortega**

![image](https://github.com/user-attachments/assets/7c078541-92dc-42aa-9b91-81dec51f5f84)
![image](https://github.com/user-attachments/assets/a22bcc9d-5a65-4ecc-82d2-4de8dc2b4818)

El eje x generalmente representa el tiempo o la posicion de la señal de entrada, mientras que el eje y muestra la amplitud de la señal resultante, dicha amplitud es la suma ponderada de la señal de entrada y la respuesta del sistema a esta, mostrando como el sistema transforma la señal.

### Valores De La Convolución y Grafica (Python)
Se emplearon las bibliotecas *matplotlib.pyplot* y *numpy*, la primera cumple la función de graficar las convoluciones y la segunda de todos los calculos matematicos necesarios. 
```bash
import matplotlib.pyplot as plt
import numpy as np
```
Posteriormente se crearon las variables array de la matriz de entrada (codigo) y la matriz de convolucion (cedula), para cada uno de los tres estudiantes, con ayuda de la funcion *np.convolve* que maneja como parametros la matriz de entrada y la de convolucion, y *mode=full* para que muestre todos los datos obtenidos.

```bash
#Matriz de entrada
codigo_salo=np.array([5,6,0,0,7,5,3])
codigo_samu=np.array([5,6,0,0,7,7,7])
codigo_santi=np.array([5,6,0,0,7,5,5])
#Matriz de convolucion
cedula_salo=np.array([1,0,1,3,2,5,9,8,3,4])
cedula_santi=np.array([1,0,9,7,7,8,2,7,2,6])
cedula_samu=np.array([1,0,7,1,1,4,2,2,5,1])
#Convolucion 
con_salo=np.convolve(codigo_salo,cedula_salo,mode='full') 
con_santi=np.convolve(codigo_santi,cedula_santi,mode='full')
con_samu=np.convolve(codigo_samu,cedula_samu,mode='full')
```
Se imprimen los vectores en pantalla utilizando el comando *print*, los cuales dan como resultado:
- Los valores de la convolución de Samuel:  [  5  6  35  47  18  33  90  78 100  77  55  56  63  56  42   7]
- Los valores de la convolución de Santiago:  [  5   6  45  89  84  87 126 141 181 168 125  99  59  87  40  30]
- Los valores de la convolución de Salome:  [  5   6   5  21  35  42  85 120  95  92 118 116  88  67  29  12]

```bash
print('Los valores de la convolución de Samuel: ',con_samu)
print('Los valores de la convolución de Santiago: ',con_santi)
print('Los valores de la convolución de Salome: ',con_salo)
```
Se grafican los datos obtenidos de la convolución mediante el comando *plot*, se ajusta el tamaño y el nombre de los ejes.
```bash
#-----------------------
#CONVOLUCION SALO
#-----------------------
plt.figure(figsize=(12,8))
plt.stem(con_salo)
plt.title('Convolución - Salome')
plt.xlabel('Cantidad de datos')
plt.ylabel('g[n]')
plt.grid()
plt.show()

#-----------------------
#CONVOLUCION SAMU
#-----------------------
plt.figure(figsize=(12,8))
plt.stem(con_samu)
plt.title('Convolución - Samuel')
plt.xlabel('Cantidad de datos')
plt.ylabel('g[n]')
plt.grid()
plt.show()

#-----------------------
#CONVOLUCION SANTI
#-----------------------
plt.figure(figsize=(12,8))
plt.stem(con_santi)
plt.title('Convolución - Santiago')
plt.xlabel('Cantidad de datos')
plt.ylabel('g[n]')
plt.grid()
plt.show()
```
El resultado obtenido de las graficas es:

**Convolución de Samuel E. Velandia**

![image](https://github.com/user-attachments/assets/8723165a-ca55-4482-ac3f-78e1e2c7f58c)


**Convolución de Santiago E. Diaz**

![image](https://github.com/user-attachments/assets/739e7924-3c07-487b-98bf-01242d9ac4bf)

**Convolución de Salome Ortega**
![image](https://github.com/user-attachments/assets/992ec216-8833-4d00-8bd2-0c39ae98be59)

## Correlación entre dos señales 
Se realizo la correlacion entre las señales 𝑥1[𝑛𝑇𝑠] = cos(2𝜋100𝑛𝑇𝑠) y 𝑥2[𝑛𝑇𝑠] = sin(2𝜋100𝑛𝑇𝑠) pra el intervalo 𝑎 0 ≤ 𝑛 < 9 para un periodo de Ts= 1.25𝑚s. Para la elaboracion se necesitan las bilbiotecas *numpy* (calculos matematicos), *matplotlib.pyplot* (para realizar la grafica), *pandas* (analisis y manipulacion de datos) y *scipy.stats* (el analisis estadisticos y probabilisticos).

```bash
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
```

Se definieron los parametros iniciales para ambas señales *N* como el numero de muestras, *z* que hace referencia al periodo en segundos y *f* como la frecuencia obtenida mediante el inverso del periodo *(1/z)*.

``` bash
# Definir parámetros
N = 9  # Número de muestras
z = 0.00125
f = 100  # Frecuencia en Hz
```
Mediante la libreria *numpy* se define el eje de tiempo con la intruccion *arange* y como parametro el el numero maximo de muestras. Y se calculan las señales dadas por el ejercicio.

``` bash
n = np.arange(N)  # n = [0, 1, 2, ..., N-1]
# Calcular las señales
y_seno = np.sin(2 * np.pi * f * n * z)
y_coseno = np.cos(2 * np.pi * f * n * z)
```
Para calcular la correlación de ambas señales se utiliza la Correlación de Pearson, que permite medir la relacion lineal entre dos variables cuantitativas. Se emplea la libreria *scipy.stats* que lo calcula el coeficiente de correlación de Pearson y el valor p, que indica si la correlación estaística es significativa, un valor p bajo sugiere que la correlacion no es debida al azar. Posteriormenete se crea un *DataFrame* con la cantidad de datos o posicion y las dos señales dadas por el ejercicio, para un analisis y proceso de representación mas simple.

```bash
# Calcular la correlación de Pearson
correlation_coefficient, p_value = pearsonr(y_seno, y_coseno)

# Crear la tabla con pandas
df = pd.DataFrame({
    'n': n,
    'Seno': y_seno,
    'Coseno': y_coseno
})
```
Se muestra en consola la tabla de valores creada a través del *DataFrame* y también la correlacion entre las dos funciones (seno y coseno)

```bash
# Mostrar la tabla en la consola
print("Tabla de valores de Seno y Coseno:")
print(df)

# Mostrar el coeficiente de correlación de Pearson
print(f"\n Correlación de Pearson entre seno y coseno: {correlation_coefficient:.4f}")
```
Obteniendo los siguientes resultados:
<p align="center">
![image](https://github.com/user-attachments/assets/7a75e91d-6d6f-45f5-84b5-148df67b95b1)
</p>
Correlación de Pearson entre seno y coseno: 0.0000
Puesto que la correclacion es 0.000 significa que no hay relacion lineal. Despues se toman los valores de la tabla anterior y se grafican las señales con el comando *stem* para la señal discreta posterioemente se establece el tamaño *plt.figure(figsize)*, las funciones a graficar *plt.stem* y las etiquetas de los ejes y del titulo, como se presenta a continuación.

![image](https://github.com/user-attachments/assets/0b93ae71-e439-412c-991f-1b3e28cd45c5)






### Grafica de la señal.

- En la base de datos de Physionet se escogió la señal “emg_neuropathy.dat” y “a04.emg_neuropathy” del estudio "Examples of Electromyograms", para que el código pueda leer correctamente los archivos es necesario que se encuentren dentro de la misma carpeta del proyecto.

- Posteriormente se agregaron las bibliotecas “wfdb”; lee los registros de las señales fisiológicas de formatos .dat y .hea, y extrae la frecuencia de muestreo y los nombres de los canales, “pandas”; se emplea para organizar datos el DataFrame, “matplotlib.pyplot”; graficar señales e histogramas., “numpy”; cálculos matemáticos y generación de ruido y “scipy.stats”; modelo estadístico y distribución normal.

```bash
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
```
- Para gráficar la señal se empieza colocando el nombre del archivo en la variable nombre_registro y el tiempo que se desea a analizar y graficar, para nuestro caso 10s.

```bash
nombre_registo ='emg_neuropathy'
tiempo_max = 10 #Analiza los primeros 10 segundos
```
- Se lee la señal original mediante la función leer_senal(nombre_registro) y se estable el tiempo máximo de muestreo y se almacena en la variable muestras_max.

```bash
senal, fs, canales, tiempo, df= leer_senal(nombre_registro)
muestras_max = int(fs *tiempo_max)
```

- La función leer_senal(nombre_registro) permite leer los archivos .dat y. hea, obtiene la señal de la matriz, la frecuencia de muestreo, establecer los nombres de los canales, la duración total en segundos y el DataFrame que organiza los datos de la señal y los asigna a un dato de tiempo, además muestra los datos en pantalla de la frecuencia de muestreo, la duración de la señal y los canales donde se encuentra la matriz de datos.

```bash
record = wfdb.rdrecord(record_name)  # Lee los archivos .dat y .hea
signal = record.p_signal  # Obtiene la señal en formato de matriz
fs = record.fs  # Frecuencia de muestreo
canales = record.sig_name  # Nombres de los canales (derivaciones)
duracion = len(signal) / fs  # Duración de la señal en segundos

print(f"Frecuencia de muestreo: {fs} Hz")
print(f"Duración de la señal: {duracion:.2f} segundos")
print(f"Canales disponibles: {canales}")

tiempo = [i / fs for i in range(len(signal))]  
df = pd.DataFrame(signal, columns=canales)  
df.insert(0, "Time (s)", tiempo)
return senal, fs, canales, tiempo, df

```
- Después se gáfico los primeros 10 segundos de la señal empleando la función graficar señal, además se muestra en la gráfica los datos estadísticos de la señal, los cuales se obtienen a partir de la función calcular_estadisticas_texto que tiene como parámetros la señal y los primero 10 segundos y muestra en consola los datos estadísticos de la señal (media, desviación estándar, coeficiente de variación, histograma y función de probabilidad).

```bash
texto_estadisticas = calcular_estadisticas_texto(senal, muestras_max)
graficar_senal (tiempo, senal, cananles, tiempo_max, "Senal-EMG")
print( "Estadísticas de la señal original:")
print (texto_estadisticas)
```

- La función “graficar_senal” tiene como parámetros tiempo, senal, canales, tiempo_max, titulo, “texto_anotacion=None”, se establecen los parámetros y el tamaño de la muestra de la señal que se quiere graficar, luego se escoge el tamaño de la gráfica, el nombre de los ejes, el titulo del gráfico, la cuadrilla y se escribe la instrucción plt.show() para que se muestre el grafico.

```bash
def graficar_senal(tiempo,senal,canales,tiempo_max,titulo, texto_anotacion=None):

muestra_max = int(fs *tiempo_max)
tiempo_limitado = tiempo[:muestras_max]
senal_limitada = senal[:muestras_max, :]
fig, ax= plt.subplots(figsize=(12, 6))
for i, canal in enumerate(canales):
 ax.plot(tiempo_limitado, senal_limitada[:, i], label=canal)
ax.set_title(titulo)
ax.set_xlabel("tiempo (s)")
ax.set_ylabel("amplitud (mV)")
ax.legend()
ax.grid()
if texto_anotacion is not None:

  ax.text(0.02, 0.98, texto_anotacion, trasform=ax.transAxes, fontsize=10,
          verticalaligment=?'top', bbox=dict(facecolor='white', alpha=0.7))
plt.tight_layout()
plt.show()
```
## Grafíca:

### Estadisticos

- Frecuencia de muestreo: 4000 Hz
- Duración de la señal: 36.96 segundos
- Canales disponibles: ['EMG']

Estadísticas para EMG:

- Media: 0.0002 mV
- Desviación estándar: 0.2325 mV
- Coeficiente de variación: 101842.59 %

## Transformada de Fourier.

- Esta nos permite convertir una señal del dominio del tiempo al dominio de la frecuencia, util para saber que frecuencias componen una señal,
  en el caso de la medicina y la ingeniera biomedica, es fundamental en la identificación de patrones anormales, mas especificamente
  en electromiografia nos ayuda a identificar patrones de activación, relacionados a diferentes patologias.

### Grafica de la transformada.
### Estadisticos de la transformada.
