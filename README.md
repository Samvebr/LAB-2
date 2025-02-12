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

![image](https://github.com/user-attachments/assets/7a75e91d-6d6f-45f5-84b5-148df67b95b1)

Correlación de Pearson entre seno y coseno: 0.0000
Puesto que la correclacion es 0.000 significa que no hay relacion lineal. Despues se toman los valores de la tabla anterior y se grafican las señales con el comando *stem* para la señal discreta posterioemente se establece el tamaño *plt.figure(figsize)*, las funciones a graficar *plt.stem* y las etiquetas de los ejes y del titulo, como se presenta a continuación.

![image](https://github.com/user-attachments/assets/0b93ae71-e439-412c-991f-1b3e28cd45c5)

Como se observa en el gráfico y en el coeficiente de correlación de Pearson es 0.000 esto se debe a que las dos señales estan desfasadas 90°, estadisticamente los cambios en una señal no estan relacionados linealmente con los cambios en la otra. 

## Señal De Electromiografía (EMG)

En la base de datos de Physionet se escogió la señal “emg_neuropathy.dat” y “a04.emg_neuropathy” del estudio "Examples of Electromyograms". La señal en cuanto a su clasificacion es continua en su formato natural, ya que esta definida para todos los instantes de tiempo. No obstante, puesto que se encuentra en en formato digital la señal es discreta gracias al proceso de muestreo que requieren los archivos .dat, impar debido a que mantiene asimetria natural y no periodica a causa de que no tiene un patrón repetitivo regular por la variabilidad en la activacion muscular. Para que el código pueda leer correctamente los archivos es necesario que se encuentren dentro de la misma carpeta del proyecto.

Posteriormente se agregaron las bibliotecas “wfdb”; lee los registros de las señales fisiológicas de formatos .dat y .hea, y extrae la frecuencia de muestreo y los nombres de los canales, “pandas”; se emplea para organizar datos el DataFrame, “matplotlib.pyplot”; graficar señales e histogramas., “numpy”; cálculos matemáticos y generación de ruido y “scipy.signal”; se utiliza para calcular la correlación entre dos secuencias unidimensionales.

```bash
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
```
Para gráficar la señal se empieza colocando el nombre del archivo en la variable nombre_registro.

```bash
nombre_registo ='emg_neuropathy'
```
Se leen los datos de la señal con ayuda de la biblioteca *wfdb* para procesar registros de datos fisiologicos con *rdrecord(record_name)* se carga el registro, *signal* señal analógica en formato de matriz, *fs* es la frecuencia de muestreo, se especifica en canal con la variables *canales* y la *duracion* calcula la duración total en segundos. Despues se muestra la informacion del archivo en pantalla.
```bash
# Leer los datos de la señal y encabezado
record = wfdb.rdrecord(record_name)
signal = record.p_signal
fs = record.fs
canales = record.sig_name
duracion = len(signal) / fs

# Mostrar información del archivo
print("\n" + "="*50)
print("Información de la señal:")
print(f"• Frecuencia de muestreo: {fs} Hz")
print(f"• Duración total: {duracion:.2f} segundos")
print(f"• Canales disponibles: {canales}")
print("="*50)
```
Mostrando los datos presentados a continuación.

![image](https://github.com/user-attachments/assets/8f02e27f-e215-4317-80b7-f1baebfcad46)


Para analizar solo los 10 primeros segundos de las señal se crea un arreglo de tiempos, este arreglo extrae los datos del primer canal de la señal y delimita el tiempo.

```bash
max_time = 10
max_samples = int(fs * max_time)
limited_signal = signal[:max_samples, :]
limited_time = np.arange(max_samples)/fs
channel_data = limited_signal[:, 0]
```

### Transformada de Fourier
Esta nos permite convertir una señal del dominio del tiempo al dominio de la frecuencia, util para saber que frecuencias componen una señal,
en el caso de la medicina y la ingeniera biomedica, es fundamental en la identificación de patrones anormales, mas especificamente en electromiografia nos ayuda a identificar patrones de activación, relacionados a diferentes patologias.


### Estadisticos Descriptivos en Función de la Frecuencia
Para esta parte del laboratorio se calcuraron los estadisticos descriptivos en función de la frecuencia, es decir, se calculo la media, la mediana, la desviacione estandar y el Histograma de frecuencias.

Para la media se suma todas las frecuencias *frequencies * fft_magnitude*, dadno un valor ponderado de las frecuencias y se dibide por la suma total de las magnitudes *fft_magnitude*
```bash
frec_media = np.sum(frequencies * fft_magnitude)/np.sum(fft_magnitude)
```
En el caso de la mediana la instruccion *np.cumsum(fft_magnitud)* calcula la suma acumulada de las magnitudes de la FFt, luego se divide la suma espectral en en 2, para encontrar el valor correspondiente al 50% de la enería total.

````bash
# Frecuencia Mediana
cumsum = np.cumsum(fft_magnitude)
frec_mediana = frequencies[np.where(cumsum >= cumsum[-1]/2)[0][0]]
```
La desviacion estandar con la instrucción (fft_magnitude * (frequencies - frec_media)**2) en la cual multiplica la diferencia cuadrada por la magnitu de la transformada de Fourier para ponderar las diferencias dependiendo de cada frecuencia a la señal, con *sum* se suman todos los datos ponderados y se divide por la suma total de las magnitudes y con *np.sqrt(varinza)* la raiz cuadrada obtenida en la formula y se presentan los datos en pantalla.
```bash
# Desviación Estándar
varianza = np.sum(fft_magnitude * (frequencies - frec_media)**2)/np.sum(fft_magnitude)
desviacion = np.sqrt(varianza)

# Mostrar resultados en consola
print("\n" + "-"*50)
print("Estadísticas de Frecuencia:")
print(f"• Frecuencia Media: {frec_media:.2f} Hz")
print(f"• Frecuencia Mediana: {frec_mediana:.2f} Hz")
print(f"• Desviación Estándar: {desviacion:.2f} Hz")
print("-"*50 + "\n")
```
Del codigo anterior se obtivieron los siguientes datos estadisticos:
• Frecuencia Media: 696.39 Hz
• Frecuencia Mediana: 573.80 Hz
• Desviación Estándar: 514.00 Hz

A fin de conseguir el histograma para visualizar la distrubución de frecuencias de la señal en el dominio de la misma, empleando los datos de las magnitudes obtenidas de la transformada de Fourier, se adecuo el tamño de grafica y con *plt.hist* se establecieron los parametros de *frequencie* que corresponde a cada punto de la magnitud de la transformada, *bins* define el numero total de intervalos y *weights=fft_magnitude* que pondera las barras del histograma de acuerdo a la magnitud de la transformada.

```bash
plt.figure(figsize=(12, 6))
plt.hist(frequencies, bins=200, weights=fft_magnitude, 
         edgecolor='k', alpha=0.7)
plt.title('Distribución de Frecuencias')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad')
plt.grid()
plt.xlim(0, 500)
plt.tight_layout()
plt.show()
```
Se obtuvo la siguiente grafica.

![image](https://github.com/user-attachments/assets/68b2647e-3f4a-4fe0-8b50-ab29a46fa6dd)

