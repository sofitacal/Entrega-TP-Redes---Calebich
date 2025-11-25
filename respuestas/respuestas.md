# Preguntas sobre el ejemplo de clasificación de imágenes con PyTorch y MLP
## 1. Dataset y Preprocesamiento
* ¿Por qué es necesario redimensionar las imágenes a un tamaño fijo para una MLP?
Las MLP reciben como entrada vectores 1D de tamaño fijo. Si se ingresan imágenes con distinta cantidad de píxeles no se tendrán la misma cantidad de valores al aplanarlas a vectores. Cómo se diseña la MLP con cierta cantidad de pesos, si ingresan menos o más valores de los esperados no funcionan. Además, el tener imágenes del mismo tamaño permite crear tensors para lograr un batching más eficiente. 

* ¿Qué ventajas ofrece Albumentations frente a otras librerías de transformación como torchvision.transforms?
Según la documentación de Albumentations, las principales diferencias con Torchvision son: el uso de arrays de Numpy en vez de Pytorch tensors, se usan más parámetros para transformaciones más detalladas, tiene soporte incluido para “mask augmentation” y maneja mejor los bounding boxes y keypoints. Además, es más veloz para transformaciones pesadas, incluye técnicas como CLAHE, ElasticTransform, GridDistortion, entre otras que torchvision no tiene y permite declarar probabilidades directamente en cada transformación.
La documentación incluye también una tabla extensa con las transformaciones geométricas básicas de albumentations y torchvision, argumentando por qué la primera es mejor o igual. 


* ¿Qué hace A.Normalize()? ¿Por qué es importante antes de entrenar una red?
La función A.Normalize() realiza dos etapas de normalización antes de que ingresen a la red neuronal. Primero escala los píxeles, pasando del rango original [0, 255] a floats entre [0,1]. Esto se realiza porque los modelos de aprendizaje profundo trabajan de manera más estable con valores pequeños y continuos, ya que se evita la saturación de funciones de activaciones que luego generan gradientes muy pequeños y una actualización de pesos deficiente. El segundo paso es estandarizar por canal restando la media y dividiendo por el desvío estándar. Esto ayuda a que la distribución de datos se centre en el cero y que tenga una varianza unitaria. Así se logra mejorar la estabilidad numérica del entrenamiento, acelerar la convergencia del optimizador y lograr escalas comparables entre todos los canales.

* ¿Por qué convertimos las imágenes a ToTensorV2() al final de la pipeline?
Para la clasificación se utiliza MLPClassifier de la librería PyTorch. Si bien las transformaciones de Albumentations operan sobre imágenes representadas como arrays de NumPy, la librería Pytorch trabaja con datos en forma de tensores. Es por esto que se convierte cada imágen en tensor con ToTensorV2() antes de pasar por la red neuronal y ser clasificado. 

## 2. Arquitectura del modelo

* ¿Por qué usamos una red MLP en lugar de una CNN aquí? ¿Qué limitaciones tiene?
En este caso, se utiliza una MLP como baseline para poder entender cuál es el proceso sin arquitecturas complejas y analizar el impacto del preprocesamiento y las transformaciones sin efectos de convoluciones o kernels. Además, nos sirve para saber que la CNN debe poder superar el rendimiento de la red MLP. 
Hay varias limitaciones en el uso de las redes MLP. Una de ellas es la pérdida de información espacial, ya que requieren que se aplane la imágen originalmente con tres canales RGB en un vector de una única dimensión. Esto genera que se pierda la relación espacial entre vectores, dificultando la búsqueda de relaciones en cuanto a bordes, texturas y formas, que son esenciales para el diagnóstico de piel. Por otro lado, al aplanar los vectores, estos pasan a tener una gran cantidad de valores. En el caso de la red que utilizamos, las imágenes son de 64x64x3 (después de aplicar la normalización de la resolución), por lo que los vectores de entrada tienen 12288 valores. Como cada valor tendrá su propio parámetro, según la cantidad de neuronas por capa se necesitará una gran cantidad de parámetros, lo que lo hace más propenso al overfitting y lento de entrenar. Esto lleva entonces a que tenga peor capacidad de generalización y un menor accuracy de validation. 

* ¿Qué hace la capa Flatten() al principio de la red?
La capa flatten() convierte las imágenes de entrada de 64x64x3 en vectores de una única dimensión con 12.288 valores. 

* ¿Qué función de activación se usó? ¿Por qué no usamos Sigmoid o Tanh?

Se utiliza la función de activación ReLU, que pone en cero los valores negativos y deja los valores positivos como están. Así resuelve el problema de saturación que si tienen la sigmoidea y tanh: como comprimen los valores en rangos limitados y generan gradientes pequeños cuando los valores son muy grandes o chicos se causa el vanishing gradient (valores del gradiente tienden a cero y deja de aprender el modelo). Además de esto, es más simple computacionalmente ya que solo requiere comparar con cero. 

* ¿Qué parámetro del modelo deberíamos cambiar si aumentamos el tamaño de entrada de la imagen?
Se debe ajustar el input_size utilizado en la primera capa lineal, ya que de este dependen la cantidad de pesos que tendrá la misma (un peso por valor de entrada). 

## 3. Entrenamiento y optimización

* ¿Qué hace optimizer.zero_grad()?

La función del optimizer.zero grad() es reiniciar a cero los gradientes de todos los tensores que gestiona el optimizador. Lo que hace PyTorch es acumular los gradientes en .grad cada vez que se invoca loss.backward(). Esto significa que, si existen valores previos, los nuevos gradientes calculados se suman a ellos aritméticamente en lugar de sobrescribirlos. Si no se ejecutara optimizer.zero_grad() al inicio de cada iteración, los gradientes del lote actual se sumarían a los de todos los lotes anteriores. Esto provocaría que la magnitud de los gradientes creciera indefinidamente, causando actualizaciones de pesos erróneas y excesivas que impedirían la convergencia del modelo.

* ¿Por qué usamos CrossEntropyLoss() en este caso?

Se utiliza porque el problema a resolver es de clasificación multiclase, o sea que cada imágen debe asignarse a una única clase o categoría. Ésta  mide qué tan lejos están las predicciones del modelo respecto de la clase verdadera y penaliza más fuertemente cuando el modelo asigna alta probabilidad a clases incorrectas, promoviendo una mejor separación entre categorías. Además, funciona bien con salidas no normalizadas (logits), lo que evita problemas de saturación que aparecen con activaciones como sigmoide o tanh en la última capa.

* ¿Cómo afecta la elección del tamaño de batch (batch_size) al entrenamiento?
La elección del tamaño del batch va a afectar en la velocidad y calidad del entrenamiento. Si es grande se promedia sobre más ejemplos, por lo que el cálculo del gradiente es más estable. Sin embargo, si es demasiado grande, puede requerir mucha memoria y generar que el modelo no logre generalizar (overfitting). Por otro lado, un batch más chico introduce más ruido en el gradiente (si hay pocas muestras es dificil representar la distribución real del dataset) pero el entrenamiento es menor estable y requiere de más épocas para converger. 

* ¿Qué pasaría si no usamos model.eval() durante la validación?
Esta es la función que permite que el modelo no se comporte como si estuviera en modo entrenamiento al momento de hacer la validación. En este modo, capas como Dropout continúan apagando neuronas de manera aleatoria y Batch Normalization sigue recalculando medias y varianzas con cada batch, produciendo salidas inestables e inconsistentes. Como resultado, las predicciones durante la validación varían entre iteraciones, la pérdida y la accuracy se vuelven ruidosas, y la evaluación deja de reflejar el desempeño real del modelo. Esto puede llevar a interpretar erróneamente el progreso, activar mal el early stopping o guardar pesos incorrectos. 

## 4. Validación y evaluación

* ¿Qué significa una accuracy del 70% en validación pero 90% en entrenamiento?
Si el accuracy del entrenamiento es mucho mayor que el de validación, puede ser que se haya dado el fenómeno de overfitting. El modelo aprende tan bien los datos de entrenamiento que luego no es capaz de generalizar para otros datos, por lo que tiene un rendimiento deficiente frente a datos nuevos. 

* ¿Qué otras métricas podrían ser más relevantes que accuracy en un problema real?
Algunas de las métricas que deberían considerarse son el F1-Score, precision, recall y matriz de confusión.

* ¿Qué información útil nos da una matriz de confusión que no nos da la accuracy?
La matríz de confusión muestra qué clases se están “confundiendo” entre sí. Mientras que la accuracy solo indica el porcentaje total de aciertos, la matriz de confusión muestra explícitamente cuántos ejemplos de cada clase fueron correctamente clasificados (diagonal) y cuántos fueron mal asignados a otras clases (celdas fuera de la diagonal). Esto permite identificar patrones de error: por ejemplo, detectar si dos tipos de lesiones de piel se confunden sistemáticamente, revisar si hay imágenes mal anotadas, evaluar si el modelo necesita más datos de una clase específica o si deben ajustarse hiperparámetros.

* En el reporte de clasificación, ¿qué representan precision, recall y f1-score?
Estos describen distintos aspectos del desempeño del modelo para cada clase. La precisión indica qué proporción de las predicciones positivas del modelo son correctas. Es decir, mide los falsos positivos. El Recall  indica qué proporción de los casos realmente pertenecientes a una clase fueron detectados correctamente por el modelo; o sea los falsos negativos. Por último, el f1-score resume ambas métricas en un solo valor, equilibrando la capacidad del modelo para evitar falsos positivos y falsos negativos.

## 5. TensorBoard y Logging

* ¿Qué ventajas tiene usar TensorBoard durante el entrenamiento?
Usar TensorBoard durante el entrenamiento permite visualizar en tiempo real cómo está aprendiendo el modelo y diagnosticar problemas que no se ven solo mirando la loss final. Da la posibilidad de monitorear curvas de loss y métricas (train y valid) para detectar sobreajuste, subentrenamiento o inestabilidad; inspeccionar pesos, gradientes e histogramas y revisar imágenes, embeddings o predicciones ejemplo a ejemplo. Además, TensorBoard facilita comparar diferentes experimentos, hiperparámetros o versiones del modelo dentro de una misma interfaz, lo que mejora la trazabilidad del proceso de entrenamiento.

* ¿Qué diferencias hay entre loguear add_scalar, add_image y add_text?
Cada uno registra formatos de datos diferentes. El add_scalar se usa para registrar valores numéricos que cambian a lo largo del entrenamiento como loss, accuracy, learning rate, entre otros. Se ven como curvas y permite analizar tendencias, overfitting o problemas de convergencia. El add_image permite guardar imágenes, y se usa para a visualizar datos de entrada, ejemplos reconstruidos, máscaras de segmentación, mapas de calor, etc. El add_text registra texto arbitrario, como etiquetas, mensajes, ejemplos de predicción, información de estados internos o comentarios.

* ¿Por qué es útil guardar visualmente las imágenes de validación en TensorBoard?
Guardar imágenes de validación en TensorBoard permite inspeccionar visualmente qué está prediciendo el modelo y detectar problemas que no se ven solo con métricas numéricas. Esto ayuda a identificar si las imágenes están correctamente preprocesadas, si las clases difíciles se confunden, si existe ruido, mala anotación o si el modelo aprende patrones irrelevantes.

* ¿Cómo se puede comparar el desempeño de distintos experimentos en TensorBoard?
TensorBoard permite comparar experimentos simplemente cargando múltiples runs en la misma interfaz. Cada run corresponde a una carpeta diferente de logs (por ejemplo, distintos valores de batch_size, modelos o tasas de aprendizaje). Al seleccionarlos en el panel lateral, TensorBoard superpone las curvas de métricas como loss, accuracy o F1-score, lo que facilita comparar cómo evoluciona cada experimento. También pueden contrastarse histogramas, imágenes o gráficos de pesos entre los runs.

## 6. Generalización y transferencia 

* ¿Qué cambios habría que hacer si quisiéramos aplicar este mismo modelo a un dataset con 100 clases?
Principalmente habría que modificar la capa de salida (el último linear) para que tenga 100 neuronas, ya que cada neurona representa la probabilidad de una clase distinta. También habría que revisar la capacidad del modelo, específicamente ver si las capas ocultad también deben tener mayor cantidad de neuronas para capturar la mayor variabilidad presente en los datos o no. 

* ¿Por qué una CNN suele ser más adecuada que una MLP para clasificación de imágenes?
Esto se debe a que está diseñada específicamente para analizar la estructura espacial de las imágenes. Las CNN utilizan convoluciones, que permiten detectar patrones locales (bordes, texturas, formas) y construir representaciones jerárquicas cada vez más abstractas, manteniendo la relación entre píxeles vecinos. Esto hace que necesiten muchos menos parámetros que una MLP, ya que los filtros se comparten en toda la imagen, lo cual reduce el riesgo de sobreajuste y permite entrenar modelos mucho más profundos y expresivos. En cambio, como ya se vino mencionando, una MLP debe aplanar la imagen en un vector unidimensional, perdiendo la información espacial, y además requiere una cantidad enorme de pesos para procesar imágenes grandes, lo que la vuelve ineficiente, propensa al overfitting y con bajo poder para aprender características visuales complejas.

* ¿Qué problema podríamos tener si entrenamos este modelo con muy pocas imágenes por clase?
El principal problema va a ser el overfitting, o sea que va a aprender detalles específicos de las pocas imágenes disponibles en lugar de aprender patrones generales de cada clase. Va a tener un accuracy muy alto en la etapa de training y mucho más bajo en la etapa de validación. 

* ¿Cómo podríamos adaptar este pipeline para imágenes en escala de grises?
Se debe modificar el número de canales y, en consecuencia, el tamaño de entrada del modelo. Una imagen gris tiene forma (1, H, W) en lugar de (3, H, W), por lo que al aplanarla se obtiene un vector mucho más pequeño. Por ejemplo, si la imagen es de 64×64, el tamaño de entrada pasa de 3·64·64 = 12 288 a 1·64·64 = 4096. Además, en el preprocesamiento hay que ajustar A.Normalize() para usar solo una media y una desviación estándar, y luego convertirla con ToTensorV2() como siempre. 

## 7. Regularización

* ¿Qué es la regularización en el contexto del entrenamiento de redes neuronales?
La regularización es un conjunto de técnicas que se aplican durante el entrenamiento para evitar que la red neuronal memorice los datos de entrenamiento y aprenda patrones que generalicen bien a datos nuevos. Su objetivo principal es reducir el overfitting. 

* ¿Cuál es la diferencia entre Dropout y regularización L2 (weight decay)?
Dropout funciona desactivando aleatoriamente un porcentaje de neuronas, obligando a la red a no depender tanto de unidades específicas y fomentando representaciones más distribuidas. En cambio, la regularización L2 (weight decay) penaliza pesos grandes sumando esa penalización a la función de pérdida, lo que empuja a los parámetros a valores más pequeños y estables. Mientras Dropout introduce aleatoriedad estructural, L2 introduce restricción directa sobre los pesos. Ambas reducen overfitting, pero por mecanismos distintos.

* ¿Qué es BatchNorm y cómo ayuda a estabilizar el entrenamiento?
Batch Normalization normaliza las activaciones internas de la red en cada batch, restando por la media y dividiendo por el desvío estándar. Esto evita que las distribuciones internas cambien demasiado durante el entrenamiento. Al estabilizar estas distribuciones, el entrenamiento se vuelve más predecible, los gradientes fluyen mejor y se reducen oscilaciones bruscas en la pérdida y el accuracy.

* ¿Cómo se relaciona BatchNorm con la velocidad de convergencia?
BatchNorm acelera la convergencia porque permite usar tasas de aprendizaje más altas sin que el entrenamiento se vuelva inestable. Como las activaciones se normalizan continuamente, los gradientes se vuelven más consistentes y el optimizador avanza de manera más rápida hacia una solución. En la práctica, se observa que con BatchNorm la red alcanza su mejor desempeño en menos épocas y con curvas más suaves.

* ¿Puede BatchNorm actuar como regularizador? ¿Por qué?
Sí, BatchNorm actúa también como un regularizador. Esto ocurre porque la normalización introduce ruido adicional proveniente del cálculo del promedio y la varianza sobre un batch específico. Ese ruido funciona como una forma sutil de perturbación durante el entrenamiento, lo que dificulta que la red se adapte exactamente a cada ejemplo del conjunto de entrenamiento, ayudando a reducir el overfitting. No es tan fuerte como Dropout, pero aporta una regularización ligera.

* ¿Qué efectos visuales podrías observar en TensorBoard si hay overfitting?
En TensorBoard, el overfitting se ve principalmente como una divergencia creciente entre las curvas de entrenamiento y validación. El accuracy del train continúa subiendo mientras el de validación se estanca o cae, y la loss de train baja mientras la loss de validación comienza a subir. Las curvas se separan cada vez más y suelen volverse más ruidosas en validación, mostrando que el modelo deja de generalizar correctamente.

* ¿Cómo ayuda la regularización a mejorar la generalización del modelo?
La regularización introduce una penalización sobre los pesos o sobre la complejidad del modelo, evitando que éste se adapte demasiado al ruido o a patrones irrelevantes del conjunto de entrenamiento. Fuerza al modelo a aprender representaciones más simples y robustas que se mantengan válidas para datos nuevos.

Métricas previo a los cambios:

![Métricas previo a cambios](respuestas\previo_cambios.png)

Con BatchNorm:

![Métricas con BatchNorm](respuestas\batchnorm.png)

Al utilizar BatchNorm, el entrenamiento se volvió más eficiente y la duración disminuyó en alrededor de un segundo (de 2.2 a 2.1 min). Esto ocurre porque al normalizar las activaciones, el gradiente se vuelve más estable y el optimizador progresa de manera más directa, sin tantos saltos.
Los gráficos muestran curvas de train accuracy y train loss mucho más suaves, sin picos bruscos, lo que indica una mayor estabilidad durante el entrenamiento.
En cuanto al rendimiento, mejoraron tanto el accuracy de entrenamiento como el de validación:
train_accuracy: 65.95% → 71.98%
val_accuracy: 59.44% → 61.11%
Además, la validation loss bajó de 0.87 a 0.75, señal de que el modelo ahora hace predicciones más consistentes y mejor calibradas.

Con BatchNorm y Dropout(0.5)

![Métricas con BatchNorm y Dropout(0.5)](respuestas\batchnorm_dropout.png)

Al combinar BatchNorm con Dropout, vemos que se estabilizó la validación, pero en el caso de mi dataset disminuyó el accuracy de validación a 56,11%. Vale la pena aclarar que ahora el accuracy del train y de validación son muy similares (diferencia del 1%), mientras que antes tenían 10% de diferencia entre sí. Esto tiene sentido porque ambas técnicas actúan como regularizadores que reducen el overfitting: el modelo deja de “memorizar” el conjunto de entrenamiento y aprende representaciones más generales, lo que hace que el rendimiento de train y val se acerquen entre sí. Además, Dropout introduce ruido durante el entrenamiento apagando neuronas al azar, lo que suele bajar el accuracy de train y también puede bajar el de validación en las primeras épocas. Por otro lado, BatchNorm normaliza las activaciones, haciendo el entrenamiento más estable pero también reduciendo la capacidad del modelo de sobreajustarse. Como resultado, el modelo tiende a ser más conservador, menos propenso a picos artificiales de accuracy y más consistente entre train y validación, aunque a costa de una menor performance.

BatchNorm, Dropout(0.5) y L2

![Métricas con BatchNorm, Dropout(0.5) y L2](respuestas\batchnorm_dropout_L2.png)

Al combinar BatchNorm, Dropout (0.5) y regularización L2, el entrenamiento se volvió más estable y los gráficos muestran curvas suaves sin picos bruscos, lo que indica que el modelo generaliza mejor. El train accuracy alcanzó 56.9% y el val accuracy 51.7%, mientras que las pérdidas disminuyeron de manera consistente durante las épocas (la val loss bajó hasta 1.12). Sin embargo, a diferencia de cuando usé solo BatchNorm o BatchNorm + L2, esta combinación no logró mejorar el rendimiento sino que lo redujo, probablemente porque Dropout al 50% está apagando demasiadas neuronas y, sumado al L2, aumenta el sesgo del modelo. En síntesis, aunque la estabilidad mejoró, esta combinación resultó demasiado agresiva como regularización para el dataset y terminó afectando la capacidad predictiva.

¿Cuál fue el mejor?

El modelo que combinó BatchNorm y Dropout (0.5) fue el que mostró el mejor desempeño general, ya que logró el val_accuracy más alto (≈56%) y una diferencia muy pequeña entre el accuracy de entrenamiento y validación, lo que indica una buena capacidad de generalización. En cambio, el modelo con solo BatchNorm tendió más al overfitting, ya que el accuracy de entrenamiento creció más que el de validación, mostrando que el modelo aprendía demasiado los patrones del train pero no mejoraba en datos nuevos. Por otro lado, la combinación BatchNorm + Dropout + L2 introdujo una regularización más fuerte que estabilizó el aprendizaje, pero terminó reduciendo levemente el rendimiento en validación. Por eso, la combinación de BatchNorm + Dropout ofrece el mejor equilibrio entre estabilidad, regularización y performance. En todos se hizo Albumentations

## 8. Inicialización de parámetros

* ¿Por qué es importante la inicialización de los pesos en una red neuronal?
Es importante porque determina cómo comienzan las activaciones y gradientes a propagarse por la red. Una mala inicialización puede hacer que haya vanishing gradient o que estos tiendan a infinito, generar entrenamientos inestables, lentos o incluso impedir que el modelo aprenda, mientras que una buena inicialización mantiene la varianza de las señales entre capas, permitiendo un entrenamiento más eficiente.

* ¿Qué podría ocurrir si todos los pesos se inicializan con el mismo valor?
Si todos los pesos se inicializan con el mismo valor, todas las neuronas de una misma capa producen exactamente la misma salida y reciben el mismo gradiente, lo que provoca que todas aprendan lo mismo y la red pierda capacidad de generalización.

* ¿Cuál es la diferencia entre las inicializaciones de Xavier (Glorot) y He?
La inicialización Xavier está diseñada para mantener la varianza de las activaciones y gradientes aproximadamente constante en redes con activaciones simétricas como tanh o sigmoid, escalando los pesos según el tamaño de la capa. La inicialización de He, en cambio, está pensada para ReLU y sus variantes, utilizando una varianza mayor para compensar que ReLU anula la mitad de las activaciones, manteniendo la propagación estable hacia adelante y hacia atrás.

* ¿Por qué en una red con ReLU suele usarse la inicialización de He?
En redes con ReLU se usa inicialización de He porque esta activación anula aproximadamente el 50% de las salidas, lo que reduce la varianza de las señales conforme avanzan por la red. Esta proporciona pesos con suficiente varianza para contrarrestar esa pérdida y así evitar que las activaciones se hagan demasiado pequeñas, preservando un flujo de gradientes adecuado y acelerando la convergencia del entrenamiento.

* ¿Qué capas de una red requieren inicialización explícita y cuáles no?
Requieren inicialización explícita las capas con parámetros entrenables, como las capas densas, convolucionales y, en general, cualquier capa que tenga pesos y sesgos; no requieren inicialización las capas sin parámetros como activaciones (ReLU, sigmoid, softmax), normalización por lotes en algunos frameworks (que usualmente inicializan sus parámetros automáticamente) y operaciones puramente funcionales como pooling o reshape, que no contienen pesos aprendibles.

![Inicialización de parámetros](respuestas\inicialización_param.png)

En azul He, en rojo Xavier y en celeste Uniform

En los experimentos se observa que la inicialización afecta claramente la velocidad y estabilidad de convergencia: la inicialización He (azul oscuro) converge más rápido y alcanza las mejores accuracies tanto en entrenamiento como en validación, lo cual es esperable porque está diseñada para capas con ReLU y mantiene la varianza estable entre capas. La inicialización Xavier (rojo) también converge de manera estable, pero un poco más lento y con métricas finales levemente inferiores. En cambio, la inicialización Uniform (celeste) muestra el peor desempeño: su aprendizaje es más ruidoso, las curvas tienen más variabilidad y en varias épocas arranca más abajo que las otras dos, lo que indica que distribuye mal la magnitud inicial de los pesos. Ninguna inicialización produjo NaNs ni explosión de pérdida, pero Uniform sí mostró la convergencia más irregular. Esto también se refleja en las métricas de validación: He logra consistentemente mejores accuracies y menor pérdida, Xavier queda en el medio y Uniform queda último. Finalmente, los bias se inicializan en cero porque no afectan la propagación de varianza ni generan problemas de simetría (a diferencia de los pesos), y comenzar con bias = 0 facilita que el modelo aprenda rápidamente desplazamientos sin introducir ruido inicial innecesario.
