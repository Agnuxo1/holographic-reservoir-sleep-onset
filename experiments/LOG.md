# Registro de experimentos — Track 03 Sleep onset (EEG/EMG Foundation Challenge 2026)

Métrica oficial: W-bMAE por grabación, promediada entre grabaciones (menor es mejor).
Mi reimplementación de la métrica se comprobó idéntica a `benchmark_utils.metrics`
del kit oficial sobre entradas aleatorias (238,18562721678177 en ambas, diferencia 0).

Referencias publicadas por los organizadores en el split público de Muse:

| Control publicado | W-bMAE (s) |
|---|---|
| Constante 35 s (elegida en validación) | 60,7 |
| EEGNet | 60,6 |
| CTNet | 60,1 |
| REVE afinado | 72,9 |

Es decir: **ningún control publicado del campo mejora de forma apreciable a una
constante**. Eso no es casual, y la hipótesis de abajo explica por qué.

## Hipótesis de partida

**H0 (estructura de la tarea).** El objetivo es `y = min(600, D - e)`, donde `e` es el
tiempo transcurrido desde el inicio de la grabación y `D` el instante de la primera
época N2. Las ventanas llegan al modelo *en orden temporal, desde el inicio de la
grabación*, una a una, y el evaluador oficial dice literalmente «A model can adapt
within a stream» y llama a `reset_state()` al empezar cada grabación
(`benchmark_utils/streaming.py`). Por tanto un modelo **con estado** conoce `e`
exactamente: le basta contar sus propias llamadas. Un modelo sin estado por ventana
—EEGNet, CTNet, REVE— no puede conocer `e` por mucho que mejore su lectura del EEG,
y de ahí que todos empaten con la constante.

**H1 (tecnología propia).** El reservorio holográfico, que mantiene estado dentro de
la grabación y lo borra entre grabaciones, mejora la métrica oficial frente al control
profesional del campo, porque integra la historia del EEG *además* de llevar el reloj.

**H1b (honestidad del reparto).** Parte de la mejora de H1 es aritmética pura (saber
`e`) y no mérito del EEG ni del reservorio. Para no atribuir al reservorio lo que no
produce, la escalera de modelos separa las dos cosas:

| Modelo | Reloj | EEG de la ventana | Historia del EEG | Qué aísla |
|---|---|---|---|---|
| A constante | no | no | no | suelo publicado |
| B ridge sin estado | no | sí | no | lo que ve EEGNet |
| C solo reloj | sí | **no** | no | la aritmética sola |
| D reloj + ventana | sí | sí | no | aporte del EEG instantáneo |
| E reservorio holográfico | sí | sí | sí | sistema propuesto |
| F reservorio ESN | sí | sí | sí | igual que E sin la entropía SHA-256 |
| G reservorio sin reloj | no* | sí | sí | el reservorio por sí solo |

\* G sí lleva estado, pero sin la codificación explícita del tiempo ni la tabla a priori.

El aporte real del reservorio es **E − D**. El aporte de la parte *holográfica*
concreta (la entropía SHA-256 en la diagonal de la matriz recurrente) es **E − F**.
Si E − D no es distinto de cero, se dice tal cual.

## Validación

Dos regímenes, porque la fase sellada puntúa un macropromedio de participantes vistos
y no vistos en entrenamiento:

- `unseen`: k-fold agrupado por participante (sujeto excluido).
- `seen`: grabaciones excluidas de participantes que sí están en entrenamiento.

Todo se ajusta solo con la parte de entrenamiento del pliegue, incluida la tabla del
reloj y la estandarización de rasgos. Reservorios con 3 semillas.

## Entradas

| id | fecha | hipótesis | config | semilla | métrica | decisión |
|---|---|---|---|---|---|---|

### Iteración 1 — el reservorio tal como lo diseñé primero (30-10, pliegue 0, régimen unseen)

| Modelo | W-bMAE (s) |
|---|---|
| A constante | 60,39 |
| B ridge sin estado | 53,70 |
| C solo reloj | 59,18 |
| D reloj + ventana | 52,45 |
| E reservorio holográfico (s0/s1/s2) | 61,47 / 61,41 / 61,01 |
| F reservorio ESN (s0/s1/s2) | 61,48 / 61,43 / 61,00 |
| G reservorio sin reloj (s0/s1) | 61,92 / 61,62 |

Tres cosas, y ninguna es la que esperaba:

1. **A da 60,39, muy cerca del 60,7 publicado.** La tubería y la métrica reproducen el
   suelo de los organizadores, así que los números son comparables con su tabla.
2. **H0 es falsa en su parte fuerte.** El reloj por sí solo (C) baja de 60,39 a 59,18:
   1,2 s, no el salto que predije. La correlación de la latencia con el tiempo transcurrido
   es 0,419, no 1: las duraciones varían mucho y el tope de 600 s aplana las ventanas
   tempranas. Lo que de verdad mueve la métrica es el EEG con una lectura ajustada a la
   pérdida correcta (B, 53,70).
3. **E ≈ F en las tres semillas** (61,47/61,48, 61,41/61,43, 61,01/61,00): la entropía
   SHA-256 de la diagonal no cambia nada medible. Era previsible por su escala (0,05) y
   ahora está medido.

Causa del mal resultado de E: **fallo mío de diseño**, no del reservorio. Su lectura veía
el estado pero no los rasgos de la ventana actual, así que con `leak=0,3` el estado es un
promedio borroso del pasado reciente y el modelo no contenía al control con el que se le
comparaba. Corregido: la lectura recibe estado + ventana actual + reloj, con lo que E pasa
a ser un superconjunto estricto de D y `E − D` mide lo que aporta la *historia*. Se añade
el peldaño H (reservorio solo con estado) para conservar medido el diseño anterior.

### Iteración 2 — el reservorio corregido, barrido en el split de validación oficial

Elegir sobre `val` (participantes sin grabación de test, 88 grabaciones) y no sobre las 40
de test, que son el set de evaluación del warm-up y se dejan intactas.

Ridge: alpha es casi plano (B: 55,7 entre 3 y 30; D: 52,84 con alpha 3, 52,95 con 10).

Reservorio (size 100): el patrón es inequívoco y va en contra de H1.

| leak | alpha 10 | alpha 100 | alpha 1000 |
|---|---|---|---|
| 0,1 | 80,36 | 84,54 | 62,51 |
| 0,3 | 56,13 | 55,05 | 62,73 |
| 0,7 | **52,80** | 53,28 | 63,11 |

**Cuanto menos memoria tiene el reservorio, mejor puntúa.** En su extremo rápido
(leak 0,7) da 52,80 frente a 52,84 de D: un empate, 0,04 s. Con `size=300` y `leak=0,1`
se va a 112,44.

Diagnóstico de por qué el estado estorba, con train y val a la vista
(size 100, leak 0,3, alpha 10):

| | l1_iters=0 | 3 | 12 |
|---|---|---|---|
| D val / train | 58,71 / 55,25 | 53,78 / 50,93 | 52,95 / 50,52 |
| E val / train | 58,27 / 59,37 | 55,46 / 55,51 | 56,13 / 55,87 |

E empeora **también en entrenamiento**. No es sobreajuste: las dimensiones de estado, muy
autocorreladas entre sí, mal condicionan el sistema y degradan el ajuste de la parte que sí
informa. La tabla también deja claro que el ajuste a L1 por mínimos cuadrados reponderados
vale 5,8 s de métrica en D (58,71 → 52,95), bastante más que cualquier cosa que haya
aportado el reservorio.

### Iteración 3 — resultado final: 5 pliegues, 2 regímenes, 3 semillas

Medias entre pliegues (desviación entre pliegues), 5 pliegues por régimen:

| Modelo | seen | unseen |
|---|---|---|
| A constante | 61,89 ± 1,29 | 61,88 ± 1,02 |
| C solo reloj | 55,76 ± 2,98 | 56,75 ± 1,56 |
| B ridge sin estado | 53,69 ± 1,42 | 54,40 ± 1,01 |
| G reservorio sin reloj | 53,69 ± 1,08 | 54,55 ± 1,40 |
| H reservorio solo estado | 52,75 ± 2,43 | 53,84 ± 1,37 |
| E reservorio holográfico | 51,47 ± 1,82 | 52,72 ± 1,37 |
| F reservorio ESN | 51,47 ± 1,82 | 52,71 ± 1,37 |
| J reservorio residual | 50,77 ± 1,99 | 52,11 ± 1,47 |
| D reloj + ventana | 50,41 ± 2,09 | 51,73 ± 1,57 |
| **I memoria explícita 10 s** | **50,06 ± 2,05** | **51,21 ± 1,44** |

La variación entre pliegues (±1,5 a ±2,5 s) es mayor que las diferencias entre los modelos
de cabeza, así que comparar medias no decide nada. Lo que decide es la diferencia **dentro
de cada pliegue**, donde las grabaciones excluidas son las mismas para los dos modelos:

| Comparación | seen | unseen | pliegues ganados |
|---|---|---|---|
| E − F (la entropía SHA-256) | **+0,00 ± 0,00** | **+0,00 ± 0,00** | 0 de 5 |
| E − D (el reservorio frente al control) | +1,06 ± 0,77 | +0,99 ± 1,02 | 1 de 5, 1 de 5 |
| J − D (reservorio como corrección) | +0,36 ± 0,36 | +0,38 ± 0,18 | 1 de 5, 0 de 5 |
| I − D (memoria explícita de 10 s) | **−0,35 ± 0,16** | **−0,52 ± 0,24** | **5 de 5, 5 de 5** |
| I − E | −1,41 ± 0,73 | −1,51 ± 0,82 | 5 de 5, 5 de 5 |
| H − E (quitarle la ventana actual) | +1,28 ± 0,67 | +1,12 ± 0,28 | 0 de 5 |
| D − B (añadir el reloj) | −3,28 ± 2,55 | −2,67 ± 1,56 | 5 de 5 |
| B − A (añadir el EEG) | −8,20 ± 1,34 | −7,48 ± 0,69 | 5 de 5 |
| C − A (solo el reloj) | −6,12 ± 2,45 | −5,13 ± 2,22 | 5 de 5 |

**Conclusiones, con el signo que han salido:**

1. **H1 RECHAZADA.** El reservorio holográfico no mejora al control profesional: es **peor**
   por 1,0 s, y pierde en 4 de 5 pliegues en los dos regímenes. Como corrección sobre el
   control (J) sigue perdiendo por 0,4 s. JEV lo confirma con 0,93
   (`jev/20261030-h1/answer.json`, provenance=jev).
2. **La parte holográfica no existe a efectos de medida.** E − F = +0,00 ± 0,00 en los dos
   regímenes: la entropía SHA-256 de la diagonal da el mismo número hasta el último decimal
   que no ponerla. Lo que hay ahí es una red de estado de eco.
3. **La memoria sí aporta, pero poco y corta.** Una media móvil exponencial de 10 s gana al
   control por 0,5 s **en los 10 pliegues de los dos regímenes**: pequeño pero no es ruido.
   La memoria larga perjudica.
4. **Lo que de verdad mueve la métrica** es la combinación de rasgos espectrales clásicos
   (−7,5 s sobre la constante), el reloj (−2,7 s más) y, sobre todo, ajustar la lectura a la
   pérdida real del concurso en vez de a la cuadrática (−5,8 s).
5. **Contra la tabla publicada:** 51,21 (unseen) y 50,06 (seen) frente a 60,1 del mejor
   control publicado (CTNet). Verificado con el arnés oficial sobre el split público de
   test: **55,816**, idéntico al número offline, con 11,9 s de inferencia para 5.869
   ventanas (el límite oficial son 80 minutos).

Aviso honesto sobre el ruido: en ese mismo split de test público el reservorio da 55,741,
es decir **0,08 s mejor** que el sistema elegido, al revés que en los 10 pliegues de
validación cruzada. Con 40 grabaciones esa diferencia no significa nada; se deja escrita
porque va en contra de la conclusión y omitirla sería elegir los datos.
