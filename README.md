# Tensor++ 

## Descripción

Tensor++ es una librería desarrollada en C++ que permite manejar tensores de hasta 3 dimensiones, inspirada en herramientas como NumPy y PyTorch.

El proyecto implementa operaciones matemáticas, transformaciones y manejo eficiente de memoria dinámica, culminando en la simulación de una red neuronal simple.

---

## Objetivos

* Implementar una clase `Tensor` desde cero
* Manejar memoria dinámica correctamente (Regla de 5)
* Soportar operaciones matemáticas y transformaciones
* Aplicar conceptos de programación orientada a objetos y polimorfismo
* Simular un flujo básico de red neuronal

---

## Funcionalidades implementadas

### 🔹 Clase Tensor

* Soporte para tensores de 1D, 2D y 3D
* Almacenamiento en memoria dinámica (`double*`)
* Constructor con validación de dimensiones

---

### 🔹 Creación de tensores

* `zeros(shape)` → tensor lleno de ceros
* `ones(shape)` → tensor lleno de unos
* `random(shape, min, max)` → valores aleatorios
* `arange(start, end)` → secuencia numérica

---

### 🔹 Gestión de memoria (Regla de 5)

* Constructor de copia (deep copy)
* Constructor de movimiento
* Operador de asignación (copy y move)
* Destructor

---

### 🔹 Operaciones matemáticas

* Suma: `A + B`
* Resta: `A - B`
* Multiplicación elemento a elemento: `A * B`
* Multiplicación por escalar: `A * 2.0`

✔ Incluye soporte de broadcasting básico

---

### 🔹 Transformaciones de dimensiones

* `view()` → cambia la forma sin copiar datos
* `unsqueeze()` → agrega dimensiones

---

### 🔹 Operaciones avanzadas

* `concat()` → concatenación de tensores
* `dot()` → producto punto
* `matmul()` → multiplicación matricial

---

### 🔹 Polimorfismo

* Clase abstracta `TensorTransform`
* Implementaciones:

  * `ReLU`
  * `Sigmoid`

---

## Red Neuronal Implementada

Se implementa un flujo completo de procesamiento:

```
Input:        1000 × 20 × 20
↓ view
              1000 × 400
↓ matmul (W1)
              1000 × 100
↓ + bias
              1000 × 100
↓ ReLU
              1000 × 100
↓ matmul (W2)
              1000 × 10
↓ + bias
              1000 × 10
↓ Sigmoid
Output:       1000 × 10
```

---

## Ejecución del programa

El archivo `main.cpp` contiene pruebas completas de la librería, incluyendo:

* creación de tensores
* operaciones básicas
* transformaciones
* concatenación
* producto punto y multiplicación matricial
* simulación de una red neuronal simple

---

## Requisitos

Para ejecutar el proyecto se necesita:

* Compilador C++ con soporte para C++17 o superior
* Git (opcional)
* Un editor o IDE como:

  * CLion
  * Visual Studio Code
  * Code::Blocks
  * Dev-C++

---

## Ejecución paso a paso

### 🔹 Opción 1: Descargar desde GitHub

1. Ir al repositorio
2. Click en **Code > Download ZIP**
3. Extraer el proyecto

#### Verificar archivos:

```
Tensor.h
Tensor.cpp
main.cpp
README.md
```

#### Compilar:

```bash
g++ -std=c++17 main.cpp Tensor.cpp -o tensor_app
```

#### Ejecutar:

En Windows:

```bash
tensor_app.exe
```

En Linux / macOS:

```bash
./tensor_app
```

---

### 🔹 Opción 2: Clonar con Git

```bash
git clone https://github.com/RenzoChenHeng/Tensor-Plus-Plus.git
cd Tensor-Plus-Plus
g++ -std=c++17 main.cpp Tensor.cpp -o tensor_app
```

Ejecutar:

```bash
./tensor_app
```

---

### 🔹 Opción 3: Ejecutar en CLion

1. Abrir CLion
2. Seleccionar **Open** y abrir la carpeta del proyecto
3. Esperar a que cargue CMake
4. Presionar Run 

---

## Salida esperada

El programa imprimirá en consola:

* pruebas de tensores
* operaciones matemáticas
* transformaciones
* resultados de la red neuronal

Ejemplo de salida:

```
input shape = {1000, 20, 20}
x shape = {1000, 400}
z1 shape = {1000, 100}
y1 shape = {1000, 100}
a1 shape = {1000, 100}
z2 shape = {1000, 10}
y2 shape = {1000, 10}
output shape = {1000, 10}
```

---

## Posibles errores

* `g++ no reconocido` → instalar compilador
* error de compilación → verificar archivos `.cpp`
* no ejecuta → revisar ruta del ejecutable

---

## Estructura del proyecto

```
Tensor++
 ┣ Tensor.h
 ┣ Tensor.cpp
 ┣ main.cpp
 ┣ README.md
```

---

## Autor

* Renzo Chen Heng Liang Corrales

---

## Conclusión

Este proyecto demuestra la implementación completa de una librería de tensores desde cero en C++, integrando:

* manejo de memoria
* programación orientada a objetos
* operadores sobrecargados
* polimorfismo
* y una aplicación práctica con una red neuronal
