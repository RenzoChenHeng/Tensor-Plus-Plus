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

## ⚙️ Funcionalidades implementadas

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

El archivo `main.cpp` incluye:

* Pruebas de creación de tensores
* Operaciones básicas
* Transformaciones
* Ejecución de la red neuronal

---

##  Estructura del proyecto

```
 Tensor++
 ┣ 📜 Tensor.h      # Declaraciones
 ┣ 📜 Tensor.cpp    # Implementación
 ┣ 📜 main.cpp      # Pruebas y ejecución
 ┣ 📜 README.md     # Documentación
```

---

## Consideraciones importantes

* Máximo 3 dimensiones por tensor
* `view` y `unsqueeze` no copian datos
* `concat` sí reserva nueva memoria
* Validación de dimensiones en todas las operaciones
* Manejo seguro de memoria para evitar fugas

---

## Autor(es)

* Renzo Chen Heng Liang Corrales
* (Agregar compañero si aplica)

---

## Conclusión

Este proyecto demuestra la implementación desde cero de una estructura compleja de datos, combinando:

* manejo de memoria
* programación orientada a objetos
* operadores sobrecargados
* polimorfismo
* y simulación de modelos de machine learning

---
