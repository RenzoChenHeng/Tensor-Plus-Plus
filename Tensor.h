#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

class TensorTransform;

// Clase Tensor que representa arreglos de hasta 3 dimensiones.
// Maneja memoria dinámica y operaciones tipo NumPy.
// Soporta creación, operaciones, transformaciones y broadcasting.
// Base para simulación de red neuronal.
class Tensor {
public:
    std::vector<size_t> shape_;
    size_t total_size_;
    double* data_;
    size_t* ref_count_;

    static size_t compute_total_size(const std::vector<size_t>& shape);
    static void validate_shape(const std::vector<size_t>& shape);
    static std::vector<size_t> normalize_shape(const std::vector<size_t>& shape, size_t target_dims = 3);
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

    void allocate_and_fill(const std::vector<size_t>& shape, const std::vector<double>& values);
    void release();
    void retain();

public:
    // Constructor por defecto
    Tensor();
    // Inicializa el tensor con shape y valores.
    // Valida que values coincida con el tamaño total.
    // Usa memoria dinámica contigua.
    // Constructor principal
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values);

    // Implementación de la regla de 5.
    // Permite copiar y mover objetos correctamente.
    // Evita fugas de memoria.
    // Regla de 5
    Tensor(const Tensor& other);                 // Copia profunda
    Tensor(Tensor&& other) noexcept;             // Movimiento
    Tensor& operator=(const Tensor& other);      // Asignación copia
    Tensor& operator=(Tensor&& other) noexcept;  // Asignación movimiento
    ~Tensor();

    // Métodos de creación de tensores.
    // Permiten inicializar con ceros, unos, aleatorios o secuencias.
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
    static Tensor random(const std::vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    // Información
    const std::vector<size_t>& shape() const;
    size_t ndim() const;
    size_t size() const;
    bool empty() const;

    // Acceso plano a datos
    double& operator[](size_t index);
    const double& operator[](size_t index) const;

    // Reinterpreta la forma del tensor sin copiar datos.
    // Mantiene el mismo bloque de memoria.
    // Mejora el rendimiento.
    // Utilidades
    Tensor view(const std::vector<size_t>& new_shape) const;
    // Inserta una dimensión de tamaño 1.
    // No modifica los datos internos.
    // Útil para operaciones matriciales.
    Tensor unsqueeze(size_t dim) const;
    Tensor apply(const TensorTransform& transform) const;

    // Une varios tensores en una dimensión específica.
    // Verifica compatibilidad de dimensiones.
    // Crea nueva memoria para el resultado.
    static Tensor concat(const std::vector<Tensor>& tensors, size_t dim);

    std::string shape_string() const;
    void print(const std::string& name = "") const;

    // Operaciones elemento a elemento.
    // Soporta broadcasting básico.
    // No modifica los tensores originales.
    friend Tensor operator+(const Tensor& a, const Tensor& b);
    friend Tensor operator-(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, const Tensor& b);   // element-wise
    friend Tensor operator*(const Tensor& a, double scalar);
    friend Tensor operator*(double scalar, const Tensor& a);

    // Calcula el producto punto entre dos tensores.
    // Ambos deben tener igual tamaño.
    // Retorna un tensor escalar.
    friend Tensor dot(const Tensor& a, const Tensor& b);
    // Multiplicación matricial entre tensores 2D.
    // Requiere dimensiones compatibles.
    // Implementa producto fila por columna.
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);
};

// Interfaz abstracta
class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

// ReLU
// Transformaciones aplicadas a cada elemento.
// ReLU elimina negativos.
// Sigmoid comprime valores entre 0 y 1.
class ReLU : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

// Sigmoid
class Sigmoid : public TensorTransform {
public:
    Tensor apply(const Tensor& t) const override;
};

#endif