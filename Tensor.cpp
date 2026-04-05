//
// Created by liang on 5/04/2026.
//
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

// =========================
// Helpers privados
// =========================

size_t Tensor::compute_total_size(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("El shape no puede estar vacio.");
    }

    size_t total = 1;
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Las dimensiones deben ser mayores que 0.");
        }
        total *= dim;
    }
    return total;
}

void Tensor::validate_shape(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("El shape no puede estar vacio.");
    }
    if (shape.size() > 3) {
        throw std::invalid_argument("Tensor solo soporta hasta 3 dimensiones.");
    }
    for (size_t dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("Las dimensiones deben ser mayores que 0.");
        }
    }
}

std::vector<size_t> Tensor::normalize_shape(const std::vector<size_t>& shape, size_t target_dims) {
    if (shape.size() > target_dims) {
        throw std::invalid_argument("No se puede normalizar un shape con mas dimensiones que el objetivo.");
    }

    std::vector<size_t> normalized(target_dims, 1);
    size_t offset = target_dims - shape.size();

    for (size_t i = 0; i < shape.size(); ++i) {
        normalized[offset + i] = shape[i];
    }
    return normalized;
}

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    if (shape.empty()) return strides;

    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

void Tensor::allocate_and_fill(const std::vector<size_t>& shape, const std::vector<double>& values) {
    validate_shape(shape);

    size_t total = compute_total_size(shape);
    if (values.size() != total) {
        throw std::invalid_argument("El numero de values no coincide con el producto del shape.");
    }

    shape_ = shape;
    total_size_ = total;
    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = values[i];
    }
}

void Tensor::release() {
    if (ref_count_ != nullptr) {
        (*ref_count_)--;
        if (*ref_count_ == 0) {
            delete[] data_;
            delete ref_count_;
        }
    }

    data_ = nullptr;
    ref_count_ = nullptr;
    total_size_ = 0;
    shape_.clear();
}

void Tensor::retain() {
    if (ref_count_ != nullptr) {
        (*ref_count_)++;
    }
}

// =========================
// Constructores / destructor
// =========================

Tensor::Tensor()
    : shape_(), total_size_(0), data_(nullptr), ref_count_(nullptr) {}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<double>& values)
    : shape_(), total_size_(0), data_(nullptr), ref_count_(nullptr) {
    allocate_and_fill(shape, values);
}

// Copia profunda
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), total_size_(other.total_size_), data_(nullptr), ref_count_(nullptr) {
    if (other.data_ == nullptr || other.total_size_ == 0) {
        return;
    }

    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = other.data_[i];
    }
}

// Movimiento
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      total_size_(other.total_size_),
      data_(other.data_),
      ref_count_(other.ref_count_) {
    other.total_size_ = 0;
    other.data_ = nullptr;
    other.ref_count_ = nullptr;
    other.shape_.clear();
}

// Asignación copia profunda
Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    release();

    shape_ = other.shape_;
    total_size_ = other.total_size_;

    if (other.data_ == nullptr || other.total_size_ == 0) {
        data_ = nullptr;
        ref_count_ = nullptr;
        return *this;
    }

    data_ = new double[total_size_];
    ref_count_ = new size_t(1);

    for (size_t i = 0; i < total_size_; ++i) {
        data_[i] = other.data_[i];
    }

    return *this;
}

// Asignación movimiento
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    shape_ = std::move(other.shape_);
    total_size_ = other.total_size_;
    data_ = other.data_;
    ref_count_ = other.ref_count_;

    other.total_size_ = 0;
    other.data_ = nullptr;
    other.ref_count_ = nullptr;
    other.shape_.clear();

    return *this;
}

Tensor::~Tensor() {
    release();
}

// =========================
// Métodos estáticos
// =========================

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    validate_shape(shape);
    size_t total = compute_total_size(shape);
    std::vector<double> values(total, 0.0);
    return Tensor(shape, values);
}

Tensor Tensor::ones(const std::vector<size_t>& shape) {
    validate_shape(shape);
    size_t total = compute_total_size(shape);
    std::vector<double> values(total, 1.0);
    return Tensor(shape, values);
}

Tensor Tensor::random(const std::vector<size_t>& shape, double min, double max) {
    validate_shape(shape);

    if (min >= max) {
        throw std::invalid_argument("En random, min debe ser menor que max.");
    }

    size_t total = compute_total_size(shape);
    std::vector<double> values(total);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    for (size_t i = 0; i < total; ++i) {
        values[i] = dist(gen);
    }

    return Tensor(shape, values);
}

Tensor Tensor::arange(int start, int end) {
    if (start >= end) {
        throw std::invalid_argument("En arange, start debe ser menor que end.");
    }

    std::vector<double> values;
    for (int i = start; i < end; ++i) {
        values.push_back(static_cast<double>(i));
    }

    return Tensor({values.size()}, values);
}

// =========================
// Información
// =========================

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::ndim() const {
    return shape_.size();
}

size_t Tensor::size() const {
    return total_size_;
}

bool Tensor::empty() const {
    return total_size_ == 0 || data_ == nullptr;
}

// =========================
// Acceso
// =========================

double& Tensor::operator[](size_t index) {
    if (index >= total_size_) {
        throw std::out_of_range("Indice fuera de rango.");
    }
    return data_[index];
}

const double& Tensor::operator[](size_t index) const {
    if (index >= total_size_) {
        throw std::out_of_range("Indice fuera de rango.");
    }
    return data_[index];
}

// =========================
// Utilidades
// =========================

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    validate_shape(new_shape);

    if (compute_total_size(new_shape) != total_size_) {
        throw std::invalid_argument("view: el nuevo shape debe conservar el mismo numero de elementos.");
    }

    Tensor result;
    result.shape_ = new_shape;
    result.total_size_ = total_size_;
    result.data_ = data_;
    result.ref_count_ = ref_count_;
    result.retain(); // comparte datos sin copiar

    return result;
}

Tensor Tensor::unsqueeze(size_t dim) const {
    if (shape_.empty()) {
        throw std::runtime_error("unsqueeze no puede aplicarse a un tensor vacio.");
    }

    if (shape_.size() >= 3) {
        throw std::invalid_argument("unsqueeze excede el maximo de 3 dimensiones.");
    }

    if (dim > shape_.size()) {
        throw std::out_of_range("unsqueeze: la posicion es invalida.");
    }

    std::vector<size_t> new_shape = shape_;
    new_shape.insert(new_shape.begin() + static_cast<long>(dim), 1);

    Tensor result;
    result.shape_ = new_shape;
    result.total_size_ = total_size_;
    result.data_ = data_;
    result.ref_count_ = ref_count_;
    result.retain(); // comparte datos sin copiar

    return result;
}

Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

Tensor Tensor::concat(const std::vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("concat: la lista de tensores no puede estar vacia.");
    }

    const std::vector<size_t>& base_shape = tensors[0].shape_;
    if (dim >= base_shape.size()) {
        throw std::out_of_range("concat: dimension invalida.");
    }

    for (const auto& t : tensors) {
        if (t.shape_.size() != base_shape.size()) {
            throw std::invalid_argument("concat: todos los tensores deben tener el mismo numero de dimensiones.");
        }

        for (size_t i = 0; i < base_shape.size(); ++i) {
            if (i != dim && t.shape_[i] != base_shape[i]) {
                throw std::invalid_argument("concat: dimensiones incompatibles.");
            }
        }
    }

    std::vector<size_t> new_shape = base_shape;
    new_shape[dim] = 0;
    for (const auto& t : tensors) {
        new_shape[dim] += t.shape_[dim];
    }

    size_t new_total = compute_total_size(new_shape);
    std::vector<double> values;
    values.reserve(new_total);

    // Concatenación para tensores contiguos alineados por bloques
    if (base_shape.size() == 1) {
        for (const auto& t : tensors) {
            for (size_t i = 0; i < t.total_size_; ++i) {
                values.push_back(t.data_[i]);
            }
        }
    } else if (base_shape.size() == 2) {
        size_t rows = new_shape[0];
        size_t cols = new_shape[1];

        (void)rows;
        (void)cols;

        if (dim == 0) {
            for (const auto& t : tensors) {
                for (size_t i = 0; i < t.total_size_; ++i) {
                    values.push_back(t.data_[i]);
                }
            }
        } else {
            for (size_t r = 0; r < base_shape[0]; ++r) {
                for (const auto& t : tensors) {
                    size_t row_offset = r * t.shape_[1];
                    for (size_t c = 0; c < t.shape_[1]; ++c) {
                        values.push_back(t.data_[row_offset + c]);
                    }
                }
            }
        }
    } else if (base_shape.size() == 3) {
        size_t A = new_shape[0];
        size_t B = new_shape[1];
        size_t C = new_shape[2];
        (void)A; (void)B; (void)C;

        if (dim == 0) {
            for (const auto& t : tensors) {
                for (size_t i = 0; i < t.total_size_; ++i) {
                    values.push_back(t.data_[i]);
                }
            }
        } else if (dim == 1) {
            for (size_t i = 0; i < base_shape[0]; ++i) {
                for (const auto& t : tensors) {
                    size_t block = t.shape_[1] * t.shape_[2];
                    size_t start = i * block;
                    for (size_t j = 0; j < block; ++j) {
                        values.push_back(t.data_[start + j]);
                    }
                }
            }
        } else {
            for (size_t i = 0; i < base_shape[0]; ++i) {
                for (size_t j = 0; j < base_shape[1]; ++j) {
                    for (const auto& t : tensors) {
                        size_t start = i * (t.shape_[1] * t.shape_[2]) + j * t.shape_[2];
                        for (size_t k = 0; k < t.shape_[2]; ++k) {
                            values.push_back(t.data_[start + k]);
                        }
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("concat: numero de dimensiones no soportado.");
    }

    return Tensor(new_shape, values);
}

std::string Tensor::shape_string() const {
    std::ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}

void Tensor::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " ";
    }
    std::cout << "shape=" << shape_string() << "\n";
    std::cout << *this << "\n";
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "[";
    for (size_t i = 0; i < t.total_size_; ++i) {
        os << std::fixed << std::setprecision(4) << t.data_[i];
        if (i + 1 < t.total_size_) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// =========================
// Broadcasting helper
// =========================

static std::vector<size_t> broadcast_shape_of(const std::vector<size_t>& a_shape, const std::vector<size_t>& b_shape) {
    std::vector<size_t> a = Tensor::normalize_shape(a_shape, 3);
    std::vector<size_t> b = Tensor::normalize_shape(b_shape, 3);
    std::vector<size_t> result(3);

    for (size_t i = 0; i < 3; ++i) {
        if (a[i] == b[i]) {
            result[i] = a[i];
        } else if (a[i] == 1) {
            result[i] = b[i];
        } else if (b[i] == 1) {
            result[i] = a[i];
        } else {
            throw std::invalid_argument("Shapes incompatibles para broadcasting.");
        }
    }

    // quitar los 1 iniciales innecesarios manteniendo al menos 1 dimensión
    while (result.size() > 1 && result.front() == 1) {
        result.erase(result.begin());
    }

    return result;
}

static size_t flat_index_from_broadcast(
    size_t result_flat_idx,
    const std::vector<size_t>& result_shape,
    const std::vector<size_t>& operand_shape
) {
    std::vector<size_t> result3 = Tensor::normalize_shape(result_shape, 3);
    std::vector<size_t> op3 = Tensor::normalize_shape(operand_shape, 3);

    std::vector<size_t> result_strides = Tensor::compute_strides(result3);
    std::vector<size_t> op_strides = Tensor::compute_strides(op3);

    size_t rem = result_flat_idx;
    std::vector<size_t> idx3(3, 0);

    for (size_t i = 0; i < 3; ++i) {
        idx3[i] = rem / result_strides[i];
        rem %= result_strides[i];
    }

    size_t op_index = 0;
    for (size_t i = 0; i < 3; ++i) {
        size_t coord = (op3[i] == 1 ? 0 : idx3[i]);
        op_index += coord * op_strides[i];
    }

    return op_index;
}

static Tensor elementwise_binary_op(
    const Tensor& a,
    const Tensor& b,
    char op
) {
    std::vector<size_t> out_shape = broadcast_shape_of(a.shape_, b.shape_);
    size_t out_total = Tensor::compute_total_size(out_shape);

    std::vector<double> out_values(out_total);

    for (size_t i = 0; i < out_total; ++i) {
        size_t ia = flat_index_from_broadcast(i, out_shape, a.shape_);
        size_t ib = flat_index_from_broadcast(i, out_shape, b.shape_);

        if (op == '+') {
            out_values[i] = a.data_[ia] + b.data_[ib];
        } else if (op == '-') {
            out_values[i] = a.data_[ia] - b.data_[ib];
        } else if (op == '*') {
            out_values[i] = a.data_[ia] * b.data_[ib];
        } else {
            throw std::runtime_error("Operacion no soportada.");
        }
    }

    return Tensor(out_shape, out_values);
}

// =========================
// Operadores
// =========================

Tensor operator+(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b, '+');
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b, '-');
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return elementwise_binary_op(a, b, '*');
}

Tensor operator*(const Tensor& a, double scalar) {
    std::vector<double> out(a.total_size_);
    for (size_t i = 0; i < a.total_size_; ++i) {
        out[i] = a.data_[i] * scalar;
    }
    return Tensor(a.shape_, out);
}

Tensor operator*(double scalar, const Tensor& a) {
    return a * scalar;
}

// =========================
// Funciones amigas
// =========================

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.total_size_ != b.total_size_) {
        throw std::invalid_argument("dot: ambos tensores deben tener la misma cantidad de elementos.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.total_size_; ++i) {
        sum += a.data_[i] * b.data_[i];
    }

    return Tensor({1}, {sum});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape_.size() != 2 || b.shape_.size() != 2) {
        throw std::invalid_argument("matmul: ambos tensores deben ser 2D.");
    }

    size_t m = a.shape_[0];
    size_t n = a.shape_[1];
    size_t n2 = b.shape_[0];
    size_t p = b.shape_[1];

    if (n != n2) {
        throw std::invalid_argument("matmul: dimensiones incompatibles.");
    }

    std::vector<double> out(m * p, 0.0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += a.data_[i * n + k] * b.data_[k * p + j];
            }
            out[i * p + j] = sum;
        }
    }

    return Tensor({m, p}, out);
}

// =========================
// Transformaciones
// =========================

Tensor ReLU::apply(const Tensor& t) const {
    std::vector<double> out(t.size());

    for (size_t i = 0; i < t.size(); ++i) {
        out[i] = std::max(0.0, t[i]);
    }

    return Tensor(t.shape(), out);
}

Tensor Sigmoid::apply(const Tensor& t) const {
    std::vector<double> out(t.size());

    for (size_t i = 0; i < t.size(); ++i) {
        out[i] = 1.0 / (1.0 + std::exp(-t[i]));
    }

    return Tensor(t.shape(), out);
}