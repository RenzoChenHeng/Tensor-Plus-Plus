#include "Tensor.h"

#include <iostream>

int main() {
    try {
        std::cout << "===== PRUEBAS BASICAS =====\n";

        Tensor A = Tensor::zeros({2, 3});
        Tensor B = Tensor::ones({2, 3});
        Tensor C = Tensor::random({2, 2}, 0.0, 1.0);
        Tensor D = Tensor::arange(0, 6);

        A.print("A = zeros({2,3})");
        B.print("B = ones({2,3})");
        C.print("C = random({2,2}, 0.0, 1.0)");
        D.print("D = arange(0, 6)");

        std::cout << "\n===== VIEW =====\n";
        Tensor E = D.view({2, 3});
        E.print("E = D.view({2,3})");

        std::cout << "\n===== UNSQUEEZE =====\n";
        Tensor F = D.unsqueeze(0); // {1,6}
        Tensor G = D.unsqueeze(1); // {6,1}
        F.print("F = D.unsqueeze(0)");
        G.print("G = D.unsqueeze(1)");

        std::cout << "\n===== OPERADORES =====\n";
        Tensor H = A + B;
        Tensor I = B - A;
        Tensor J = B * 2.0;
        Tensor K = B * B;

        H.print("H = A + B");
        I.print("I = B - A");
        J.print("J = B * 2.0");
        K.print("K = B * B");

        std::cout << "\n===== CONCAT =====\n";
        Tensor L = Tensor::concat({A, B}, 0);
        L.print("L = concat({A, B}, 0)");

        std::cout << "\n===== DOT =====\n";
        Tensor v1 = Tensor::arange(1, 4); // [1,2,3]
        Tensor v2 = Tensor::ones({3});    // [1,1,1]
        Tensor dot_result = dot(v1, v2);
        dot_result.print("dot(v1, v2)");

        std::cout << "\n===== MATMUL =====\n";
        Tensor M({2, 3}, {
            1, 2, 3,
            4, 5, 6
        });

        Tensor N({3, 2}, {
            7, 8,
            9, 10,
            11, 12
        });

        Tensor O = matmul(M, N);
        O.print("O = matmul(M, N)");

        std::cout << "\n===== TRANSFORMACIONES =====\n";
        Tensor P = Tensor::arange(-5, 5).view({2, 5});
        ReLU relu;
        Sigmoid sigmoid;

        Tensor Q = P.apply(relu);
        Tensor R = P.apply(sigmoid);

        P.print("P");
        Q.print("Q = P.apply(ReLU)");
        R.print("R = P.apply(Sigmoid)");

        std::cout << "\n===== RED NEURONAL PEDIDA =====\n";

        // Simulación de red neuronal simple.
        // Flujo: input -> matmul -> bias -> ReLU -> matmul -> Sigmoid.
        // Procesa datos y genera salida final.
        // 1) Entrada: 1000 x 20 x 20
        Tensor input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
        std::cout << "input shape = " << input.shape_string() << "\n";

        // 2) view -> 1000 x 400
        Tensor x = input.view({1000, 400});
        std::cout << "x shape = " << x.shape_string() << "\n";

        // 3) Multiplicar por W1: 400 x 100
        Tensor W1 = Tensor::random({400, 100}, -0.5, 0.5);
        Tensor z1 = matmul(x, W1);
        std::cout << "z1 shape = " << z1.shape_string() << "\n";

        // 4) Sumar bias b1: 1 x 100
        Tensor b1 = Tensor::random({1, 100}, -0.1, 0.1);
        Tensor y1 = z1 + b1; // broadcasting
        std::cout << "y1 shape = " << y1.shape_string() << "\n";

        // 5) ReLU
        Tensor a1 = y1.apply(relu);
        std::cout << "a1 shape = " << a1.shape_string() << "\n";

        // 6) Multiplicar por W2: 100 x 10
        Tensor W2 = Tensor::random({100, 10}, -0.5, 0.5);
        Tensor z2 = matmul(a1, W2);
        std::cout << "z2 shape = " << z2.shape_string() << "\n";

        // 7) Sumar bias b2: 1 x 10
        Tensor b2 = Tensor::random({1, 10}, -0.1, 0.1);
        Tensor y2 = z2 + b2; // broadcasting
        std::cout << "y2 shape = " << y2.shape_string() << "\n";

        // 8) Sigmoid
        Tensor output = y2.apply(sigmoid);
        std::cout << "output shape = " << output.shape_string() << "\n";

        std::cout << "\nPrimeros 20 valores del output:\n";
        for (size_t i = 0; i < 20 && i < output.size(); ++i) {
            std::cout << output[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "\nTodo ejecuto correctamente.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}