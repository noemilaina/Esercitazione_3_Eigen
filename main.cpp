#include <iostream>
#include "Eigen/Eigen"
#include <cmath>

using namespace Eigen;

VectorXd x_true(2);

double RelativeError(const Eigen::VectorXd& x_computed, const Eigen::VectorXd& x_true) {
	return (x_computed - x_true).norm() / x_true.norm();
}

int main()
{
	x_true << -1.0, -1.0;
	
	MatrixXd A1(2, 2);
	VectorXd b1(2);
	A1 <<  5.547001962252291e-01, -3.770900990025203e-02,
          8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	MatrixXd A2(2, 2);
	VectorXd b2(2);
	A2 << 5.547001962252291e-01, -5.540607316466765e-01,
          8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;
	
	MatrixXd A3(2, 2);
	VectorXd b3(2);
	A3 << 5.547001962252291e-01, -5.547001955851905e-01,
          8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	FullPivLU<MatrixXd> lu1(A1);
	VectorXd x1 = lu1.solve(b1);
	std::cout << "La soluzione del sistema 1 con il metodo PALU:" << x1.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema vale:" << RelativeError(x1, x_true) << std::endl;
	
	HouseholderQR<MatrixXd> qr1(A1);
	VectorXd x1_qr = qr1.solve(b1);
	std::cout << "La soluzione del sistema 1 con il metodo QR è:" << x1_qr.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema 1 è:" << RelativeError(x1_qr, x_true) << std::endl;
	
	FullPivLU<MatrixXd> lu2(A2);
	VectorXd x2 = lu2.solve(b2);
	std::cout << "La soluzione del sistema 2 con il metodo PALU:" << x2.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema vale:" << RelativeError(x2, x_true) << std::endl;
	
	HouseholderQR<MatrixXd> qr2(A2);
	VectorXd x2_qr = qr2.solve(b2);
	std::cout << "La soluzione del sistema 2 con il metodo QR è:" << x2_qr.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema 2 è:" << RelativeError(x2_qr, x_true) << std::endl;
	
	FullPivLU<MatrixXd> lu3(A3);
	VectorXd x3 = lu3.solve(b3);
	std::cout << "La soluzione del sistema 3 con il metodo PALU:" << x3.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema vale:" << RelativeError(x3, x_true) << std::endl;
	
	HouseholderQR<MatrixXd> qr3(A3);
	VectorXd x3_qr = qr3.solve(b3);
	std::cout << "La soluzione del sistema 3 con il metodo QR è:" << x3_qr.transpose() << std::endl;
	std::cout << "L'errore relativo del sistema 3 è:" << RelativeError(x3_qr, x_true) << std::endl;
	
    return 0;
}
