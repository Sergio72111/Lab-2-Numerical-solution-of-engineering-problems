import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
import math


# Объявление функции y(x) с использованием библиотеки numpy
def y(x):
    return (np.log(x + 1) ** 2 / x) * np.exp(-x)


# Границы интегрирования
c = 0.1
d = 1
e1 = 10 ** (-3)
e2 = 10 ** (-6)

# Создание массива значений x
x_values = np.linspace(c, d, 100)

# Построение графика функции y(x)
plt.plot(x_values, y(x_values), color='purple')
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid()
plt.xlim(c, d)
plt.show()

# Вычисляем определенный интеграл
J, error = quad(y, c, d)
print(f"J = {J:.10f}")

x = sp.Symbol('x')
y_sym = (sp.log(x + 1) ** 2 / x) * sp.exp(-x)

# Вычисляем неопределенный интеграл
integral = sp.integrate(y_sym, x)
print(f"Неопределённый интеграл y(x) = {integral} + C")

# Вычисляем 4-ю производную
fourth_derivative = sp.diff(y_sym, x, 4)
fourth_derivative_func = sp.lambdify(x, fourth_derivative, 'numpy')

# Вычисление максимума четвёртой производной
x_test = np.linspace(c, d, 100)
y_test = fourth_derivative_func(x_test)
M4 = np.max(np.abs(y_test))
print(f"Максимальное значение 4-й производной (M4): {M4:.10f}")

# Построение графика четвёртой производной
plt.plot(x_test, y_test, color='blue')
plt.title("4-я производная функции y(x)")
plt.xlabel("x")
plt.ylabel("y''''(x)")
plt.grid()
plt.xlim(c, d)
plt.show()


# Функция для вычисления интеграла методом Симпсона
def simpson_integral(f, a, b, n):
    if n % 2 == 1:
        n += 1  # Симпсон требует четного n
    h = (b - a) / n
    x_odd = np.array([a + (2 * i + 1) * h for i in range(n // 2)])
    x_even = np.array([a + (2 * j) * h for j in range(1, n // 2)])

    sum_odd = np.sum(y(x_odd))
    sum_even = np.sum(y(x_even))

    return (h / 3) * (y(a) + 4 * sum_odd + 2 * sum_even + y(b))


n1 = math.ceil(((M4 * (d - c) ** 5) / (180 * e1)) ** (1 / 4))
I1 = simpson_integral(y, c, d, n1)
print(f"Значение интеграла I1: {I1:.10f}")


n2 = math.ceil(((M4 * (d - c) ** 5) / (180 * e2)) ** (1 / 4))
I2 = simpson_integral(y, c, d, n2)
print(f"Приближенное значение интеграла I2: {I2:.10f}")

# Вычисление абсолютных и относительных погрешностей
abs_delta_1 = abs(J - I1)
relative_delta_1 = abs_delta_1 / abs(J) * 100
print(f"Абсолютная погрешность Δ1: {abs_delta_1:.10f}")
print(f"Относительная погрешность Δ1: {relative_delta_1:.10f}%")

abs_delta_2 = abs(J - I2)
relative_delta_2 = abs_delta_2 / abs(J) * 100
print(f"Абсолютная погрешность Δ2: {abs_delta_2:.10f}")
print(f"Относительная погрешность Δ2: {relative_delta_2:.10f}%")
