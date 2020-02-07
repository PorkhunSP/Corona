import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datetime import timedelta

adres = 'https://tirinox.ru/wp-content/uploads/2020/01/corona.csv'

df = pd.read_csv(adres, parse_dates=['day'])

print(df.head())

df.plot(kind='bar', x='day', y=['infected', 'dead'])
plt.show()


def fit_func_exp(x, a, b, c):
    '''Selected function'''
    return a * np.exp(c * x) + b


infected = df['infected']
days = range(len(df['day']))


p0 = np.ones(3)

(a, b, c), garb = curve_fit(fit_func_exp, days, infected, p0=p0)

max_day = 60
x_days = np.linspace(0, max_day - 1, max_day)
y_infected = fit_func_exp(x_days, a, b, c)

plt.xlabel('Дни')
plt.ylabel('Кол-во Больных')
plt.yscale('Log')

plt.scatter(days, infected, marker='D', label='Реальные')
plt.plot(x_days[:30], y_infected[:30], 'r', label='Предсказание')
plt.legend()
plt.show()
E_population = 7_530_000_000

doom_index = np.argmax(y_infected >= E_population)
doom_day = x_days[doom_index]

day0 = df['day'][0]
doom_date = day0 + timedelta(days=int(doom_day))
print(f'Doom date: {doom_date:%d.%m.%Y}')
