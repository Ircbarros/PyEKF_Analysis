import numpy as np
import timeit
import time
import plotly.graph_objs as go
import plotly.offline
from plotly import tools
import psutil as psu

# Contador
script_init = timeit.default_timer()  # Contador Benchmark

# Alocação de Memória para Variáveis Utilizadas no Plot
cpu_dicc = {'cpu1': [], 'cpu2': [], 'cpu3': [], 'cpu4': []}
time_dicc = {'time': []}
memory_dicc = {'memory': []}


# Scheduler para Armazenar valor de Clock, Memória, Frequência e Load
def computer_data():
    load = np.array(psu.cpu_percent(interval=0.1, percpu=True))
    load = load.tolist()
    # Append das Listas nos Dicionários para Elaboração do Plot
    cpu_dicc['cpu1'] += [load[0]]
    cpu_dicc['cpu2'] += [load[1]]
    cpu_dicc['cpu3'] += [load[2]]
    cpu_dicc['cpu4'] += [load[3]]

    # Append do tempo para Elaboração do Plot
    time_list = [(timeit.default_timer())-script_init]
    time_dicc['time'] += [time_list[0]]
    # Append da Memória para Elaboração do Plot
    memory_percent = [psu.virtual_memory()[2]]
    memory_dicc['memory'] += [memory_percent[0]]


def plot():
    trace0 = go.Scatter(
            y=cpu_dicc['cpu1'],
            x=time_dicc['time'],
            name='CPU 01',
            line=dict(
                      color=('rgb(74, 140, 121)'),
                      width=4,)
    )
    trace1 = go.Scatter(
        y=cpu_dicc['cpu2'],
        x=time_dicc['time'],
        name='CPU 02',
        line=dict(
                    color=('rgb(72, 191, 147)'),
                    width=4,)
    )
    trace2 = go.Scatter(
        y=cpu_dicc['cpu3'],
        x=time_dicc['time'],
        name='CPU 03',
        line=dict(
                    color=('rgb(216, 111, 73)'),
                    width=4,)
    )
    trace3 = go.Scatter(
        y=cpu_dicc['cpu4'],
        x=time_dicc['time'],
        name='CPU 04',
        line=dict(
                    color=('rgb(165, 80, 72)'),
                    width=4,)
    )
    trace4 = go.Bar(
        y=memory_dicc['memory'],
        x=time_dicc['time'],
        text=memory_dicc['memory'],
        name='Memory Usage',
        textposition='auto',
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
            ),
        opacity=0.6
    )
    # Plotagem do Primeiro Gráfico (Múltiplos CPU's)
    fig = tools.make_subplots(rows=2, cols=2,
                              subplot_titles=('CPU 1', 'CPU 2',
                                              'CPU 3', 'CPU 4'))
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 2, 2)
    fig['layout'].update(height=800, width=1000,
                         title='CPU LOAD with Anaconda Full Distribution' +
                               ' for Python 3 (with MultiThreading (' +
                               ' Loop and Plot) & Scheduling)',
                         xaxis=dict(title='Time (s)'),
                         yaxis=dict(title='Load (%)'),
                         )
    plotly.offline.plot(fig, image='jpeg', image_filename='CPU_sub_conda')
    # Plotagem do Segundo Gráfico (CPU's em Conjunto e Memória)
    time.sleep(1)  # Sleep para Efetuação da Impressão do Primeiro Plot
    fig = tools.make_subplots(rows=2, cols=1)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 2, 1)
    fig['layout'].update(height=800, width=1000,
                         title='CPU Load with Anaconda Full Distribution' +
                               ' for Python 3 (with MultiThreading' +
                               ' & Scheduling)',
                         xaxis=dict(title='Time (s)'),
                         yaxis=dict(title='Load (%)'),
                         )
    plotly.offline.plot(fig, image='jpeg', image_filename='CPU&Mem_conda')


def main():
    computer_data()
