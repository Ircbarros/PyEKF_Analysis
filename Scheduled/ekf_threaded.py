import numpy as np
import math
import matplotlib.pyplot as plt
import timeit
import time
from time import process_time as pt
import schedule
import plotly.graph_objs as go
import plotly.offline
from plotly import tools
import psutil as psu
import threading


# Contadores
process_init = pt()  # Contador de processo
script_init = timeit.default_timer()  # Contador Benchmark


# Alocação de Memória para Variáveis Utilizadas no Plot
cpu_dicc = {'cpu1': [], 'cpu2': [], 'cpu3': [], 'cpu4': []}
time_dicc = {'time': []}
memory_dicc = {'memory': []}


# Estimation parameter of EKF
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0])**2  # predict state covariance
R = np.diag([1.0, 1.0])**2  # Observation x,y position covariance

#  Simulation parameter
Qsim = np.diag([1.0, np.deg2rad(30.0)])**2
Rsim = np.diag([0.5, 0.5])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


def observation(xTrue, xd, u):

    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    zx = xTrue[0, 0] + np.random.randn() * Rsim[0, 0]
    zy = xTrue[1, 0] + np.random.randn() * Rsim[1, 1]
    z = np.array([[zx, zy]]).T

    # add noise to input
    ud1 = u[0, 0] + np.random.randn() * Qsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Qsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F@x + B@u

    return x


def observation_model(x):
    #  Observation Model
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H@x

    return z


def jacobF(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacobH(x):
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):

    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacobF(xPred, u)
    PPred = jF@PEst@jF.T + Q

    #  Update
    jH = jacobH(xPred)
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH@PPred@jH.T + R
    K = PPred@jH.T@np.linalg.inv(S)
    xEst = xPred + K@y
    PEst = (np.eye(len(xEst)) - K@jH)@PPred

    return xEst, PEst


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
    R = np.array([[math.cos(angle), math.sin(angle)],
                  [-math.sin(angle), math.cos(angle)]])
    fx = R@(np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


# Scheduler para Armazenar valor de Clock, Memória, Frequência e Load
def computer_data():
    print('DATA assigned to thread:' +
          '{}'.format(threading.current_thread().name))
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


# Definição do Thread para elaboração da armazenagem de dados em paralelo
def run_threaded(computer_data):
    job_thread = threading.Thread(target=computer_data, args=(1,), daemon=True)
    job_thread.start()


# 0.5 seg para armazenagem de dados
schedule.every(0.5).seconds.do(run_threaded, computer_data)


def ekf_analysis():
    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    # Contadores para o Loop
    process_loop_init = pt()  # Contador de processo
    script_loop_init = timeit.default_timer()  # Contador Benchmark

    while SIM_TIME >= time:
        print('SIMULATION assigned to thread:' +
              '{}'.format(threading.current_thread().name))
        time += DT
        schedule.run_pending()
        u = calc_input()
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

    # Finalização dos Contadores Loop
    process_loop = pt() - process_loop_init
    script_loop = timeit.default_timer() - script_loop_init
    print("--------------------------------------------------")
    print("Tempo do Processo Loop: ", process_loop)
    print("Tempo do Script: Loop: ", script_loop, "\n")
    plot()


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
                               ' for Python 3 (with Threading & Scheduling)',
                         xaxis=dict(title='Time (s)'),
                         yaxis=dict(title='Load (%)'),
                         )
    plotly.offline.plot(fig, image='jpeg', image_filename='CPU_sub_conda')
    # Plotagem do Segundo Gráfico (CPU's em Conjunto e Memória)
    time.sleep(5)  # Sleep para Efetuação da Impressão do Primeiro Plot
    fig = tools.make_subplots(rows=2, cols=1)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 1)
    fig.append_trace(trace3, 1, 1)
    fig.append_trace(trace4, 2, 1)
    fig['layout'].update(height=800, width=1000,
                         title='CPU Load with Anaconda Full Distribution' +
                               ' for Python (with Threading & Scheduling)',
                         xaxis=dict(title='Time (s)'),
                         yaxis=dict(title='Load (%)'),
                         )
    plotly.offline.plot(fig, image='jpeg', image_filename='CPU&Mem_conda')


def main():
    print(__file__ + " start!!")
    ekf_analysis()
    process_end = pt() - process_init
    script_end = timeit.default_timer() - script_init
    print("----------------FINAL DA SIMULAÇÃO----------------")
    print("Tempo do Processo Total: ", process_end)
    print("Tempo do Script: Main: ", script_end, "\n")
    print("--------------------------------------------------")


if __name__ == '__main__':
    main()
