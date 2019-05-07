from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import timeit
from time import process_time as pt
import simulation
import cpu_data


def simulate_serial():
    print('Iniciando Modo Serial:')
    process_init = pt()  # Contador de processo
    script_init = timeit.default_timer()  # Contador Benchmark

    simulation.main()

    process_end = pt() - process_init
    script_end = timeit.default_timer() - script_init
    print('Tempo de Finalização do Processo (Serial): ', process_end)
    print('Tempo de Finalização do Script (Serial): ', script_end)
    print('--------------------------')


def simulate_threading():
    # Contadores
    print('Iniciando Multithreading:\n')
    process_init = pt()  # Contador de processo
    script_init = timeit.default_timer()  # Contador Benchmark

    with ThreadPoolExecutor(max_workers=None) as executor:
        executor.map(simulation.main())
        # task2 = executor.map(cpu_data.main())

    process_end = pt() - process_init
    script_end = timeit.default_timer() - script_init
    print('Tempo de Finalização do Processo (Threading): ', process_end)
    print('Tempo de Finalização do Script (Threading): ', script_end)
    print('--------------------------')


def simulate_processing():
    # Contadores
    print('Iniciando Multiprocessing:\n')
    print('--------------------------')
    process_init = pt()  # Contador de processo
    script_init = timeit.default_timer()  # Contador Benchmark

    with ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(simulation.main())
        # task2 = executor.map(cpu_data.main())

    process_end = pt() - process_init
    script_end = timeit.default_timer() - script_init
    print('Tempo de Finalização do Processo (Processing): ', process_end)
    print('Tempo de Finalização do Script (Processing): ', script_end)
    print('--------------------------')


def main():
    simulate_threading()
    simulate_processing()
    cpu_data.plot()


if __name__ == '__main__':
    main()
