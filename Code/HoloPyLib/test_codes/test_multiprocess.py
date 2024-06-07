from multiprocessing import Process
from time import sleep


def task(sleep_time_s, nb_iteration):
    for i in range(nb_iteration):
        sleep(sleep_time_s)
        print("sleeping, iteration {}".format(i))


if __name__ == "__main__":

    p1 = Process(target=task, args=(0.1, 100))
    p2 = Process(target=task, args=(0.2, 50))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
