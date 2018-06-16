"""import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
ust created a child process (%s).' % (os.getpid(), pid))else:
    print('I (%s) j
"""

class f(object):

    def __init__(sel, tmp):
        a = 0
        for i in range(tmp*500):
            a += 1
        #print(a)

#from multiprocessing import 
from multiprocessing import Pool
#pool = multiprocessing.pool
#import multiprocessing.Pool as pool
import time

#print(dir(multiprocessing.multiprocessing))

start = time.time()
for i in range(1000,1100):
    f(i)
print('signal',time.time() - start)

start = time.time()
with Pool(10) as p:
    p.map(f, ([i for i in range(1000,1100)]))
print('multi', time.time() - start)



import tarfile
import os

def ExtractData( data_path):
    for file_name in os.listdir(data_path):
        with tarfile.open(data_path + file_name, 'r:gz') as tar: 
            tar.extractall(data_path)

ExtractData('/media/zhou/0004DD1700005FE8/AI/00/data/test/')