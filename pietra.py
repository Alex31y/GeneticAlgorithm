import threading
import time

def mt(i):
    print (i)
    time.sleep(1)

def main():
    for i in range(5):
        threadProcess = threading.Thread(name='simplethread', target=mt, args=[i])
        threadProcess.daemon = True
        threadProcess.start()
main()