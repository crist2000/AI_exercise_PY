import struct
import print_ex as prex

if __name__ != '__main__':

    def get_pyarch():
        return struct.calcsize("P") * 8
else:
    prex.print_error("Not runnable py file. Use proper main.py instead.")