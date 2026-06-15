import struct
import print_ex as prex

if __name__ != '__main__':

    def get_pyarch():
        return struct.calcsize("P") * 8

    def make_ids(numrows):
        res = [*range(numrows)]
        res_str= b = list(map(str, res))
        return res_str

else:
    prex.print_error("Not runnable py file. Use proper main.py instead.")