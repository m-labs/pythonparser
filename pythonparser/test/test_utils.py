# coding:utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

unicode = type("")

class BytesOnly(bytes):
    def __new__(cls, s):
        if isinstance(s, unicode):
            s = s.encode()
        return bytes.__new__(BytesOnly, s)

    def __eq__(self, o):
        return isinstance(o, bytes) and bytes.__eq__(self, o)

    def __ne__(self, o):
        return not self == o

class UnicodeOnly(unicode):
    def __eq__(self, o):
        return isinstance(o, unicode) and unicode.__eq__(self, o)

    def __ne__(self, o):
        return not self == o
