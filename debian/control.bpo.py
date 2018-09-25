#!/usr/bin/python3.5
# Backport control.py to python3.5
import re

lines = [x for x in open('debian/control.py', 'r').readlines()]
nlines = []
for (n, line) in enumerate(lines, 1):
    m = re.match('(.*f[\'"]{1}.*[\'"]{1}.*)', line)
    if m:
        print(n, line)
        x = re.sub('(f)([\'"]{1})(.*?)([\'"]{1})', '\\2\\3\\4.format(**locals())', line)
        print('->', x)
        nlines.append(x)
    else:
        nlines.append(line)
with open('debian/control.py', 'w') as f:
    f.writelines(nlines)
