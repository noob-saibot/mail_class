import re
st = '[1][2][3][1313][0453]'
print(re.findall('\[\d+\]', st))