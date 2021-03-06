#!/usr/bin/python3
#Small Python CPU Benchmark for Linux CLI by Alexander Ochs - noxmiles.de

import time
import platform

print('Simple Python Benchmark for measureing the CPU speed.')

if "Linux" == platform.system():
  print('Processor:')
  with open('/proc/cpuinfo') as f:
    for line in f:
        # Ignore the blank line separating the information between
        # details about two processing units
        if line.strip():
            if line.rstrip('\n').startswith('model name'):
                model_name = line.rstrip('\n').split(':')[1]
                with open('single_core.txt', 'w') as f:
                    print(model_name, file=f)
                    print(model_name)
                break
else:
  print('Your CPU is only shown automatic on Linux system.')

laeufe = 1000
if laeufe == '': laeufe = 1000
laeufe = int(laeufe)
wiederholungen = 10
if wiederholungen == '': wiederholungen = 10
wiederholungen = int(wiederholungen)

schnitt = 0

with open('single_core.txt', 'a') as f:
  print('testing',file=f)

for a in range(0,wiederholungen):

  start = time.time()

  for i in range(0,laeufe):
    for x in range(1,1000):
      3.141592 * 2**x
    for x in range(1,10000):
      float(x) / 3.141592
    for x in range(1,10000):
      float(3.141592) / x

  ende = time.time()
  dauer = (ende - start)
  dauer = round(dauer, 3)
  schnitt += dauer
  with open('single_core.txt','a') as f:
    print('Time: ' + str(dauer) + 's', file=f)
  print(dauer)

schnitt = round(schnitt / wiederholungen, 3)
with open('single_core.txt','a') as f:
    print('Avarage: ' + str(schnitt) + 's\nDone\n', file=f)
print('Avarage: ' + str(schnitt) + 's')
