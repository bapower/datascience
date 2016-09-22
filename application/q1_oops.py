import random
import heapq

T = 8
N = 2

def getStream (T):
  stream = []
  for x in range(T):
    stream.append(random.randint(1, 10))
  return stream

def getMax (stream, N):
  return heapq.nlargest(N, stream)

def getLast (stream, N):
  return stream[len(stream)-N:]

stream = getStream(T)
max = getMax(stream, N)
last = getLast(stream, N)

print(stream)
print(max)
print(last)
