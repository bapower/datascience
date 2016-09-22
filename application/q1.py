import random
import statistics

T = 10;
S = 0;
current = 0;

moves = {
    0 : [6, 4],
    1 : [8, 6],
    2 : [7, 9],
    3 : [4, 8],
    4 : [3, 9, 0],
    5 : [],
    6 : [1, 7, 0],
    7 : [2, 6],
    8 : [1, 3],
    9 : [4, 2]
}

def move (moves, S, current) :
    current = random.choice(moves[current])
    S += current
    return (S, current)

def play (moves, S, current, T) :
    for x in range(T+1):
        S, current = move(moves, S, current)
    return S

def monte_carlo(mod, S) :
  mods=[]
  for x in range(1000) :
    S = play(moves, S, current, T)
    mods.append(S%mod)
  return mods


S = play(moves, S, current, T)

mods = monte_carlo(T, S)
sd = statistics.stdev(mods)

print("S:")
print(S)

#The sum after moving 10 moves is a random variable. The sum modulo 10 is the remainder after dividing by 10, so it could be any number between 0 and 9.

print("quantity of S mod 10:")
print(S%10)

#The standard deviation of a monte carlo simulation of 100 times with T=10
print ("standard deviation of S mod 10:")
print(sd)

T = 1024
S = play(moves, S, current, T)

mods = monte_carlo(T, S)  
sd = statistics.stdev(mods)

print("S:")
print(S)

##The sum after moving 10 moves is a random variable. The sum modulo 1024 is the remainder after dividing by 1024, so it could be any number between 0 and 1023.

print("quantity of S mod 1024:")
print(S%1024)

#The standard deviation of a monte carlo simulation of 100 times with T=1024
print ("standard deviation of S mod 1024:")
print(sd)

def getPossibilities (T, sums= [{'sum' : 6, 'lastNum' : 6},{'sum' : 4, 'lastNum' : 4}]) :
    for x in range(T+1) :
      for i,sumDat in enumerate(sums[:]) :
        n = sumDat['lastNum']
        for currentNum in moves[n][:] :
          sums.append({'sum' : sumDat['sum']+currentNum, 'lastNum' : currentNum})
        del sums[i]
    return sums
   
P = getPossibilities(10)
seq = [x['sum']%10 for x in P]

seven = [elem for elem in P if elem['sum']%7 == 0]
countSeven = len(seven)

five = [elem for elem in seven if elem['sum']%5 == 0]
countFive = len(five)

prob = countFive/countSeven

print(prob)


  