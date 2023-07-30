load('impl-gen.sage')


#set_random_seed(0)

n,d,c_max, x_max, p_max = 64,  3, 4, 4, 24  
#n,d,c_max, x_max, p_max = 96, 3, 4, 4, 26
#n,d,c_max, x_max, p_max = 128,3, 4, 4, 28
 
q = 2^48

p = mkP()

R =[]
for j in range(1000):
    x1, h1, x2, h2, F1, F2, z = keygen()

    G = mkG()
    print ("p: ", p)
    #print ("G0: ")
    #print (G[0])

    #print ("G1: ")
    #print (h1)

    #print ("G2: ")
    #print (G[2])
    correct=0
    count=0
    sum  = 0

    T = []
    for i in range(20):
        u, s, trials =sign(x1, x2, F1, F2, z, h1, h2, i)
        sum = sum + trials
        T = T + [trials]
        #print ("-----------: s, u")
        #print (s[0])
        #print (s[1])
        #print (u)
        r=verify(i,h1, h2, u, s)
        count = count +1
        if r:
            correct=correct+1
        else:
            print ("Wrong signature")
            break

    if not r:
        break
    R = R + [(j, float(correct/count), float(sum/count))]
    print ("trlias: ", j, T)
    if float(sum/count) > 4:
        print ("big one: ", i, j, float(sum/count), T)
        break

print("correctness ratio, trials ratio:", R)

