# ================================================
# eMLE-Sig 2.0 - revised 
# ================================================
from hashlib import sha256, sha3_256, sha3_512

#set_random_seed(1)

def sumList(l, n):
    sum = 0
    for i in range(n):
        sum = sum + l[i]
    return sum


def mul(a, b, n, m):

    c = [0 for _ in range(n)]
    noise = vector(ZZ, [0 for _ in range(n)])
    for i in range(n):
        c[i] = 0 
        for j in range(n):
            c[i] = (c[i]+a[j]*b[(i-j)%n])
            if m>0:
                c[i] = c[i]%m 

    return vector(ZZ, c)


def mkVecC(msg): #only for reference
    c1 = vector(ZZ, [0 for i in range(n)])
    c2 = vector(ZZ, [0 for i in range(n)])
    for i in range(n):
        hv = sha3_256((str(i)+msg).encode()).hexdigest()
        c1[i] =  int(hv[0:8], 16)%(c_max)
        c2[i] =  int(hv[8:16], 16)%(c_max)

    return c1, c2 

def mkP():
    p = vector(ZZ, [0 for _ in range(d)])
    for i in range(d):
        if i == 0:
            p[i] = next_prime(c_max)
        if i== d-1:
                p[i] = 2^p_max 

        if i>1 and i < d-1:
            p[i] = next_prime(2*(n)*(c_max-1)*(p[i-1])+p[i-1] ) 
        if i==1:
            p[i] = next_prime((n//2)*(c_max-1)*(p[i-1])+p[i-1]+7*n) 

    return p

def mkG():
    p = mkP()
    G = []
    for l in range(d): 
        g = vector(ZZ, [(int(sha3_256((str(l)+str(k)+str(n)+str(d)+str(c_max)+str(x_max)+str(p)+str(0)).encode()).hexdigest(),16))%p[l] for k in range(n)] )
        gp = vector(ZZ, [(int(sha3_256((str(l)+str(k)+str(n)+str(d)+str(c_max)+str(x_max)+str(p)+str(1)).encode()).hexdigest(),16))%p[l] for k in range(n)] )
        G = G + [[g, gp]]

    return G

def getVC(n): 
    if n==64:
        return [255, 255, 912, 2560]
    if n==96:
        return [351, 351, 2228, 5632]
    if n==128:
        return [415, 415, 4324, 9750]


def eMLE(x, o, a):
    p = mkP()
    G = mkG()	
    h = vector(ZZ, [0 for _ in range(n)])
    sumR = 0
    F  = []
    for i in range(d):
        if i==0:
            h = (h+mul(G[i][0], x[0]+(o%p[0]), n, p[i])+mul(G[i][1], x[1]+(o%p[0]), n, p[i]))%p[i]
        else:
            h = (h+mul(G[i][0], x[0], n, p[i])+mul(G[i][1], x[1], n, p[i]))%p[i]        

        if (i ==d-2):   
            num = int(((int(p[i+1]))-sumList(h, n)*(c_max-1))/(c_max*p[i])) 
            if num < 0:
                num = 0

            num = num//2
            t = num
            locations  = int(n/(2))
            print ("noise amount: ", i, num)  
            k = vector(ZZ, [0 for _ in range(n) ])
            w = vector (ZZ, [randint(0, n-1) for _ in range(locations)])   
            for j in range(locations-1):
                if int(num/(locations-j)) > 0:
                    r = randint(0, int(num/(locations-j)))
                else:
                    r = 0    

                h[w[j]] = h[w[j]]+(r%q)*p[i]
                k[w[j]] = k[w[j]] +(r%q)  
                num = num - r
            
            h[w[locations-1]] = h[w[locations-1]]+(num%q)*p[i]
            k[w[locations-1]] = k[w[locations-1]] + (num%q) 

            w = vector (ZZ, [randint(0, n-1) for _ in range(2)])
            r = randint(0, int(t/3))
            for j in range(n):
                    if h[(j+w[0])%n] < p[i]:
                        h[(j+w[0])%n] = h[(j+w[0])%n]-(r%q)*p[i]
                        k[(j+w[0])%n] = k[(j+w[0])%n] - (r%q)
                        break
            #r = randint(0, int(t/3) - r)
            r = int(t/3) - r
            for j in range(n):                    
                    if h[(j+w[1])%n] < p[i] and h[(j+w[1])%n]>0:
                        h[(j+w[1])%n] = h[(j+w[1])%n]-(r%q)*p[i]
                        k[(j+w[1])%n] = k[(j+w[1])%n] - (r%q)
                        break

            for j in range(n):
                    if h[j] < p[i] and h[j] > 0: 
                        if a==1:
                            r = randint(-2*int(n), 2*int(n))
                        else:
                            r = randint(-2*int(n), 2*int(n))
                            sumR = sumR + r
                        if r>0:
                            h[(j)] = h[(j)]+(r%q)*p[i]
                            k[(j)] = k[(j)] + (r%q)
                        else:
                            h[(j)] = h[(j)]-((-r)%q)*p[i]
                            k[(j)] = k[(j)] - ((-r)%q)   
                        
            print (k)  

        F = F + [h]

    return h, F, sumR

def keygen():
    p = mkP()
    G = mkG()

    while 1:
        x1 = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)])
        x2 = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)])
        x1p = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)])
        x2p = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)])

        sumX = sumList(x1+x2+x1p+x2p, n) #+ sumList(x2, n)
        if sumX < 0:
            sumX = -sumX;

        if sumX < n//2 :
            break

    while 1:        
        h1, F1, sumR1 = eMLE([x1, x1p], G[1][0], 0)
        h2, F2, sumR2 = eMLE([x2, x2p], G[1][0], 0)
        sumR = sumR1+sumR2
        if sumR < 0:
            sumR = -sumR;

        if sumR < 8*n:
            break

    z = vector(ZZ, [ randint(0, p[d-1]-1) for _ in range(n)])
    h1 = (h1+z)%p[d-1]
    h2 = (h2+z)%p[d-1] 

    return [x1, x1p], h1, [x2, x2p],  h2, F1, F2, z

def sign(x1, x2, F1, F2, z, h1, h2, m):
    p = mkP()
    G = mkG()
    vc =  getVC(n)

    trials =0
    cp1, cp2 = mkVecC(str(m)+str(h1)+str(h2))
    
    while 1:
        trials = trials + 1 
    
        y = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)]) 
        yp = vector(ZZ, [ randint(-x_max, x_max) for _ in range(n)]) 
        u, F3, _ = eMLE((y,yp), cp1+cp2, 1)    
        u = (u-z)%p[d-1]   
        
        c1, c2 = mkVecC(str(m)+str(u)+str(h1)+str(h2))
        s = mul(x1[0]+y, c1, n, 0)+ mul(x2[0]+y, c2, n, 0) 
    
        valid = ckSize(s, -vc[0], vc[1])
        if not valid:
            continue

        sp = mul(x1[1]+yp, c1, n, 0)+ mul(x2[1]+yp, c2, n, 0) 

        valid = ckSize(sp, -vc[0], vc[1])
        if not valid:
            continue

        valid = true
        for l in range(d-1):
            L = mul(F1[l]+F3[l], c1, n, 0)+mul(F2[l]+F3[l], c2, n, 0)
            valid = valid and ckSize(L, 0, p[l+1]-1)

            if l==0:
                g = mul(G[1][0]+cp1+cp2, c1+c2, n, p[l])
                g1 = (mul(G[l][0], s+g, n, p[l])+mul(G[l][1], sp+g, n, p[l])) 
                K0 = (L-g1)/p[0]
                a = int(sumList(K0, n)/n)
                k01 = vector(ZZ, [ ((K0[j])- a) for j in range(n)])
                valid =valid and (k01.norm()*k01.norm() > vc[2]) and (k01.norm()*k01.norm() < vc[3])              

        if valid:
            break 

    return u, [s, sp], trials

def ckSize(s, min, max):
    v = True
    for i in range(n):
        v = v and s[i] <= max and s[i] >= min 
    return v

def verify(m, h1, h2, u, s):
    p = mkP()
    G = mkG()
    vc =  getVC(n)

    cp1, cp2 = mkVecC(str(m)+str(h1)+str(h2))
    c1, c2 = mkVecC(str(m)+str(u)+str(h1)+str(h2))

    v = ckSize(s[0], -vc[0], vc[1])
    v = v and ckSize(s[1], -vc[0], vc[1])

    t = mul(h1+u, c1, n, p[d-1]) + mul(h2+u, c2, n, p[d-1]) 
    for i in range(d):			
        if d-i-1  ==0:
            g = mul(G[1][0]+cp1+cp2, c1+c2, n, p[d-i-1])
            g1 = (mul(G[d-i-1][0], s[0]+g, n, p[d-i-1])+mul(G[d-i-1][1], s[1]+g, n, p[d-i-1])) 
            K0a = (t-g1 )/p[d-i-1]
            K0 = vector(ZZ, [int(K0a[k]) for k in range(n)])

            a = int(sumList(K0, n)/n)
            k01 = vector(ZZ, [ (K0[k] - a) for k in range(n)])
            v = v  and (k01.norm()*k01.norm() > vc[2]) and (k01.norm()*k01.norm() < vc[3])
            t = (t-g1)%p[d-i-1]
        else:
            t = (t-mul(G[d-i-1][0], s[0], n, p[d-i-1])-mul(G[d-i-1][1], s[1], n, p[d-i-1]))%p[d-i-1]

    v = v and (t==0)
    #print ("Verification layer 0------------")
    print (f't:  \x1b[32m{t}\x1b[0m')	 	
    return v
