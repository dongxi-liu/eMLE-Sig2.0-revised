# =========================
# Compact-LWE+ Reference Implementation 
# =========================
q=2^64
t=2^12
np=3
w = 2^10
wp = 2^10
p_size = 2^24
R=Integers(q)
e_max  = 2^8

# ========================
def prikey_gen(m ,n, a_size, type=1):
    S = random_matrix(R,np, n) 
    Sp = random_matrix(R,np, n)

    p_min = 2*t*(2*t*m+w*m+wp+m+1) 
    p_max = int(q/(2*m*t*(e_max+1)))

    if type ==3 :
        Spp = random_matrix(R,np, n)
        p_min = 2*t*(3*t*m+w*m+wp+m+1+t) 
        p_max = int(q/(2*(m*t+t+(m-1)*2)*(e_max+1)))

    if p_min + p_size >= p_max:
       print "\n wrong parameters for p!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    p = p_min + randint(0, p_size-1) #randint(p_min+1, p_max-1)
    while gcd(p,q)>1 :
        p = p + 1 #randint(p_min+1, p_max-1)
 

    if p >= p_max:
       print "\n too big p!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

    #print "p_min, p_max:", p_min, p_max
 
    F = random_matrix(R,3,3)
    F = F.change_ring(R)
  
    while gcd(F.det(), q)>1:
		F =random_matrix(R,3,3)


    Fp =random_matrix(R,3,3)
  
    while gcd(Fp.det(), q)>1:
		Fp =random_matrix(R,3,3)

    if type ==3 :
        Fpp =random_matrix(R,3,3)
  
        while gcd(Fpp.det(), q)>1:
		    Fpp =random_matrix(R,3,3)

        #k is new in this variant
        k = randint(0, t-1)
        while gcd(k, t)>1:
		    k = randint(0, t-1)
    
        return S, p, F, Fp, Sp, Fpp, Spp, k


    return S, p, F, Fp, Sp

def pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, Fpp=0, Spp=0, k=0, type=1):
    Rp = Integers(p)
    Rt = Integers(t)

    A1 = random_matrix(ZZ,m,n,x=0,y=a_size)
    A2 = random_matrix(ZZ,m,n,x=0,y=a_size) #A1

    u  = vector(Rt,[randint(0, t-1) for _ in range(m)])
    if type==3:
        u[0] = k

    u1 = vector(Rt,[randint(0, t-1) for _ in range(m)])
    u2 = u - u1

    tilde_e_max = int(q/(2*m*t*p)) -1 
    if type==3:
        tilde_e_max = int(q/(2*(m*t+t+(m-1)*2)*p)) -1 

    e11 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
    e21 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
    e31 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])

    e12 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
    e22 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
    e32 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])

    if type==1:
        up = vector(ZZ,[randint(0, t-1) for i in range(m)])
    else:
        up = vector(ZZ,[randint(0, p-1) for i in range(m)])

    if type==3:
        up[0] = t 

    tInv = Rp(1)/Rp(t)
    tInv = tInv.lift()

    r11 =vector(R, [(up[i]*tInv-randint(0,w))%p for i in range(m)])
    if type==3:
        r11[0] = 0 

    r21 =vector(R, [randint(0,p-1) for i in range(m)])
    r31 =vector(R,  [(u1[i].lift()-r11[i].lift()-r21[i].lift())%p for i in range(m)])
    
    r12 =vector(R, [(randint(0,p-1)) for i in range(m)])    
    r22 =vector(R, [randint(0,p-1) for i in range(m)])
    r32 =vector(R, [(u2[i].lift()-r12[i].lift()- r22[i].lift())%p for i in range(m)])


    r12  = vector(R, [ (r12[i].lift()-r11[i].lift())%p for i in range(m)])


    b1 = A1*S[0] + (r11 + p*e11)
    d1 = A1*S[1]+  (r21 + p*e21)
    g1 = A1*S[2] + (r31 + p*e31)


    temp1 = b1*F[0,0] + d1*F[1,0]+g1*F[2,0] 
    temp2 = b1*F[0,1] + d1*F[1,1]+g1*F[2,1]
    temp3 = b1*F[0,2] + d1*F[1,2]+g1*F[2,2]

    b1 = temp1    
    d1 = temp2
    g1 = temp3     


    b2 = A2*Sp[0] + (r12 + p*e12)
    d2 = A2*Sp[1]+  (r22 + p*e22)
    g2 = A2*Sp[2] + (r32 + p*e32)
   
    
    temp1 = b2*Fp[0,0] + d2*Fp[1,0]+g2*Fp[2,0] 
    temp2 = b2*Fp[0,1] + d2*Fp[1,1]+g2*Fp[2,1]
    temp3 = b2*Fp[0,2] + d2*Fp[1,2]+g2*Fp[2,2]
 
    b2 = temp1    
    d2 = temp2
    g2 = temp3 


    if type==3:
        A3 = random_matrix(ZZ,m,n,x=0,y=a_size)
        A4 = random_matrix(ZZ,m,n,x=0,y=a_size)     

        e13 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
        e23 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
        e33 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])

        e14 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
        e24 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])
        e34 =vector(R, [randint(-tilde_e_max,tilde_e_max) for _ in range(m)])

        r13 =vector(R, [randint(0,p-1) for i in range(m)])    
        r23 =vector(R, [randint(0,p-1) for i in range(m)])
        r33 =vector(R, [(u[i].lift()-r13[i].lift()- r23[i].lift())%p for i in range(m)])


        r13  = vector(R, [ (r13[i].lift()+r11[i].lift())%p for i in range(m)])

        b3 = A3*Spp[0] + (r13 + p*e13)
        d3 = A3*Spp[1]+  (r23 + p*e23)
        g3 = A3*Spp[2] + (r33 + p*e33)


        temp1 = b3*Fpp[0,0] + d3*Fpp[1,0]+g3*Fpp[2,0] 
        temp2 = b3*Fpp[0,1] + d3*Fpp[1,1]+g3*Fpp[2,1]
        temp3 = b3*Fpp[0,2] + d3*Fpp[1,2]+g3*Fpp[2,2]

        b3 = temp1    
        d3 = temp2
        g3 = temp3     


        r14 =vector(R, [(randint(0,p-1)) for i in range(m)])    
        r24 =vector(R, [randint(0,p-1) for i in range(m)])
        r34 =vector(R, [(up[i]-r14[i].lift()- r24[i].lift())%p for i in range(m)])

        b4 = A4*Spp[0] + (r14 + p*e14)
        d4 = A4*Spp[1]+  (r24 + p*e24)
        g4 = A4*Spp[2] + (r34 + p*e34)


        temp1 = b4*Fpp[0,0] + d4*Fpp[1,0]+g4*Fpp[2,0] 
        temp2 = b4*Fpp[0,1] + d4*Fpp[1,1]+g4*Fpp[2,1]
        temp3 = b4*Fpp[0,2] + d4*Fpp[1,2]+g4*Fpp[2,2]

        b4 = temp1    
        d4 = temp2
        g4 = temp3     


        return A1, b1.lift(),d1.lift(), g1.lift(), A2, b2.lift(),d2.lift(), g2.lift(), A3, b3.lift(),d3.lift(), g3.lift(), A4, b4.lift(),d4.lift(), g4.lift()


    return A1, b1.lift(),d1.lift(), g1.lift(), A2, b2.lift(),d2.lift(), g2.lift(), u.lift(), up.lift()


def enc(m, n, a_size, A1,b1,d1, g1, A2, b2, d2, g2, u, up, v, A3=0, b3=0,d3=0, g3=0, A4=0, b4=0,d4=0, g4=0, type=1):


    l = vector(ZZ,[randint(-t+1,t-1) for _ in range(m)])

    l1 = vector(ZZ, [0 for _ in range(m)])
    r = vector(ZZ, [randint(0,1) for _ in range(m)])
    for i in range(m):
        if (r[i]==1):
            if l[i] > 0:
                l1[i] = l[i] - t
            else:
                l1[i] = l[i] + t
        else:
            l1[i] = l[i]


    ca1 = l1 * A1

    cb1 = (l1 * b1) % q
    cd1 = (l1 * d1) % q
    cg1 = (l1 * g1) % q 


    l2 = vector(ZZ, [0 for _ in range(m)])
    r = vector(ZZ, [randint(0,1) for _ in range(m)])


    for i in range(m):
        if (r[i]==1):
            if l[i] > 0:
                l2[i] = l[i] - t
            else:
                l2[i] = l[i] + t
        else:
            l2[i] = l[i]


    ca2 = l2 * A2

    cb2 = (l2 * b2) % q
    cd2 = (l2 * d2) % q
    cg2 = (l2 * g2) % q 

    if type==1 or type==2: 
        l3 = (l1-l2)/t
        if type==1:
            z = (l3*up + l*u-v)%t
        else:
            z = l3*up + (l*u-v)%t + (randint(-wp+1, wp-1))*t
        return ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, l1, l2, z
    else:
        l3 = vector(ZZ, [(2*l1[i]-l2[i]-l[i])/t for i in range(m)]) 
        l3[0] = randint(-wp+1, wp-1)
        l[0] = (l[0] - v)%t
        ca3 = l * A3
        cb3 = (l * b3) % q
        cd3 = (l * d3) % q
        cg3 = (l * g3) % q 
        ca4 = l3 * A4
        cb4 = (l3 * b4) % q
        cd4 = (l3 * d4) % q
        cg4 = (l3 * g4) % q 
  
        return ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, l1, l2, ca3+ca4, (cb3+cb4)%q, (cd3+cd4)%q, (cg3+cg4)%q  
      

def dec(m, n, a_size, S, p, F, Fp, Sp, ca1,cb1,cd1, cg1, ca2, cb2, cd2, cg2, z, Fpp=0, Spp=0, k=0, ca3=0, cb3=0, cd3=0, cg3=0, type=1):

    FInv = F.inverse()
    FpInv = Fp.inverse()
    #check

    temp1 = cb1*FInv[0,0] + cd1*FInv[1,0]+cg1*FInv[2,0] 
    temp2 = cb1*FInv[0,1] + cd1*FInv[1,1]+cg1*FInv[2,1]
    temp3 = cb1*FInv[0,2] + cd1*FInv[1,2]+cg1*FInv[2,2]
 

    d1 = ((temp1 -S[0]*ca1))%q
    d2 = ((temp2-S[1]*ca1))%q
    d3 = ((temp3 -S[2]*ca1))%q
    
    d1 = d1.lift()
    if d1 > int(q/2):       
       d1 = d1 - q

    d2 = d2.lift()
    if d2 > int(q/2):
       d2 = d2 - q

    d3 = d3.lift() 
    if d3 > int(q/2):
       d3 = d3 - q
   
    temp1 = cb2*FpInv[0,0] + cd2*FpInv[1,0]+cg2*FpInv[2,0] 
    temp2 = cb2*FpInv[0,1] + cd2*FpInv[1,1]+cg2*FpInv[2,1]
    temp3 = cb2*FpInv[0,2] + cd2*FpInv[1,2]+cg2*FpInv[2,2]

    d1p = ((temp1 -Sp[0]*ca2))%q
    d2p = ((temp2- Sp[1]*ca2))%q
    d3p = ((temp3 -Sp[2]*ca2))%q
    
    d1p = d1p.lift()
    if d1p > int(q/2):       
       d1p = d1p - q

    d2p = d2p.lift() 
    if d2p > int(q/2):
       d2p = d2p - q

    d3p = d3p.lift() 
    if d3p > int(q/2):
       d3p = d3p - q

    if type==1 or type==2:
        u = (d1+d2+d3+d1p+d2p+d3p+ d1-z)%p  
        if u > int(p/2):
            u =  u - p
        v = (u) %t
    
        return v
    else:
        FppInv = Fpp.inverse()
        temp1 = cb3*FppInv[0,0] + cd3*FppInv[1,0]+cg3*FppInv[2,0] 
        temp2 = cb3*FppInv[0,1] + cd3*FppInv[1,1]+cg3*FppInv[2,1]
        temp3 = cb3*FppInv[0,2] + cd3*FppInv[1,2]+cg3*FppInv[2,2]

        d13 = ((temp1 -Spp[0]*ca3))%q
        d23 = ((temp2 -Spp[1]*ca3))%q
        d33 = ((temp3 -Spp[2]*ca3))%q
    
        d13 = d13.lift()
        if d13 > int(q/2):       
            d13 = d13 - q
        d23 = d23.lift()
        if d23 > int(q/2):
            d23 = d23 - q

        d33 = d33.lift() 
        if d33 > int(q/2):
            d33 = d33 - q

        z = (d13+d23+d33)%p 
        u = (d1+d2+d3+d1p+d2p+d3p+ 2*d1- z)%p  

        if u > int(p/2):
            u =  u - p

        Rt = Integers(t)
        kInv = Rt(1)/Rt(k)
        kInv = kInv.lift()
        v = (u*kInv) %t

        return v 

def SIS_original(m, n, a_size, A,b,d,g, ca,cb,cd, cg):
    #kappa=q
    #kappa2=kappa
    #kappa3=t

    kappa=t*t*t
    kappa2=t*t*t
    kappa3=1

    L=block_matrix(ZZ, \
            [[1, 0, -kappa*ca.row(), -kappa2 * cb, -kappa2 * cd, -kappa2 * cg], \
             [0, kappa3*identity_matrix(m), kappa*A, kappa2 * b.column(), kappa2 * d.column(), kappa2 * g.column()], \
             [0, 0, 0, kappa2*q, 0,0],\
             [0, 0, 0, 0, kappa2*q,0],\
             [0, 0, 0, 0,0, kappa2*q]])

    L=L.LLL() #BKZ(block_size=10) 

    idx=next((i for i,x in enumerate(L.column(0).list()) if x!=0))
    lp = vector(ZZ,L[idx][1:(m)+1]/kappa3) if L[idx][0] == 1 else vector(ZZ,-L[idx][1:(m)+1]/kappa3)

    return lp

def SIS_short(m, n, a_size, A1,b1,d1,g1, ca1,cb1,cd1, cg1, A2, b2,d2, g2, ca2,cb2,cd2, cg2):
    #kappa=q
    #kappa2=kappa
    #kappa3=t*t*t*t

    kappa=t*t*t
    kappa2=t*t*t
    kappa3=1
    tail=vector(ZZ, [0 for _ in range(m)])
    L=block_matrix(ZZ, \
            [[1, 0, -kappa*ca1.row(), -kappa*ca2.row(), -kappa2 * cb1, -kappa2 * cd1, -kappa2 * cg1, -kappa2 * cb2, -kappa2 * cd2, -kappa2 * cg2], \
             [0, kappa3*identity_matrix(m), kappa*A1, kappa*A2, kappa2 * b1.column(), kappa2 * d1.column(), kappa2 * g1.column(), 
                 kappa2 * b2.column(), kappa2 * d2.column(), kappa2 * g2.column()], \
             [0, 0, 0, 0, kappa2*q,0, 0,0,0,0],\
             [0, 0, 0, 0, 0, kappa2*q, 0,0,0,0],\
             [0, 0, 0, 0,0, 0, kappa2*q,0,0,0],\
             [0, 0, 0, 0,0, 0, 0,kappa2*q,0,0],\
             [0, 0, 0, 0,0, 0, 0,0,kappa2*q,0],\
             [0, 0, 0, 0,0, 0, 0,0,0,kappa2*q]
             ])


    L=L.LLL() #block_size=10 m-1

    #index of first non-zero entry in the first column of L
    idx=next((i for i,x in enumerate(L.column(0).list()) if x!=0))

    lp = vector(ZZ,L[idx][1:(m)+1]/kappa3) if L[idx][0] == 1 else vector(ZZ,-L[idx][1:(m)+1]/kappa3)

    #print "\n cipher", ca2, cb2, cd2, cg2
    #print "cracked:", lp*A2, (lp*b2) %q, (lp*d2)%q, (lp*g2)%q


    return lp

def SIS_type_2(up , z):
    kappa=2
    kappa2=2
    kappa3=1
 
    eup = vector(ZZ, [0 for i in range(m+2)])
    for i in range(m):
        eup[i] = up[i]
    eup[m] = t
    eup[m+1] = 1

    L=block_matrix(ZZ, \
            [[1, 0,                     -kappa2 * z], \
             [0, kappa3*identity_matrix(m+2),  kappa2 * eup.column()]])

    L=L.LLL() #BKZ(block_size=10) #blenc_zock_size=10 m-1

    idx=next((i for i,x in enumerate(L.column(0).list()) if x!=0))

    lp = vector(ZZ,L[idx][1:(m+2)+1]/kappa3) if L[idx][0] == 1 else vector(ZZ,-L[idx][1:(m+2)+1]/kappa3)

    #print "\n"
    return lp, lp*eup==z

def correct(m, n, a_size, pairs=10):
    allcorrect = 1
    for i in range(pairs):
        v = randint(0,t-1)
        S, p, F, Fp, Sp = prikey_gen(m, n, a_size, type=1)
        A1, b1,d1, g1, A2, b2,d2, g2, u, up = pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, type=1)
        ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, ol1, ol2, z  = enc(m, n, a_size, A1,b1,d1, g1, A2, b2, d2, g2, u, up, v, type=1) 
        dec_v = dec(m, n, a_size, S, p, F, Fp, Sp, ca1,cb1,cd1, cg1, ca2,cb2,cd2, cg2, z, type=1)
        if(v!=dec_v):
            allcorrect = 0
            print "Compact-LWE+ not correct", v, dec_v   

        S, p, F, Fp, Sp = prikey_gen(m, n, a_size, type=2)
        A1, b1,d1, g1, A2, b2,d2, g2, u, up = pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, type=2)
        ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, ol1, ol2, z  = enc(m, n, a_size, A1,b1,d1, g1, A2, b2, d2, g2, u, up, v, type=2) 
        dec_v = dec(m, n, a_size, S, p, F, Fp, Sp, ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, z, type=2)
        if(v!=dec_v):
            allcorrect = 0
            print "Compact-LWE+ (1st variant) not correct", v, dec_v  

        S, p, F, Fp, Sp, Fpp, Spp, k = prikey_gen(m, n, a_size, type=3)
        A1, b,d, g, A2, b1,d1, g1, A3, b3,d3, g3, A4, b4,d4, g4 = pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, Fpp, Spp, k, type=3)
        ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, ol1, ol2, ca3, cb3, cd3, cg3  = enc(m, n, a_size, A1,b,d, g, A2, b1, d1, g1, 0, 0, v, A3, b3,d3, g3, A4, b4,d4, g4, type=3)
        dec_v = dec(m, n, a_size, S, p, F, Fp, Sp, ca1,cb1,cd1, cg1, ca2,cb2,cd2, cg2, 0, Fpp, Spp, k, ca3, cb3, cd3, cg3, type=3)
        if(v!=dec_v):
            allcorrect = 0
            print "Compact-LWE+ (2nd variant) not correct", v, dec_v   

    if allcorrect==1:
       print "all correct"

def attack(m, n, a_size, trials=10,pairs=10):
    short_l_attack = 0
    original_l_attack = 0
    norm_sum = 0

    for npair in range(pairs):
        S, p, F, Fp, Sp = prikey_gen(m, n, a_size, type=1)

        A1, b1,d1, g1, A2, b2,d2, g2, u, up = pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, type=1)

        for _ in range(trials):
            v = randint(0,t-1)
            ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, ol1, ol2, z  = enc(m, n, a_size, A1,b1,d1, g1, A2, b2, d2, g2, u, up, v, type=1)
 

            lp = SIS_short(m, n, a_size, A1,b1,d1,g1, ca1,cb1,cd1, cg1, A2, b2,d2, g2, ca2,cb2,cd2, cg2) 

            if ((lp*u)-z)%t == v and (lp*b2) %q == cb2:
                short_l_attack = short_l_attack +1 

            norm_sum = norm_sum + lp.norm(2)



            lp1 = SIS_original(m, n, a_size, A1,b1,d1,g1, ca1,cb1,cd1, cg1)           
            lp2 = SIS_original(m, n, a_size, A2,b2,d2,g2, ca2,cb2,cd2, cg2)


            if(sum(ol2-lp2)==0) and (sum(ol1-lp1)==0):
                if ol2[0] == lp2[0] and ol1[0] == lp1[0]:
                    original_l_attack = original_l_attack+1
         

    return short_l_attack, original_l_attack, int(norm_sum/(pairs*trials))

def z_attack(m, n, a_size, trials=10,pairs=10):
    vul = 0
    for npair in range(pairs):
        S, p, F, Fp, Sp = prikey_gen(m, n, a_size, type=2)
        A1, b1,d1, g1, A2, b2,d2, g2, u, up = pubkey_gen(m, n, a_size, S, p, F, Fp, Sp, type=2)             
        v = randint(0,t-1)
        ca1, cb1, cd1, cg1, ca2, cb2, cd2, cg2, ol1, ol2, z  = enc(m, n, a_size, A1,b1,d1, g1, A2, b2, d2, g2, u, up, v, type=2)
        lp, check = SIS_type_2(up , z)
        if check:
            print "lp=", lp      
            if lp[m+1] == (ol1*u-v)%t: #note that (ol1*u-v)%t== (l*u-v)%t 
                vul = vul +1
                print "vulnerability of 1st variant"     
                          
    return vul

# use PRG for reproducibility
seed = 1
set_random_seed(seed)

m = 64
n = m/4
a_size = n*2^((n/2))

p_min = 2*t*(2*t*m+w*m+wp+m+1) 
p_max = int(q/(2*m*t*(e_max+1)))

print "p_min, p_max:", p_min, p_max

correct(m, n, a_size, 100)

o = open('log-m-'+str(m)+'-seed-'+str(seed)+'.txt','w')

print "m=", m, "n changing"

o.write("m="+ str(m)+", n changing\n")

short_l_attack = 0 
pairs = 8
trials = 2
n_dec = n
while (short_l_attack<pairs*trials):
    short_l_attack, original_l_attack, l2_norm = attack(m, n_dec, a_size, trials, pairs)
    print n_dec, short_l_attack, original_l_attack, l2_norm
    o.write(str(n_dec) + ', ' + str(short_l_attack)+', '+str(original_l_attack)+', '+str(l2_norm)+'\n')
    n_dec = n_dec - 1

original_l_attack=0
n_inc = n +1
while (original_l_attack<pairs*trials):
    short_l_attack, original_l_attack, l2_norm = attack(m, n_inc, a_size, trials, pairs)
    print n_inc, short_l_attack, original_l_attack, l2_norm
    o.write(str(n_inc) + ', ' + str(short_l_attack)+', '+str(original_l_attack)+', '+str(l2_norm)+'\n')
    n_inc = n_inc + 1

print "m=", m, "a_size changing"
o.write("\n m="+ str(m)+", a_size changing\n")

short_l_attack = 0 

n_dec = n
while (short_l_attack<pairs*trials):
    a_size = n_dec*2^(int(n_dec/2))
    short_l_attack, original_l_attack, l2_norm = attack(m, n, a_size, trials, pairs)
    print n_dec, a_size, short_l_attack, original_l_attack, l2_norm
    o.write(str(n_dec) + ', '+ str(a_size)+', ' + str(short_l_attack)+', '+str(original_l_attack)+', '+str(l2_norm)+'\n')
    n_dec = n_dec - 2

original_l_attack=0
n_inc = n + 2
while (original_l_attack<pairs*trials):
    a_size = n_inc*2^(int(n_inc/2))
    short_l_attack, original_l_attack, l2_norm  = attack(m, n, a_size, trials, pairs)
    print n_inc, a_size, short_l_attack, original_l_attack, l2_norm
    o.write(str(n_inc) + ', '+ str(a_size)+', ' + str(short_l_attack)+', '+str(original_l_attack)+', '+str(l2_norm)+'\n')
    n_inc = n_inc + 2

o.close()
#other attack

vul = z_attack(m, n, a_size, trials=10,pairs=10)
print "vul of 1st variant=", vul
