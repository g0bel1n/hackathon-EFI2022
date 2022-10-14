#N number of assets

def s_derivated_calculation(N, mu, delta, T, theta, r, x, F):
    A, B, R= R_calculation(N, mu, delta,T,  r, F)
    S=A/math.sqrt(B-A**2)
    s_d_theta=0
    F_d_theta=numpy.zeros(N)
    for t in range(1, T+1):
        first_term= (S*(1+S**2)*A - S**3*R[t])/(A**2*T)
        sgn =  numpy.sign(F[t,:] - F[t-1,:])
        second_term = ( (-1)*mu*delta*sgn)*(1- F[t,:] @ F[t,:])*(x[t] +  theta[1+(M+1)*N :1+N*(M+2),: ]*F_d_theta)+ ((-1)*r[t]*mu + mu*delta*sgn)*F_d_theta)
        s_d_theta+= first_term*second_term
        F_d_theta =(1- F[t,] @ F[t,])*(x[t] + theta[1+(M+1)*N :1+N*(M+2), ]*F_d_theta)

    return s_derivated


#we know r[N] from the price data
# CEO
#besoin de T vecteurs r de r1 à rT
#besoin de T vecteurs F de F1 à FT
def R_calculation(N, mu, delta, T, r, F):
    A=0
    B=0
    R= [0 for a in range(T)]
    for t in range(1,T+1):
        R_t= mu*(numpy.dot(F[t,:],r[t,:])- delta* numpy.linalg.norm((F[t,:]- F[t-1,:]), ord=1)
        R[t]=R_t
        A+= R[t]
        B+= R[t]**2
    return(A/T, B/T, R)