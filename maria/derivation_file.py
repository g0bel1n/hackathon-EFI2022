

#must define A, B and S in function of T
def s_derivated_calculation(mu, delta, T, theta):
    x=x(T)
    A, B = R_calculation(mu, delta, r, F, T)
    S=A/ math.sqrt(B-A^2)
    s_derivated=0
    F_derivated=0
    for t in range(1, T+1):
        first_term= (S*(1+S^2)*A - S^3*R[t])/(A^2*T)
        sgn =  numpy.sign(F[t] - F[t-1])
        second_term = ( (-1)*mu*delta*sgn)*(1- numpy.tanh(numpy.dot(x[t],theta))^2)*(x[t] + theta[M+2]*F_derivated)+ ((-1)*R[t] + mu*delta*sgn*F_derivated)
        s_derivated+= first_term*second_term
        F_derivated =(1- numpy.tanh(numpy.dot(x[t],theta))^2)*(x[t] + theta[M+2]*F_derivated)

    return s_derivated

def R_calculation(mu, delta, T, price_T, F_T):
    A=0
    B=0
    for t in range(1,T):
        r_t = log(price_T[t]) - log(price_T[t-1])
        R= mu*(F[t-1]*r[t]- delta*numpy.absolute(F[t]- F[t-1]))
        A+= R
        B+= R^2
    return(A, B)
