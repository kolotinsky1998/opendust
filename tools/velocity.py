import numpy as np

def velocity(E, p):
    m_i = 6.6335209e-26

    e = 1.6e-19
    c = 3e8
    a_p = 0.1
    a_E = c/1e4
    a_v = 1e-2
    a_a = a_v*a_p/a_E
    a_b = a_p/a_E

    a_cgs = 5.84e8
    b_cgs = 1.06e4

    a_si = a_a*a_cgs
    b_si = a_b*b_cgs

    v = a_si*(E/p)/np.sqrt(1+b_si*(E/p))
    nu = e*E/(m_i*v)

    return v, nu

if __name__ == "__main__":
    E = 4000
    p = 18.131842
    v_fl, nu = velocity(E, p)
    print(v_fl, nu)