import numpy as np
import CoolProp.CoolProp as cp

'''
Useless function that calculates density and speed of sound from p,T, fluid
Honestly I don't remember why I called it PVRT 
'''
def PVRT(T,p,fluid):

    d = cp.PropsSI('D','T',T,'P',p,fluid)
    c = cp.PropsSI('A','T',T,'P',p,fluid)

    return d, c
