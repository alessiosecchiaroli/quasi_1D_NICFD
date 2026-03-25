import numpy as np
import CoolProp.CoolProp as cp


def PVRT(T,p,fluid):

    d = cp.PropsSI('D','T',T,'P',p,fluid)
    c = cp.PropsSI('A','T',T,'P',p,fluid)

    return d, c
