import numpy as np
from time import time
from matplotlib import pyplot as plt
import sys
from pathlib import Path

import numpy as np

# Import yroots from this checkout instead of whatever is installed.
YROOTS_DIR = Path("/data1/acme/Roots/hyun/Rootfinding-serial")
if YROOTS_DIR.is_dir():
    sys.path.insert(0, str(YROOTS_DIR))

import yroots as yr  # noqa: E402


def resids(funcs, roots):
    """ Finds the residuals of the given function at the roots.
    Parameters
    ---------
        func : function
            The function to find the residuals of.
        roots : numpy array
            The coordinates of the roots.

    Returns
    -------
        float : avg_resid
            The average residual
        float : max_resio
            The maximum residual
    """
    all_resids = np.array([])
    for func in funcs:
        all_resids = np.append(all_resids,np.abs(func(roots[:,0],roots[:,1])))
    return np.mean(all_resids), max(all_resids)

def test_roots_1_1():
    # Test 1.1
    f = lambda x,y: 144*(x**4+y**4)-225*(x**2+y**2) + 350*x**2*y**2+81
    g = lambda x,y: y-x**6
    funcs = [f,g]
    a, b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start
    return t, resids(funcs,yroots)

def test_roots_1_2():
    # Test 1.2
    f = lambda x,y: (y**2-x**3)*((y-0.7)**2-(x-0.3)**3)*((y+0.2)**2-(x+0.8)**3)*((y+0.2)**2-(x-0.8)**3)
    g = lambda x,y: ((y+.4)**3-(x-.4)**2)*((y+.3)**3-(x-.3)**2)*((y-.5)**3-(x+.6)**2)*((y+0.3)**3-(2*x-0.8)**3)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_1_3():
    # Test 1.3
    f = lambda x,y: y**2-x**3
    g = lambda x,y: (y+.1)**3-(x-.1)**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_1_4():
    # Test 1.4
    f = lambda x,y: x - y + .5
    g = lambda x,y: x + y
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_1_5():
    # Test 1.5
    f = lambda x,y: y + x/2 + 1/10
    g = lambda x,y: y - 2.1*x + 2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_2_1():
    # Test 2.1
    f = lambda x,y: np.cos(10*x*y)
    g = lambda x,y: x + y**2
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_2_2():
    # Test 2.2
    f = lambda x,y: x
    g = lambda x,y: (x-.9999)**2 + y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_2_3():
    # Test 2.3
    f = lambda x,y: np.sin(4*(x + y/10 + np.pi/10))
    g = lambda x,y: np.cos(2*(x-2*y+ np.pi/7))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_2_4():
    # Test 2.4
    f = lambda x,y: np.exp(x-2*x**2-y**2)*np.sin(10*(x+y+x*y**2))
    g = lambda x,y: np.exp(-x+2*y**2+x*y**2)*np.sin(10*(x-y-2*x*y**2))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_2_5():
    # Test 2.5
    f = lambda x,y: 2*y*np.cos(y**2)*np.cos(2*x)-np.cos(y)
    g = lambda x,y: 2*np.sin(y**2)*np.sin(2*x)-np.sin(x)
    funcs = [f,g]
    a,b = np.array([-4,-4]), np.array([4,4])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_3_1():
    # Test 3.1
    f = lambda x,y: ((x-.3)**2+2*(y+0.3)**2-1)
    g = lambda x,y: ((x-.49)**2+(y+.5)**2-1)*((x+0.5)**2+(y+0.5)**2-1)*((x-1)**2+(y-0.5)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_3_2():
    # Test 3.2
    f = lambda x,y: ((x-0.1)**2+2*(y-0.1)**2-1)*((x+0.3)**2+2*(y-0.2)**2-1)*((x-0.3)**2+2*(y+0.15)**2-1)*((x-0.13)**2+2*(y+0.15)**2-1)
    g = lambda x,y: (2*(x+0.1)**2+(y+0.1)**2-1)*(2*(x+0.1)**2+(y-0.1)**2-1)*(2*(x-0.3)**2+(y-0.15)**2-1)*((x-0.21)**2+2*(y-0.15)**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_4_1():
    # Test 4.1
    # This system hs 4 true roots, but ms fails (finds 5).
    f = lambda x,y: np.sin(3*(x+y))
    g = lambda x,y: np.sin(3*(x-y))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_4_2():
    # Test 4.2
    f = lambda x,y: ((90000*y**10 + (-1440000)*y**9 + (360000*x**4 + 720000*x**3 + 504400*x**2 + 144400*x + 9971200)*(y**8) +
                ((-4680000)*x**4 + (-9360000)*x**3 + (-6412800)*x**2 + (-1732800)*x + (-39554400))*(y**7) + (540000*x**8 +
                2160000*x**7 + 3817600*x**6 + 3892800*x**5 + 27577600*x**4 + 51187200*x**3 + 34257600*x**2 + 8952800*x + 100084400)*(y**6) +
                ((-5400000)*x**8 + (-21600000)*x**7 + (-37598400)*x**6 + (-37195200)*x**5 + (-95198400)*x**4 +
                (-153604800)*x**3 + (-100484000)*x**2 + (-26280800)*x + (-169378400))*(y**5) + (360000*x**12 + 2160000*x**11 +
                6266400*x**10 + 11532000*x**9 + 34831200*x**8 + 93892800*x**7 + 148644800*x**6 + 141984000*x**5 + 206976800*x**4 +
                275671200*x**3 + 176534800*x**2 + 48374000*x + 194042000)*(y**4) + ((-2520000)*x**12 + (-15120000)*x**11 + (-42998400)*x**10 +
                (-76392000)*x**9 + (-128887200)*x**8 + (-223516800)*x**7 + (-300675200)*x**6 + (-274243200)*x**5 + (-284547200)*x**4 +
                (-303168000)*x**3 + (-190283200)*x**2 + (-57471200)*x + (-147677600))*(y**3) + (90000*x**16 + 720000*x**15 + 3097600*x**14 +
                9083200*x**13 + 23934400*x**12 + 58284800*x**11 + 117148800*x**10 + 182149600*x**9 + 241101600*x**8 + 295968000*x**7 +
                320782400*x**6 + 276224000*x**5 + 236601600*x**4 + 200510400*x**3 + 123359200*x**2 + 43175600*x + 70248800)*(y**2) +
                ((-360000)*x**16 + (-2880000)*x**15 + (-11812800)*x**14 + (-32289600)*x**13 + (-66043200)*x**12 + (-107534400)*x**11 +
                (-148807200)*x**10 + (-184672800)*x**9 + (-205771200)*x**8 + (-196425600)*x**7 + (-166587200)*x**6 + (-135043200)*x**5 +
                (-107568800)*x**4 + (-73394400)*x**3 + (-44061600)*x**2 + (-18772000)*x + (-17896000))*y + (144400*x**18 + 1299600*x**17 +
                5269600*x**16 + 12699200*x**15 + 21632000*x**14 + 32289600*x**13 + 48149600*x**12 + 63997600*x**11 + 67834400*x**10 +
                61884000*x**9 + 55708800*x**8 + 45478400*x**7 + 32775200*x**6 + 26766400*x**5 + 21309200*x**4 + 11185200*x**3 + 6242400*x**2 +
                3465600*x + 1708800)))
    g = lambda x,y: 1e-4*(y**7 + (-3)*y**6 + (2*x**2 + (-1)*x + 2)*y**5 + (x**3 + (-6)*x**2 + x + 2)*y**4 + (x**4 + (-2)*x**3 + 2*x**2 +
                x + (-3))*y**3 + (2*x**5 + (-3)*x**4 + x**3 + 10*x**2 + (-1)*x + 1)*y**2 + ((-1)*x**5 + 3*x**4 + 4*x**3 + (-12)*x**2)*y +
                (x**7 + (-3)*x**5 + (-1)*x**4 + (-4)*x**3 + 4*x**2))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_5():
    # Test 5.1
    f = lambda x,y: 2*x*y*np.cos(y**2)*np.cos(2*x)-np.cos(x*y)
    g = lambda x,y: 2*np.sin(x*y**2)*np.sin(3*x*y)-np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-2,-2]), np.array([2,2])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_6_1():
    # Test 6.1
    f = lambda x,y: (y - 2*x)*(y+0.5*x)
    g = lambda x,y: x*(x**2+y**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_6_2():
    # Test 6.2
    f = lambda x,y: (y - 2*x)*(y+.5*x)
    g = lambda x,y: (x-.0001)*(x**2+y**2-1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_6_3():
    # Test 6.3
    f = lambda x,y: 25*x*y - 12
    g = lambda x,y: x**2+y**2-1
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_7_1():
    # Test 7.1
    f = lambda x,y: (x**2+y**2-1)*(x-1.1)
    g = lambda x,y: (25*x*y-12)*(x-1.1)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_7_2():
    # Test 7.2
    f = lambda x,y: y**4 + (-1)*y**3 + (2*x**2)*(y**2) + (3*x**2)*y + (x**4)
    h = lambda x,y: y**10-2*(x**8)*(y**2)+4*(x**4)*y-2
    g = lambda x,y: h(2*x,2*(y+.5))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_7_3():
    # Test 7.3
    c = 1.e-09
    f = lambda x,y: np.cos(x*y/(c**2))+np.sin(3*x*y/(c**2))
    g = lambda x,y: np.cos(y/c)-np.cos(2*x*y/(c**2))
    funcs = [f,g]
    a,b = np.array([-1e-9, -1e-9]), np.array([1e-9, 1e-9])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_7_4():
    # Test 7.4
    f = lambda x,y: np.sin(3*np.pi*x)*np.cos(x*y)
    g = lambda x,y: np.sin(3*np.pi*y)*np.cos(np.sin(x*y))
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_8_1():
    # Test 8.1
    f = lambda x,y: np.sin(10*x-y/10)
    g = lambda x,y: np.cos(3*x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_8_2():
    # Test 8.2
    f = lambda x,y: np.sin(10*x-y/10) + y
    g = lambda x,y: np.cos(10*y-x/10) - x
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_9_1():
    # Test 9.1
    f = lambda x,y: x**2+y**2-.9**2
    g = lambda x,y: np.sin(x*y)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_9_2():
    # Test 9.2
    f = lambda x,y: x**2+y**2-.49**2
    g = lambda x,y: (x-.1)*(x*y - .2)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

def test_roots_10():
    # Test 10.1
    f = lambda x,y: (x-1)*(np.cos(x*y**2)+2)
    g = lambda x,y: np.sin(8*np.pi*y)*(np.cos(x*y)+2)
    funcs = [f,g]
    a,b = np.array([-1,-1]), np.array([1,1])
    start = time()
    yroots = yr.solve(funcs,a,b)
    t = time() - start

    return t, resids(funcs,yroots)

if __name__ == "__main__":
    # Run all the tests!
    tests = np.array([test_roots_1_1,
                        test_roots_1_2,
                        test_roots_1_3,
                        test_roots_1_4,
                        test_roots_1_5,
                        test_roots_2_1,
                        test_roots_2_2,
                        test_roots_2_3,
                        test_roots_2_4,
                        test_roots_2_5,
                        test_roots_3_1,
                        test_roots_3_2,
                        test_roots_4_1,
                        test_roots_4_2,
                        test_roots_5,
                        test_roots_6_1,
                        test_roots_6_2,
                        test_roots_6_3,
                        test_roots_7_1,
                        test_roots_7_2,
                        test_roots_7_3,
                        test_roots_7_4,
                        test_roots_8_1,
                        test_roots_8_2,
                        test_roots_9_1,
                        test_roots_9_2,
                        test_roots_10])
    times = np.zeros_like(tests)
    avg_resids = np.zeros_like(tests)
    max_resids = np.zeros_like(tests)

    for test in tests:
        test()
    
    for i,test in enumerate(tests):
        t, residuals = test()
        avg_resid = residuals[0]
        max_resid = residuals[1]
        times[i] = t
        max_resids[i] = max_resid
        avg_resids[i] = avg_resid
    
    np.save("yroots_results/2d_times",times)
    np.save("yroots_results/2d_avg_resids",avg_resids)
    np.save("yroots_results/2d_max_resids",max_resids)

    print("Program finished successfully!")