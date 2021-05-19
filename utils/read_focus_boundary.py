import numpy as np
def read_focus_boundary(filename):
    """Read FOCUS/FAMUS plasma boundary file
    Args:
        filename (str): File name and path.
    Returns:
        boundary (dict): Dict contains the parsed data.
            nfp : number of toroidal periods
            nfou : number of Fourier harmonics for describing the boundary
            nbn : number of Fourier harmonics for Bn
            surface : Toroidal surface dict, containing 'xm', 'xn', 'rbc', 'rbs', 'zbc', 'zbs'
            bnormal : Input Bn dict, containing 'xm', 'xn', 'bnc', 'bns'
    """
    boundary = {}
    surf = {}
    bn = {}
    with open(filename, "r") as f:
        line = f.readline()  # skip one line
        line = f.readline()
        num = int(line.split()[0])  # harmonics number
        nfp = int(line.split()[1])  # number of field periodicity
        nbn = int(line.split()[2])  # number of Bn harmonics
        boundary["nfp"] = nfp
        boundary["nfou"] = num
        boundary["nbn"] = nbn
        # read boundary harmonics
        xm = []
        xn = []
        rbc = []
        rbs = []
        zbc = []
        zbs = []
        line = f.readline()  # skip one line
        line = f.readline()  # skip one line
        for i in range(num):
            line = f.readline()
            line_list = line.split()
            n = int(float(line_list[0]))
            m = int(float(line_list[1]))
            xm.append(m)
            xn.append(n)
            rbc.append(float(line_list[2]))
            rbs.append(float(line_list[3]))
            zbc.append(float(line_list[4]))
            zbs.append(float(line_list[5]))
        surf["xm"] = np.array(xm)
        surf["xn"] = np.array(xn)
        surf["rbc"] = np.array(rbc)
        surf["rbs"] = np.array(rbs)
        surf["zbc"] = np.array(zbc)
        surf["zbs"] = np.array(zbs)
        boundary["surface"] = surf
        # read Bn fourier harmonics
        xm = []
        xn = []
        bnc = []
        bns = []
        if nbn > 0:
            line = f.readline()  # skip one line
            line = f.readline()  # skip one line
        for i in range(nbn):
            line = f.readline()
            line_list = line.split()
            n = int(float(line_list[0]))
            m = int(float(line_list[1]))
            xm.append(m)
            xn.append(n)
            bnc.append(float(line_list[2]))
            bns.append(float(line_list[3]))
        bn["xm"] = np.array(xm)
        bn["xn"] = np.array(xn)
        bn["bnc"] = np.array(bnc)
        bn["bns"] = np.array(bns)
        boundary["bnormal"] = bn
    return boundary
