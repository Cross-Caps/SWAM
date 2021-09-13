import numpy as np
from numpy import linalg as LA
import yaml
from scipy.io import savemat
from datetime import datetime
import json
import copy

class DictionaryMetrics:
    def __init__(self, D, gap = None, rad = None):
        self.D = D
        self.D = self.D/np.sqrt(np.sum(self.D**2, axis = 0, keepdims = True))
        self.gap = gap
        if self.gap is None:
            self.gap = True
        
        self.rad = rad
        if self.rad is None:
            self.rad = False

        if self.rad:
            self.angle = 'radians'
            self.angleFactor = 1
        else:
            self.angle = 'degrees'
            self.angleFactor = 180/np.pi
        
        self.N , self.K = D.shape

        self.nNaN = int(np.sum(np.isnan(D)))
        self.nInf = int(np.sum(np.isinf(D)))
        self.nNZ = int(np.sum(D != 0))
        
        self.lambd = None
        self.A = None
        self.B = None

        self.betamin = None
        self.betaavg = None
        self.betamse = None

        self.mu = None
        self.muavg = None
        self.mumse = None

        self.babel = None

        self.betagap = None
        self.mugap = None
        self.rgap = None
        self.xgap = None

        self.decay = None

        self.propagate()

    def propagate(self):
        if self.nNaN == 0 and self.nInf == 0:
            eigvals, eigvecs = LA.eig(self.D @ self.D.T)
            I = np.argsort(eigvals)[::-1]
            self.lambd = np.take_along_axis(eigvals, I, axis = 0)
            eigvecs = np.take_along_axis(eigvecs, np.vstack([I for _ in range(eigvecs.shape[1])]), axis = 1)
            self.B = float(self.lambd[0])
            Ia = self.lambd.shape[0] - 1
            while(self.lambd[Ia] < 1e-12):
                Ia -= 1
            self.A = float(self.lambd[Ia])
            self.mumse = float((np.sum(self.lambd**2)**0.5)/self.K)
            self.betamse = float(np.arccos(self.mumse) * self.angleFactor)
            Ga = np.abs(self.D.T@self.D - np.eye(self.K))
            self.mu = float(np.max(Ga))
            self.betamin = float(np.arccos(self.mu) * self.angleFactor)
            self.betaavg = float(np.mean(np.arccos(np.max(Ga, axis = 0)) * self.angleFactor))
            self.muavg = float(np.mean(np.max(Ga, axis = 0)))
            IGa = np.argsort(Ga, axis = 0)[::-1]
            Ga = np.take_along_axis(Ga, IGa, axis = 0)
            self.babel = np.zeros((self.K - 1, 1))
            for k in range(1, self.K):
                self.babel[k-1][0] = np.max(Ga[0, :])
                Ga[0, :] = Ga[0, :] + Ga[k, :]
            
            if self.gap:
                if self.N <= 4:
                    keep = 20
                    L = 2000
                    X = np.randn(self.N, L)
                else:
                    L = min(10*self.N, 2000)
                    if self.B/self.A > 10:
                        keep = min(15, max(int(self.N/2),4))
                        Lini = min(5, max(2,int(self.N/4)))
                    else:
                        keep = min(30, max(int(self.N/2),4))
                        Lini = min(12, max(2,int(self.N/4)))
                    if Ia > Lini:
                        X = eigvecs[:, (Ia-Lini) : Ia] @ np.random.randn(Lini, L)
                    else:
                        X = eigvecs[:, 0:Ia] @ np.random.randn(Ia, L)
                X = X / np.sqrt(np.sum(X**2, axis = 0, keepdims = True))
                self.mugap = 1
                count = 0
                for _ in range(20000):
                    Ga = np.abs(self.D.T @ X)
                    I = np.argsort(np.max(Ga, axis = 0))
                    mga = np.take_along_axis(np.max(Ga, axis = 0), I, axis = 0)
                    if mga[0] > self.mugap:
                        count += 1
                        if count > 10:
                            break
                    else:
                        self.mugap = mga[0]
                        self.xgap = X[:, 0]
                    X[:, keep:L] = X[:, I[:keep]] @ np.random.randn(keep, L - keep)
                    X = X / np.sqrt(np.sum(X**2, axis = 0, keepdims = True))
                self.mugap = float(self.mugap)
                self.betagap = float(np.arccos(self.mugap) * self.angleFactor)
                self.decay = float(self.mugap**2)
                self.rgap = float(np.sqrt(1 - self.decay))

        else:
            print("WARNING: There are nan and inf values in your dictionary, setting all metrics to nan")
            self.lambd = np.ones(self.N) * float("nan")
            self.A = float("nan") 
            self.B = float("nan")
            self.betamse = float("nan")
            self.betamin = float("nan") 
            self.betaavg = float("nan")
            self.mu = float("nan")
            self.muavg = float("nan")
            self.mumse = float("nan")
            self.babel = np.ones(self.K-1) * float("nan")
            if self.gap:
                self.mugap = float("nan")
                self.rgap = float("nan")
                self.decay = float("nan")
                self.betagap = float("nan")
                self.xgap = np.ones(self.N) * float("nan")
    
    def toDict(self, ro = 4):
        if ro:
            formatted_dict = {
                                "Dictionary": self.D,
                                "input_params": {
                                        "gap": self.gap,
                                        "rad": self.rad,
                                        "angle": self.angle,
                                        "angle_factor": self.angleFactor,
                                        "Number of nans": self.nNaN,
                                        "Number of infs": self.nInf,
                                        "Number of non-zeros": self.nNZ
                                    },
                                "metrics":{
                                        "lambda": self.lambd, 
                                        "A":round(self.A, ro), 
                                        "B": round(self.B, ro), 
                                        "betamse": round(self.betamse, ro),
                                        "betamin": round(self.betamin, ro),
                                        "betaavg": round(self.betaavg, ro),
                                        "mu": round(self.mu, ro),
                                        "muavg": round(self.muavg, ro),
                                        "mumse": round(self.mumse, ro),
                                        "babel": self.babel,
                                        "mugap": round(self.mugap, ro),
                                        "betagap": round(self.betagap, ro),
                                        "rgap": round(self.rgap, ro),
                                        "decay": round(self.decay, ro),
                                        "xgap": self.xgap
                                    }
                            }
        else:
            formatted_dict = {
                                "Dictionary": self.D,
                                "input_params": {
                                        "gap": self.gap,
                                        "rad": self.rad,
                                        "angle": self.angle,
                                        "angle_factor": self.angleFactor,
                                        "Number of nans": self.nNaN,
                                        "Number of infs": self.nInf,
                                        "Number of non-zeros": self.nNZ
                                    },
                                "metrics":{
                                        "lambda": self.lambd, 
                                        "A":self.A, 
                                        "B": self.B, 
                                        "betamse": self.betamse,
                                        "betamin": self.betamin,
                                        "betaavg": self.betaavg,
                                        "mu": self.mu,
                                        "muavg": self.muavg,
                                        "mumse": self.mumse,
                                        "babel": self.babel,
                                        "mugap": self.mugap,
                                        "betagap": self.betagap,
                                        "rgap": self.rgap,
                                        "decay": self.decay,
                                        "xgap": self.xgap
                                    }
                            }
        return formatted_dict
        
    def __str__(self):
        formatted_dict = copy.deepcopy(self.toDict())
        formatted_dict.pop("Dictionary")
        formatted_dict["metrics"].pop("lambda")
        formatted_dict["metrics"].pop("xgap")
        formatted_dict["metrics"].pop("babel")
        return yaml.dump(formatted_dict, default_flow_style = False, sort_keys = False)
    
    def save(self, name = "metrics_{}".format(datetime.now().strftime(f"%d:%m:%Y_%H:%M:%S"))):
        formatted_dict = copy.deepcopy(self.toDict())
        name = name + ".json" if name[-5:]!=".json" else name
        json.save(name, formatted_dict)
