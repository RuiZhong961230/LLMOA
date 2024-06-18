from copy import deepcopy
from scipy.stats import levy
from opfunu.cec_based.cec2014 import *
import google.generativeai as genai
import os
import re


genai.configure(api_key="Your Gemini API")
model = genai.GenerativeModel("gemini-pro")


PopSize = 100
DimSize = 10
LB = [-100] * DimSize
UB = [100] * DimSize
MaxFEs = DimSize * 1000

MaxIter = int(MaxFEs / PopSize)
curIter = 0

Pop = np.zeros((PopSize, DimSize))
Off = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
FitOff = np.zeros(PopSize)

tmpFitPop = deepcopy(FitPop)
Elites = list(range(0, int(0.1 * PopSize)))
Trials = 10

def Uniform(i):
    global DimSize, Pop, FitPop, Elites, Off
    base = Pop[Elites[np.random.randint(0, int(0.1 * PopSize))]]
    for j in range(DimSize):
        Off[i][j] = base[j] + np.random.uniform(-1, 1)
    Off[i] = np.clip(Off[i], LB, UB)


def Normal(i):
    global DimSize, Pop, FitPop, Elites, Off
    base = Pop[Elites[np.random.randint(0, int(0.1 * PopSize))]]
    for j in range(DimSize):
        Off[i][j] = base[j] + np.random.normal(0, 1)
    Off[i] = np.clip(Off[i], LB, UB)


def Levy(i):
    global DimSize, Pop, FitPop, Elites, Off
    base = Pop[Elites[np.random.randint(0, int(0.1 * PopSize))]]
    for j in range(DimSize):
        Off[i][j] = base[j] + levy.rvs()
    Off[i] = np.clip(Off[i], LB, UB)


def DEbest(i):
    global DimSize, Pop, FitPop, Elites, PopSize, Off
    r1, r2 = np.random.choice(list(range(PopSize)), 2, replace=False)
    Off[i] = Pop[Elites[0]] + np.random.normal(0.5, 0.3) * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)


def DErand(i):
    global DimSize, Pop, FitPop, PopSize, Off
    r1, r2, r3 = np.random.choice(list(range(PopSize)), 3, replace=False)
    Off[i] = Pop[r1] + np.random.normal(0.5, 0.3) * (Pop[r2] - Pop[r3])
    Off[i] = np.clip(Off[i], LB, UB)


def DEcur(i):
    global DimSize, Pop, FitPop, PopSize, Off
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + np.random.normal(0.5, 0.3) * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)


def DEcur2best(i):
    global DimSize, Pop, FitPop, Elites, PopSize, Off
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + np.random.normal(0.5, 0.3) * (Pop[Elites[0]] - Pop[i]) + np.random.normal(0.5, 0.3) * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)


def DEcur2pbest(i):
    global DimSize, Pop, FitPop, Elites, PopSize, Off
    candi = list(range(PopSize))
    candi.remove(i)
    r1, r2 = np.random.choice(candi, 2, replace=False)
    Off[i] = Pop[i] + np.random.normal(0.5, 0.3) * (np.mean(Pop[Elites], axis=0) - Pop[i]) + np.random.normal(0.5, 0.3) * (Pop[r1] - Pop[r2])
    Off[i] = np.clip(Off[i], LB, UB)


def BinElite(i):
    global DimSize, Pop, FitPop, Elites, PopSize, Off
    Cr = np.random.normal(0.5, 0.3)
    jrand = np.random.randint(0, DimSize)
    base = Pop[Elites[np.random.randint(0, int(0.1 * PopSize))]]
    for j in range(DimSize):
        if np.random.rand() < Cr or j == jrand:
            Off[i][j] = base[j]
        else:
            Off[i][j] = Pop[i][j]


def BinRand(i):
    global DimSize, Pop, FitPop, PopSize, Off
    Cr = np.random.normal(0.5, 0.3)
    jrand = np.random.randint(0, DimSize)

    candi = list(range(PopSize))
    candi.remove(i)
    r1 = np.random.choice(candi, replace=False)

    for j in range(DimSize):
        if np.random.rand() < Cr or j == jrand:
            Off[i][j] = Pop[r1][j]
        else:
            Off[i][j] = Pop[i][j]


Operators = [Uniform, Normal, Levy, DEcur, DErand, DEbest, DEcur2pbest, DEcur2best, BinElite, BinRand]
Sequence = np.zeros(PopSize)


def InitialPop(func):
    global PopSize, DimSize, Pop, FitPop, Elites, Sequence, Operators, tmpFitPop
    Pop = np.zeros((PopSize, DimSize))
    FitPop = np.zeros(PopSize)
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + np.random.rand() * (UB[j] - LB[j])
        FitPop[i] = func.evaluate(Pop[i])
        Sequence[i] = np.random.randint(0, len(Operators))
    idx_sort = np.argsort(FitPop)
    Elites = idx_sort[0: int(0.1 * PopSize)]
    tmpFitPop = deepcopy(FitPop)


def HighLevel():
    global Pop, FitPop, PopSize, Sequence, tmpFitPop
    Role = "Act as a high-level component of the hyper-heuristic algorithm. "
    Instruction = "Under the background of the minimization problem, there are " + str(
        len(Operators)) + " alternative search operators and " + str(
        PopSize) + " individuals. You need to construct a optimization sequence for updating the population. "
    Context = "An example of output is: " + str(Sequence) + ", it contains " + str(
        PopSize) + " elements. Optimized by this sequence, the best fitness of parent individuals is " + str(
        min(tmpFitPop)) + " and the best fitness of offspring individuals is " + str(min(FitPop)) + " ."
    Output = "You are required to output the array-like optimization sequence. The number of element should be equal to " + str(
        PopSize) + " and the value of each element should be larger than 0 and smaller than + " + str(
        len(Operators)) + "."

    try:
        response = model.generate_content(Role + Instruction + Context + Output)
        numbers = re.findall(r'\d+', response.text)
        Sequence = [int(number) for number in numbers][0:PopSize]
    except ValueError:
        Sequence = np.random.randint(0, len(Operators), PopSize)
    if len(Sequence) < PopSize:
        Sequence = np.random.randint(0, len(Operators), PopSize)
    for i in range(len(Sequence)):
        if Sequence[i] < 0:
            Sequence[i] = 0
        elif Sequence[i] >= len(Operators):
            Sequence[i] = len(Operators) - 1
        else:
            pass


def Rand():
    global Pop, FitPop, PopSize, Sequence, tmpFitPop
    Sequence = np.random.randint(0, len(Operators), PopSize)


def LLMHHA(func):
    global PopSize, DimSize, curIter, MaxIter, Pop, FitPop, Sequence, Elites, tmpFitPop, Off, FitOff

    InitialPop(func)
    Trace = []
    for i in range(MaxIter):
        tmpFitPop = deepcopy(FitPop)
        for j in range(PopSize):
            Operator = Operators[int(Sequence[j])]
            Operator(j)
            FitOff[j] = func.evaluate(Off[j])
            if FitOff[j] < FitPop[j]:
                FitPop[j] = FitOff[j]
                Pop[j] = deepcopy(Off[j])
        idx_sort = np.argsort(FitPop)
        Elites = idx_sort[0: int(0.1 * PopSize)]
        Trace.append(min(FitPop))
        Rand()
        # HighLevel()
    return Trace


def main(dim):
    global DimSize, LB, UB, MaxFEs, MaxIter, Trials, PopSize, Pop, Off
    DimSize = dim
    LB = [-100] * dim
    UB = [100] * dim

    Pop = np.zeros((PopSize, DimSize))
    Off = np.zeros((PopSize, DimSize))
    PopSize = 100
    MaxFEs = 1000 * dim
    MaxIter = int(MaxFEs / PopSize)

    CEC2014 = [F12014(dim), F22014(dim), F32014(dim), F42014(dim), F52014(dim), F62014(dim), F72014(dim), F82014(dim),
               F92014(dim), F102014(dim), F112014(dim), F122014(dim), F132014(dim), F142014(dim), F152014(dim),
               F162014(dim), F172014(dim), F182014(dim), F192014(dim), F202014(dim), F212014(dim), F222014(dim),
               F232014(dim), F242014(dim), F252014(dim), F262014(dim), F272014(dim), F282014(dim), F292014(dim),
               F302014(dim)]

    for i in range(len(CEC2014)):
        All_Trial_Best = []
        for j in range(Trials):
            np.random.seed(2025 + 7 * j)
            Trace = LLMHHA(CEC2014[i])
            All_Trial_Best.append(Trace)
        np.savetxt("./LLMHHA_Data/CEC2014/F" + str(i + 1) + "_" + str(dim) + "D.csv", All_Trial_Best, delimiter=",")


if __name__ == "__main__":
    if os.path.exists('LLMHHA_Data/CEC2014') == False:
        os.makedirs('LLMHHA_Data/CEC2014')
    Dims = [30, 50]
    for dim in Dims:
        main(dim)
