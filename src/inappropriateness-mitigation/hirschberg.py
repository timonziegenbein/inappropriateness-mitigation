import numpy as np

def NWScore(X, Y):
    Score = np.zeros((2, len(Y) + 1), dtype=int)
    Score[0, 0] = 0
    for j in range(1, len(Y) + 1):
        Score[0, j] = Score[0, j - 1] + Ins(Y[j - 1])
    for i in range(1, len(X) + 1):
        Score[1, 0] = Score[0, 0] + Del(X[i - 1])
        for j in range(1, len(Y) + 1):
            scoreSub = Score[0, j - 1] + Sub(X[i - 1], Y[j - 1])
            scoreDel = Score[0, j] + Del(X[i - 1])
            scoreIns = Score[1, j - 1] + Ins(Y[j - 1])
            Score[1, j] = max(scoreSub, scoreDel, scoreIns)
        Score[0, :] = Score[1, :]
    LastLine = Score[1, :]
    return LastLine

def Ins(y):
    return -2

def Del(x):
    return -2

def Sub(x, y):
    return 2 if x == y else -2


def Hirschberg(X, Y):
    Z = []
    W = []
    S = []
    if len(X) == 0:
        for i in range(len(Y)):
            Z += ["-"]
            W += [Y[i]]
            S += ['<INS>']
    elif len(Y) == 0:
        for i in range(len(X)):
            Z += [X[i]]
            W += ["-"]
            S += ['<DEL>']
    elif len(X) == 1 or len(Y) == 1:
        Z, W, S = NeedlemanWunsch(X, Y)
    else:
        xlen = len(X)
        xmid = xlen // 2
        ylen = len(Y)

        ScoreL = NWScore(X[:xmid], Y)
        ScoreR = NWScore(X[xmid:][::-1], Y[::-1])
        ymid = np.argmax(ScoreL + ScoreR[::-1])

        ZL, WL, SL = Hirschberg(X[:xmid], Y[:ymid])
        ZR, WR, SR = Hirschberg(X[xmid:], Y[ymid:])
        Z = ZL + ZR
        W = WL + WR
        S = SL + SR
    return Z, W, S

def NeedlemanWunsch(X, Y):
    Z = []
    W = []
    S = []
    Score = np.zeros((len(X) + 1, len(Y) + 1), dtype=int)
    Score[0, 0] = 0
    for i in range(1, len(X) + 1):
        Score[i, 0] = Score[i - 1, 0] + Del(X[i - 1])
    for j in range(1, len(Y) + 1):
        Score[0, j] = Score[0, j - 1] + Ins(Y[j - 1])
    for i in range(1, len(X) + 1):
        for j in range(1, len(Y) + 1):
            scoreSub = Score[i - 1, j - 1] + Sub(X[i - 1], Y[j - 1])
            scoreDel = Score[i - 1, j] + Del(X[i - 1])
            scoreIns = Score[i, j - 1] + Ins(Y[j - 1])
            Score[i, j] = max(scoreSub, scoreDel, scoreIns)
    i = len(X)
    j = len(Y)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and Score[i, j] == Score[i - 1, j - 1] + Sub(X[i - 1], Y[j - 1]):
            Z = [X[i - 1]] + Z
            W = [Y[j - 1]] + W
            if Sub(X[i - 1], Y[j - 1]) == 2:
                S = ['<KEEP>'] + S
            else:
                S = ['<SUB>'] + S
            i -= 1
            j -= 1
        elif i > 0 and Score[i, j] == Score[i - 1, j] + Del(X[i - 1]):
            Z = [X[i - 1]] + Z
            W = ["-"] + W
            S = ['<DEL>'] + S
            i -= 1
        else:
            Z = ["-"] + Z
            W = [Y[j - 1]] + W
            S = ['<INS>'] + S
            j -= 1
    return Z, W, S


if __name__ == "__main__":
    X_list = ['A', 'G', 'T', 'A', 'C', 'G', 'C', 'A']
    Y_list = ['T', 'A', 'T', 'G', 'C']
    Z, W, S = Hirschberg(X_list, Y_list)
    print(Z)
    print(W)
    print(S)
    print(len([s for s in S if s != '<KEEP>']))
    X_list = ['This', 'is', 'a', 'test']
    Y_list = ['This', 'is', 'a', 'second', 'test']
    X_list = list('GOELLER')
    Y_list = list('GOWLER')
    Z, W, S = Hirschberg(X_list, Y_list)
    print(Z)
    print(W)
    print(S)
    print(len([s for s in S if s != '<KEEP>']))
    X_list = ['Business', 'review:', '1.0', 'stars', '\r\n', 'business', 'name:', 'boomtown', 'pub', '&', 'patio.', 'city:', 'calgary.', 'categories:', 'nightlife,', 'pubs,', 'restaurants,', 'tapas', 'bars,', 'gastropubs,', 'bars:', 'I', 'had', 'a', 'less', 'than', 'satisfactory', 'experience', 'at', 'Boomtown', 'Pub', '&', 'Patio', 'in', 'Calgary.', 'The', 'place', "wasn't", 'very', 'busy,', 'but', 'the', 'service', 'was', 'slow.', 'The', 'food', 'I', 'received', 'after', 'an', 'hour', 'was', 'not', 'up', 'to', 'par.', 'I', 'would', 'not', 'recommend', 'ordering', 'the', 'mushroom', 'stroganoff,', 'as', 'the', 'sauce', 'was', 'mostly', 'milk', 'and', 'the', 'mushrooms', 'were', 'too', 'tough', 'to', 'chew.', 'Our', 'group', 'of', '12', 'people', 'in', 'the', 'patio', 'did', 'not', 'have', 'a', 'great', 'time.']
    Y_list = ['Business', 'review:', '1.0', 'stars', '\r\n', 'business', 'name:', 'boomtown', 'pub', '&', 'patio.', 'city:', 'calgary.', 'categories:', 'nightlife,', 'pubs,', 'restaurants,', 'tapas', 'bars,', 'gastropubs,', 'bars:', 'Just', 'really', 'not', 'a', 'good', 'experience', 'here.', 'The', 'place', 'was', 'not', 'too', 'busy', 'and', 'everything', 'was', 'just', 'so', 'slow.', ' ', 'The', 'staff', 'was', 'overworked', 'I', 'guess.', 'The', 'food', 'when', 'I', 'came', 'out', '1', 'hour', 'later,', 'was', 'not', 'very', 'good.', ' ', "Don't", 'order', 'the', 'mushroom', 'stroganoff.', ' ', 'It', 'was', 'inedible.', 'The', 'sauce', 'was', 'just', 'milk', 'and', 'the', 'mushrooms', 'to', 'tough', 'to', 'chew.', 'We', 'had', 'a', 'group', 'of', '12', 'of', 'us', 'in', 'the', 'patio', 'and', 'it', 'was', 'really', 'not', 'fun.']
    Z, W, S = Hirschberg(X_list, Y_list)
    print(Z)
    print(W)
    print(S)
    print(len([s for s in S if s != '<KEEP>']))
