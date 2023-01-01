import numpy as np

def produto_interno (x, y):
    "Recebe dois vetores e retorna o produto interno entre eles"
    soma = 0
    for i in range(len(x)):
        soma = soma + x[i]*y[i]
    return float(soma)

def modulo (x):
    "Recebe um vetor e retorna seu módulo"
    return (produto_interno(x, x))**(1/2)

def dimensoes (x):
    "Recebe uma matriz e retorna suas dimensões"
    try:
        return (len(x[:, 0]), len(x[0, :]))
    except IndexError:
        return (1, len(x))
    
def transposta (x):
    "Recebe uma matriz e retorna sua transposta"
    (l, c) = dimensoes(x)
    t = np.zeros((c, l))
    for i in range(c):
        for j in range(l):
            t[i,j] = x[j,i]
    return t

def mult (x, y):
    "Recebe duas matrizes e retorna o produto entre elas"
    t_linhas = dimensoes(x)[0]
    t_colunas = dimensoes(y)[1]
    t = np.zeros((t_linhas, t_colunas))
    for i in range(t_linhas):
        for j in range(t_colunas):
            t[i,j] = produto_interno(x[i, :], y[:, j])
    return t

def gs(u):
    "Recebe uma matriz u de colunas LI e retona, atraves de Gram-Schmidt, uma base ortonormal que gera o mesmo espaco que u"
    l = dimensoes(u)[0]
    c = dimensoes(u)[1]
    w = np.zeros((l, c)) # w será a matriz ortonormal contendo o resultado do processo; possui as mesmas dimensões de u
    w[:, 0] = u[:, 0]
    for i in range(1, c):
        termos_anteriores = np.zeros(l) #essa variável acumula os termos anteriores nos cálculos de cada vetor de w
        for j in range(i): 
            termos_anteriores = termos_anteriores + (produto_interno(u[:, i], w[:, j])/modulo(w[:, j])**2)*w[:, j]
        w[:, i] = u[:, i] - termos_anteriores
    for i in range(c):
        w[:, i] = w[:, i]/(modulo(w[:, i]))
    return w  

def vetor_matriz (v):
    "Recebe um vetor v e retorna uma matriz coluna x com o mesmo conteudo"
    x = np.zeros((len(v), 1))
    for i in range(len(v)):
        x[i, 0] = v[i]
    return x

def sist_linear (A, b):
    "Recebe uma matriz A e um matriz coluna b e retorna x, a solução de Ax = b, atraves da fatoracao QR, e a norma do residuo"
    Q = gs(A) #aplicando o processo de ortogonalização de Gram-Schmidt em A
    R = mult(transposta(Q), A) # R = Q'* A, R sendo uma matriz triangular
    LD = mult(transposta(Q), b) # o lado direito de R*x = Q' * b
    x = np.zeros(dimensoes(R)[0])
    for i in range(1, len(x)+1):
        x[-i] = (LD[-i, 0] - produto_interno(R[-i, :], x))/R[-i,-i]
        
    x_coluna = vetor_matriz(x) #x_coluna se torna o vetor x na forma de uma matriz coluna
    residuo = modulo(b - mult(A, x_coluna))
    
    return (x, residuo)

def leitura_ex1(arquivo):
    "Recebe um arquivo no formato do Exemplo 1 e retorna a matriz A (lado esquerdo) e a matriz coluna b (lado direito)"
    with open(arquivo) as f:
        l, c = [int(x) for x in next(f).split()] #leitura da primeira linha
        v = []
        for linha in f: #leitura das demais linhas
            v.append([float(x) for x in linha.split()])
    v = np.asarray(v) #transforma a lista lida em um numpy array
    b_vetor = v[:, -1] # cria um vetor numpy com o conteudo da ultima coluna
    b = vetor_matriz(b_vetor) #cria uma matriz coluna a partir do vetor b_vetor
    A = v[:, :-1]
    return (A, b)

def leitura_ex2(arquivo):
    "Recebe o arquivo do Exemplo 2 e retorna a matriz A (lado esquerdo) e a matriz coluna y (lado direito); além disso, retorna os vetores coluna s, t e a população original y"
    with open(arquivo) as f:
        l = [int(x) for x in next(f).split()] #leitura da primeira linha
        v = []
        for linha in f: #leitura das demais linhas
            v.append([float(x) for x in linha.split()])
    v = np.asarray(v) #transforma a lista lida em um numpy array
    y_original = v[:, -1] # cria um vetor numpy com o conteudo da ultima coluna
    t = v[:, 0]
    s = (t-1950)/50
    A = transposta(np.array([np.ones(len(s)), s, s**2, s**3])) # matriz do lado esquerdo do problema de minimos quadrado
    
    # O sistema é Ax = y
    y = vetor_matriz(y_original) #transformando o vetor y numa matriz coluna para que possa ser resolvido pela função sist_linear
    return (A, y, s, t, y_original)

def leitura_ex3(arquivo):
    "Recebe um arquivo no formato do Exemplo 3 e retorna a matriz A (lado esquerdo) e a matriz coluna b (lado direito)"
    with open(arquivo) as f:
        l = [int(x) for x in next(f).split()] #leitura da primeira linha
        v = []
        for linha in f: #leitura das demais linhas
            v.append([float(x) for x in linha.split()])
    v = np.asarray(v) #transforma a lista lida em um numpy array
    y = v[:, -1] # cria um vetor numpy com o conteudo da ultima coluna
    x = v[:, 0] # cria um vetor numpy com o conteudo da ultima coluna
    
    A = np.array([x**2, x*y, y**2, x, y])
    A = transposta(A)
    b = -np.ones(10)
    b = vetor_matriz(b)
    return (A, b)

def main():
    p = 0
    while(p != 4): #Cria um menu para o usuário escolher qual programa rodar
        print("-------------------------------------------------------------------------")
        print("Digite o tipo de problema que quer resolver: ")
        print("")
        print("Digite 1 para resolver um sistema linear no molde do Exemplo 1")
        print("Digite 2 para resolver o Exemplo 2 - Crescimento Populacional")
        print("Digite 3 para resolver o Exemplo 3 - Órbita Planetária")
        print("Digite 4 para sair")
        p = int(input("Digite aqui: "))
        
        
        if p == 1:
            print("")
            print("Digite 1 para rodar o Exemplo 1")
            print("Digite 2 para rodar um sistema genérico, com a entrada nos moldes do Exemplo 1")
            print("")
            q = int(input("Digite aqui: "))
            
            if q == 1:
                (A, b) = leitura_ex1('Exemplo1.txt')
                print("")
                print("(A b): ")
                print("")
                print(np.c_[A, b])
                (solucao, residuo) = sist_linear(A, b)
                print("")
                print("Solução: ")
                print("")
                for i, x in enumerate(solucao):
                    print("x", i+1, " = ", x)
                print("")
                print("Resíduo: ")
                print("")
                print(residuo)
            
            else:
                file = input("Digite o nome do arquivo: ")
                (A, b) = leitura_ex1(file)
                print("")
                print("(A b): ")
                print("")
                print(np.c_[A, b])
                (solucao, residuo) = sist_linear(A, b)
                print("")
                print("Solução: ")
                print("")
                for i, x in enumerate(solucao):
                    print("x", i+1, " = ", x)
                print("")
                print("Resíduo: ")
                print("")
                print(residuo)
        
        elif p == 2:    
            print("")
            print("Digite 1 para rodar o Exemplo 2")
            print("Digite 2 para rodar um exemplo genérico, com a entrada nos moldes do Exemplo 2")
            print("")
            q = int(input("Digite aqui: "))
            if q == 1:
                file = 'Exemplo2.txt'
            else:
                file = input("Digite o nome do arquivo: ")
            (A, y, s, t, y_original) = leitura_ex2(file)
            (x, residuo) = sist_linear(A, y)
            def f(z):
                return x[0] + x[1]*z + x[2]*(z**2) + x[3]*(z**3)
            print("")
            print("Resíduo: ")
            print("")
            print(residuo)
            print("")
            print("t       y        yy")
            for i, v in enumerate(s):
                if y_original[i] < 100: #condicional cujo único propósito é ajeitar a apresentação da saída do print
                    print(int(t[i]), " ", y_original[i], "  ",f(s[i]).round(3))
                else:
                    print(int(t[i]), " ",y_original[i], " ",f(s[i]).round(3))
            print("")
            print("Previsão para a população em 2010:")
            ano2010 = (2010 - 1950)/50
            print("")
            print((x[0] + x[1]*ano2010 + x[2]*(ano2010**2) + x[3]*(ano2010**3)).round(3))
          
        elif p == 3:
            print("")
            print("Digite 1 para rodar o Exemplo 3")
            print("Digite 2 para rodar um exemplo genérico, com a entrada nos moldes do Exemplo 3")
            print("")
            q = int(input("Digite aqui: "))
            if q == 1:
                file = 'Exemplo3.txt'
            else:
                file = input("Digite o nome do arquivo: ")
            (A, b) = leitura_ex3(file)
            (sol, residuo) = sist_linear(A, b)
            print("")
            a = sol[0]; b = sol[1]; c = sol[2]; d = sol[3]; e = sol[4]
            print("Solução: ")
            print("a =", a)
            print("b =", b)
            print("c =", c)
            print("d =", d)
            print("e =", e)
            print("")
            print("Resíduo: ", residuo)
main()
















