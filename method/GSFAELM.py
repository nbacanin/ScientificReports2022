#Group Search FA (GSFA) za ELM implementaciju
#ovaj algoritam koristi inspiraciju od social network search (SNS) i od njega preuzima dispataion operator
#ovaj operator vrsi pretragu u okviru grupe resenja
#ovaj operator se izvrsava u 2 moda: prvi mod vrsi pretragu u okviru slucajnih N resenja u populaciji
#drugi mod vrsi pretragu o okviru najboljih N resenja u populaciji
#group search mod 1 metoda: disputation_early_param
#group search mod 2 metoda: disputation_late_best_param
#koristi se i parametar gss, group search start, koji odredjuje kada uopste group search pocinje, ne pocinje omdah, vec malo kasnije
#modovi se shiftuju pomocu parametra cmt - change mod trigger, koji zavisi od maxiteracija, ili evaluacija funkcija i
#cmt i gss su parametri koji se podesavaju u metodi setparams, a ovde se isto podesava pocetna vrednost za parametar gsp
#osim ovoga, koristi se jos jedan parametar gsp (group search parametar), on utica na eksploraciji i eksploataciju
#u pocetku se vise vrsi eksploracija, pa je ovaj paramtar veci, ali se tokom izvrsavanja on smanjuje dinamicki,
#jednacina: gsp = gsp-t/maxIter (na kraju se smanji za 1)
#empirijski je utvrdjeno da se dobijaju najbolja resenja ako se u svakoj iteraciji najgore resenje menja sa ovako generisanim resenje
#uzeti iz SNS rada opis disputation operatora i preformulisati
#koristi se i parametar gss, group search start, koji odredjuje kada uopste group search pocinje, ne pocinje omdah, vec malo kasnije
#gsp se menja po ovoj jednacini: self.gsp = self.gsp - (t/max_iter)



from utilities.solutionELM import Solution
import copy
import numpy as np
import sys
class GSFA:
    def __init__(self, n, function):
        
        self.N = n
        self.function = function
        self.population = []
        self.best_solution = [None] * self.function.D
        self.gamma = 1         # fixed light absorption coefficient
        self.beta_zero = 1         # attractiveness at distance zero
        self.beta = 1             
        self.alpha = 0.5         # randomization parameter
        self.gsp = 2 #parameter za group search
        self.FFE = self.N #broj FFE, u pocetku je N, posto se racuna i za fazu inicijalizaije

    #pomocna metoda za setovanje parametara
    def setParams(self,max_iter):
        self.gsp = 2 #group search parametar, dinamicki parametar
        self.gss = max_iter/2 #gss  group search start, kada uopste pocinje group search
        self.cmt = self.gss + max_iter/3 #change mode trigger, kada se menja mod sa pretrage u okviru N slucajnih na pretragu u okviru N najboljih


    def initial_population(self):
        for i in range(0, self.N):
            local_solution = Solution(self.function)
            self.population.append(local_solution)
        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = copy.deepcopy(self.population[0].x)

    def update_position(self, t, max_iter):
        
        delta = 1 - (10**(-4)/0.9)**(1/max_iter)
        self.alpha = ( 1- delta) * self.alpha
        #self.disp_param = self.disp_param - self.disp_param*(t/max_iter);
        self.gsp = self.gsp - (t/max_iter)
        
        lb = self.function.lb
        ub = self.function.ub
        
        scale = []
        for i in range(self.function.D):
            scale.append(np.abs(ub[i] - lb[i]))
        
        temp_population = self.population
        
        for i in range(self.N):
          
            for j in range(self.N):
                if self.population[i].objective_function > temp_population[j].objective_function:
                    r = np.sqrt(np.sum((np.array(self.population[i].x) - (np.array(temp_population[j].x)))**2))                    
                    beta = (self.beta - self.beta_zero) * np.exp(-self.gamma * r ** 2) + self.beta_zero
                    temp = self.alpha * (np.random.rand(self.function.D) - 0.5) * scale
                    sol = np.array(self.population[i].x) * (1 - beta) + np.array(temp_population[j].x) * beta + temp
                    sol = self.checkBoundaries(sol)
                    self.FFE = self.FFE + 1
                    solution = Solution(self.function, sol.tolist())
             
                    if solution.objective_function < self.population[i].objective_function:
                        self.population[i] = solution

        self.sort_population()

        if t>self.gss: #kada uopste pocinje group search
            if t<=self.cmt:

                solution = self.disputation_early_param(-1)
                self.FFE  = self.FFE + 1
                if solution.objective_function < self.population[-1].objective_function:
                    self.population[-1] = solution
            else:
                solution = self.disputation_best_late_param(-1)
                self.FFE = self.FFE + 1
                if solution.objective_function < self.population[-1].objective_function:
                     self.population[-1] = solution

    '''
    def disputation_best(self, i):
        # disuptation_best metoda od SNS algoritma
        # MOOD MOD DISPUTATION SNS ALGORITMA
        # X_{i,new} = X_{i} + rand(0,1) x (M-AF x X_{i})
        # M = (sum_{t}^{Nr}X_{t})/N_{r}
        # AF = 1 + round(rand)
        Xi = np.array((copy.deepcopy(self.population[i].x)))
        randomVector = np.random.random(self.function.D)
        randomVector1 = np.random.random(self.function.D)
        # random number of users (group size)
        # mora od jedan, ne mozemo manje od 1 usera u grupi da selektujemo
        Nuser = np.random.randint(1, self.N)
        # select random Nuser from network
        # users = np.random.randint(0,self.N,Nuser)
        #uzimamo Nuser najboljih korisnika, ovo je best
        users = np.array(range(0, Nuser))
        # mean vrednost Nuser parametara resenja
        M = self.calculateMean(users)
        AF = 1 + np.round(randomVector1, 0)
        # print(f"Mean values {len(M)}")
        # print(f'Function D: {self.function.D}')
        # print(len(AF))
        Xnew = Xi + randomVector * (M - AF * Xi)
        Xnew = self.checkBoundaries(Xnew)
        solution = Solution(self.function, Xnew)
        return solution
    '''
    def disputation_best_late_param(self, i):
        # disuptation_best metoda od SNS algoritma
        # MOOD MOD DISPUTATION SNS ALGORITMA
        # X_{i,new} = X_{i} + rand(0,1) x (M-AF x X_{i})
        # M = (sum_{t}^{Nr}X_{t})/N_{r}
        # AF = 1 + round(rand)
        Xi = np.array((copy.deepcopy(self.population[i].x)))
        randomVector = np.random.random(self.function.D)
        randomVector1 = np.random.random(self.function.D)
        # random number of users (group size)
        # mora od jedan, ne mozemo manje od 1 usera u grupi da selektujemo
        Nuser = np.random.randint(1, self.N)
        # select random Nuser from network
        # users = np.random.randint(0,self.N,Nuser)
        # uzimamo Nuser najboljih korisnika, ovo je best
        users = np.array(range(0, Nuser))
        # mean vrednost Nuser parametara resenja
        M = self.calculateMean(users)
        #AF = 1 + np.round(randomVector1, 0)
        AF = self.gsp + np.round(randomVector1, 0)
        # print(f"Mean values {len(M)}")
        # print(f'Function D: {self.function.D}')
        # print(len(AF))
        Xnew = Xi + randomVector * (M - AF * Xi)
        Xnew = self.checkBoundaries(Xnew)
        solution = Solution(self.function, Xnew)
        return solution

    def disputation_early_param(self, i):
        # disuptation_best metoda od SNS algoritma
        # MOOD MOD DISPUTATION SNS ALGORITMA
        # X_{i,new} = X_{i} + rand(0,1) x (M-AF x X_{i})
        # M = (sum_{t}^{Nr}X_{t})/N_{r}
        # AF = 1 + round(rand)
        Xi = np.array((copy.deepcopy(self.population[i].x)))
        randomVector = np.random.random(self.function.D)
        randomVector1 = np.random.random(self.function.D)
        # random number of users (group size)
        # mora od jedan, ne mozemo manje od 1 usera u grupi da selektujemo
        Nuser = np.random.randint(1, self.N)
        # select random Nuser from network
        users = np.random.randint(0,self.N,Nuser)

        #users = np.array(range(0, Nuser))
        # mean vrednost Nuser parametara resenja
        M = self.calculateMean(users)
        # AF = 1 + np.round(randomVector1, 0)
        AF = self.gsp + np.round(randomVector1, 0)
        # print(f"Mean values {len(M)}")
        # print(f'Function D: {self.function.D}')
        # print(len(AF))
        Xnew = Xi + randomVector * (M - AF * Xi)
        Xnew = self.checkBoundaries(Xnew)
        solution = Solution(self.function, Xnew)
        return solution


        #if t<max_iter/2:
            #self.qr()

    def calculateMean(self,user):
        #pomocna funkcija za SNS disputation
        # pomocna funkcija za racunanje mean vrednosti parametara resenja
        # prima argument niz sa integer vrednostima odabranih resenja
        #user je niz od Nuser indeksa resenja
        #meanValues je lista sa prosecnom vrednoscu parametara len(user) resenja
        #funkcja vraca niz prosecnih vrednosti parametara
        meanValues = [None] * self.function.D
        #idemo kroz parametre pa kroz usere
        for i in range(self.function.D):
            sum = 0
            for j in range(len(user)):
                sum = sum + self.population[user[j]].x[i]
            meanValues[i] = (sum/len(user))
        return np.array(meanValues)


    def sort_population(self):

        self.population.sort(key=lambda x: x.objective_function)
        self.best_solution = self.population[0].x

    def get_global_best(self):
        return (self.population[0].objective_function, self.population[0].y_proba, self.population[0].y)
        
        
        #return self.population[0].objective_function
    
    def get_global_worst(self):
        return self.population[-1].objective_function
    
    def optimum(self):
        print('f(x*) = ', self.function.minimum, 'at x* = ', self.function.solution)
        
    def algorithm(self):
        return 'FA'
    
    def objective(self):
        
        result = []
        
        for i in range(self.N):
            result.append(self.population[i].objective_function)
            
        return result
    
    def average_result(self):
        return np.mean(np.array(self.objective()))
    
    def std_result(self):        
        return np.std(np.array(self.objective()))
    
    def median_result(self):
        return np.median(np.array(self.objective()))
        
       
    def print_global_parameters(self):
            for i in range(0, len(self.best_solution)):
                 print('X: {}'.format(self.best_solution[i]))
                 
    def get_best_solutions(self):
        return np.array(self.best_solution)

    def get_solutions(self):
        
        sol = np.zeros((self.N, self.function.D))
        for i in range(len(self.population)):
            sol[i] = np.array(self.population[i].x)
        return sol


    def print_all_solutions(self):
        print("******all solutions objectives**********")
        for i in range(0,len(self.population)):
              print('solution {}'.format(i))
              print('objective:{}'.format(self.population[i].objective_function))
              print('solution {}: '.format(self.population[i].x))
              print('--------------------------------------')

    def get_global_best_params(self):
        return self.population[0].x


    def checkBoundaries(self,Xnew):
        for j in range(self.function.D):
            if Xnew[j] < self.function.lb[j]:
                Xnew[j] = self.function.lb[j]

            elif Xnew[j] > self.function.ub[j]:
                Xnew[j] = self.function.ub[j]
        return Xnew


    def getFFE(self):
        #metoda za prikaz broja FFE
        return self.FFE






    def qr(self):
    #metoda za quasi-reflextive learning

        lb = self.function.lb
        ub = self.function.ub
        qr_solution = [None] * self.function.D
        for i in range(self.N):
            for j in range(self.function.D):
                if self.population[i].x[j] < (ub[j] + lb[j]) / 2:
                    qr_solution[j] = self.population[i].x[j] + (
                            (ub[j] + lb[j]) / 2 - self.population[i].x[j]) * np.random.uniform()
                else:
                    qr_solution[j] = (ub[j] + lb[j]) / 2 + (
                            self.population[i].x[j] - (ub[j] + lb[j]) / 2) * np.random.uniform()
            qr_solution_add = Solution(self.function, qr_solution)
            self.population.append(qr_solution_add)
        self.population.sort(key=lambda x: x.objective_function)
        # delete elements from population that are not needed
        del self.population[(self.N):len(self.population)]
        # self.FFE = self.FFE + self.N  # dodajemo FFE za QRBL mechanism





