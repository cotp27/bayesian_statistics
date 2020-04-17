# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------#
# librerias 
import pandas as pd
import numpy as np 
from scipy.stats import wishart
from numpy.linalg import inv
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import linalg
#import time

#----------------------------------------------------------------------------#
def hdp_empirico(x_var, y_var, prob):
    mejor = 0
    for ai in range(0, len(x_var)-1):
        for bi in range(0, len(x_var)):
            masa = np.sum(np.multiply(np.diff(x_var[ai:bi]), y_var[ai+1:bi]))
            #print(masa / (x_var[bi] - x_var[ai]))
            check1 = (masa >= prob) and ((masa / (x_var[bi] - x_var[ai])) > mejor) 
                          
            if (check1):
                # Se guarda el intervalo mas pequenho q cubre el 95%
                mejor = masa / (x_var[bi] - x_var[ai])
                ai_mejor = ai
                bi_mejor = bi

    result_hdp = [x_var[ai_mejor], x_var[bi_mejor]]
    return result_hdp


#----------------------------------------------------------------------------#
# Semilla para simulaciones
seed_rep = 10

#----------------------------------------------------------------------------#
# Directorio de trabajo
dir_trab = r'C:\Users\coyol\OneDrive\Escritorio\bayesian_python\py_programs\cap7'
data_miss = pd.read_csv(dir_trab  + r'\datos\pima_miss.csv', sep=',' ,index_col=0)
data_full = pd.read_csv(dir_trab  + r'\datos\pima_full.csv', sep=',' ,index_col=0)

#----------------------------------------------------------------------------#
# Parametros
n = data_miss.shape[0]
p = data_miss.shape[1]

#----------------------------------------------------------------------------# 
# Priors para theta
mu0 = np.array([120, 64, 26, 26])
sd0 = mu0/2
Lamb0 = np.ones((p,p))*0.1
np.fill_diagonal(Lamb0, 1)
Lamb0 = np.multiply(Lamb0, np.outer(sd0,sd0)) # Masa en dos d.e. cubre el 95%

# Prior para sigma
nu0 = p + 2
S0 = Lamb0

#----------------------------------------------------------------------------# 
# Se realiza el gibbs sampler
mc_samp = 1000
thetas_mcmc = []
sigmas_mcmc = []
ysamp_mcmc = {i:[] for i in range(0,n)}

#----------------------------------------------------------------------------#
# Se identifican los missing - matriz var-cov sobre matriz con info en todas las filas
bina_not_miss = data_miss.notna().to_numpy()
bina_miss = data_miss.isna().to_numpy()
#data_not_nan = data_miss[bina_not_miss.prod(axis=1)==1] 
#Sigma = data_not_nan.cov()
Sigma = S0

# Se llena la informacion de los nan con los promedios de las columnas
y_full = data_miss.fillna(data_miss.mean()).to_numpy()        # Este era un metodo comun

# Se realiza el Gibbs Sampling
np.random.seed(seed=seed_rep)

# Identificacion de filas con informacion completa
id_all_info = (np.sum(bina_not_miss, axis=1) == p)

# Gibbs sampler
for i in range(0, mc_samp):
    # Se muestrea para theta
    y_bar = np.mean(y_full, axis=0)
    lambn = inv(inv(Lamb0) + n*inv(Sigma))
    mun = np.dot(lambn, np.dot(inv(Lamb0), mu0)  + n* np.dot(inv(Sigma), y_bar))
    draw_theta = np.random.multivariate_normal(mun, lambn)    
    
    # Se genera draw para la varianza -cov
    rest_mean = (y_full - draw_theta)
    Sn = S0 + np.dot(rest_mean.T, rest_mean)
    Sigma = inv(wishart.rvs(nu0+n, inv(Sn) , size=1))
    
    # Se muestrea para el dato faltante (nan)
    for ij in range(0,n):                
        # No se reemplaza si el vector esta completo
        if id_all_info[ij]:
            continue

        else:            
            # Se extra la matri var-cov 
            a = bina_not_miss[ij]
            b = bina_miss[ij]
            inv_Sa = inv(Sigma[a][:,a])
            beta = np.dot(Sigma[b][:,a], inv_Sa)
            Sigma_ba = Sigma[b][:,b] - np.dot(beta, Sigma[a][:,b]) 
            theta_ba = draw_theta[b] + np.dot(beta, y_full[ij, a] - draw_theta[a]) 
                
            # Se simula para el dato faltante
            draw_ys = np.random.multivariate_normal(theta_ba, Sigma_ba)
            y_full[ij,b] = draw_ys
                
            # Se guarda la informacion de cada y muestreado
            res_ant = ysamp_mcmc[ij]
            res_ant.append(list(draw_ys))
            ysamp_mcmc[ij] = res_ant
        
    # Se guardan los parametros posterior
    thetas_mcmc.append(draw_theta)
    sigmas_mcmc.append(Sigma)       
    
    
#----------------------------------------------------------------------------#
# Figura 7.4
##############################################################################    
# Se construyen las matrices de correlacion
corr_mcmc = []
corr_conv = np.zeros((p,p))
for ix in range(0, mc_samp):
    Sig = sigmas_mcmc[ix]
    desv_est = inv(np.sqrt(np.diag(np.diag(sigmas_mcmc[ix]))))    
    corr = np.dot(np.dot(desv_est, Sig), desv_est)
    corr_mcmc.append(corr)                      # Se guardan todas la matrices
    corr_conv = corr_conv + corr

corr_conv_mean = np.true_divide(corr_conv, mc_samp)
#print(corr_conv_mean)

# Se convierte en array para facilitar manejo
corr_mcmc = np.array(corr_mcmc)       

# Se estima el HPD para cada correlacion
hpd_corr = {}
hpd_mean = {}
corr_name = {0:'glu-bp', 1:'glu-skin', 2:'bp-skin', 3:'glu-bmi', 4:'bp-bmi', 5:'skin-bmi'}
v_conteo = 0

for ic in range(1, p):
    iter_corr = list(range(0, ic))

    # Solo se trabaja con la diagonal inferior de la matriz de correlaciones
    for hc in iter_corr:
        list_val = []
        
        for jc in range(0, mc_samp):    
            #print(corr_mcmc[jc][ic,hc])
            list_val.append(corr_mcmc[jc][ic,hc])    
             
        # Se guardan los HPD
        kde = sm.nonparametric.KDEUnivariate(list_val)
        kde.fit()
        hpd_mean[corr_name[v_conteo]] = np.mean(list_val)
        hpd_corr[corr_name[v_conteo]] = hdp_empirico(kde.support, kde.density, 0.95) 
        v_conteo = v_conteo + 1
            
##############################################################################
# Se constuyen las distribucines de los betas - 7.4 derecha
# Hay un error en el libro. La figura 7.4 es hecha con la matriz de correlacion
# Ojo, que esto no tiene interpretacion.        
betas_mcmc = []
for jb in range(0,p):
    betas_por_samp = []
    aa = np.array(np.ones(p), dtype=bool) 
    aa[jb] = False
    bb = np.array(np.zeros(p), dtype=bool)
    bb[jb] = True  
    for ib in range(0, mc_samp):        
    #for ib in range(0, 2):        
        #Sigma_samp = sigmas_mcmc[ib]
        Sigma_samp = np.array(corr_mcmc[ib])
        Sigma_aa_inv = inv(Sigma_samp[aa][:,aa])
        beta_samp = np.dot(Sigma_samp[bb][:,aa], Sigma_aa_inv)                
        betas_por_samp.append(beta_samp)
    
    # Se guarda la informacion de los betas
    betas_mcmc.append(betas_por_samp)
        
# Se estima el HPD para cada correlacion
hpd_beta_mean = {}
hpd_beta_corr = {}
beta_name = {0:'glu', 1:'bp', 2:'skin', 3:'bmi'}
v_conteo = 1


for ic in range(0, p):
    to_nump_dat = np.array(betas_mcmc[ic])[:,0]    
    conteox = 0
    for jc in range(0, p):        
        # Solo se comparan cuando ic != jc
        if ic != jc:
            to_nump_dat1 = to_nump_dat[:,conteox]
            kde = sm.nonparametric.KDEUnivariate(to_nump_dat1)
            kde.fit()
            # Se guardan los HPD
            name_dict = beta_name[ic] + "-" + beta_name[jc]                          
            hpd_beta_mean[name_dict] = np.mean(to_nump_dat1)
            hpd_beta_corr[name_dict] = hdp_empirico(kde.support, kde.density, 0.95) 
            
            # contador
            conteox = conteox +1

##############################################################################
# Se hace el plot 7.4
x_ticks = ["glu", "bp", "skin", "bmi"]
x_ticks_mod = ["", "glu", "bp", "skin", "bmi"]  #Solo es para el grafico

fig, axs = plt.subplots(4, 2, sharex=True, sharey=True)

for iaxs in range(0, p):
    for jaxs in range(0, p):        
        if iaxs<=jaxs:
            str_cons = x_ticks[iaxs] + "-" + x_ticks[jaxs]
        else:
            str_cons = x_ticks[jaxs] + "-" + x_ticks[iaxs]

        if iaxs != jaxs:
            x_dom = jaxs+1
            axs[iaxs,0].errorbar(x=x_dom, y=hpd_mean[str_cons], 
                         yerr=[[hpd_mean[str_cons]-hpd_corr[str_cons][0]], 
                               [hpd_corr[str_cons][1]-hpd_mean[str_cons]]], 
                         color="black", marker="o", markersize=2, capsize=1, 
                         linestyle="None", mfc="black", mec="black")

            axs[iaxs,1].errorbar(x=x_dom, y=hpd_beta_mean[str_cons], 
                         yerr=[[hpd_beta_mean[str_cons]-hpd_beta_corr[str_cons][0]], 
                               [hpd_beta_corr[str_cons][1]-hpd_beta_mean[str_cons]]], 
                         color="black", marker="o", markersize=2, capsize=1, 
                         linestyle="None", mfc="black", mec="black")
            
            
            axs[iaxs,0].set_ylabel(x_ticks[iaxs])

plt.setp(axs, xticks=[0.8, 1, 2, 3, 4], xticklabels=x_ticks_mod, yticks=[0, 0.33, 0.66, 1])


#----------------------------------------------------------------------------#
##############################################################################
# Figura 7.5
y_predict = data_miss.copy()

for y_i in range(0,n):
    if id_all_info[y_i]:    
        continue
    else:
        # Se predice con el promedio de cada prediccion
        pred_med = np.mean(ysamp_mcmc[y_i], axis=0)
        y_predict.loc[y_i, bina_miss[y_i]] = pred_med
    
# Se realiza el grafico
axs_names = {0:'glu', 1:'skin', 2:'bp', 3:'bmi'}
y_predict = y_predict.rename(columns={'glu':'glu_pred', 'bp':'bp_pred', 'skin':'skin_pred', 'bmi':'bmi_pred'})
concatenated = pd.concat([data_full.assign(dataset='set1'), y_predict.assign(dataset='set2')], axis=1)


fig1, axs1 = plt.subplots(2, 2)
for i_plot, ax in zip(range(10), axs1.flat):
    data_rep_miss = concatenated[bina_miss[:,i_plot]]
    plt1 = sns.scatterplot(x=axs_names[i_plot], y=axs_names[i_plot] + "_pred", data=data_rep_miss, ax=ax)
    x0, x1 = plt1.get_xlim()
    y0, y1 = plt1.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    plt1.plot(lims, lims, ':k')




