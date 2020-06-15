# -*- coding: utf-8 -*-
# Librerias
import pandas as pd
import numpy as np
import seaborn as sns

# Semilla
np.random.seed(50)

# Datos
dir_trab = r'C:\Users\coyol\Desktop\bayesian_python\py_programs'
data_all = pd.read_csv(dir_trab  + r'\datos\sparrows.csv', sep=',' ,index_col=0)
age = data_all['age'].to_numpy()
fledged = data_all['fledged'].to_numpy()


# ---------------------------------------------------------------------------#
# Fig. 10.1
sns.set(style="whitegrid")
sns.boxplot(x="age", y="fledged", data=data_all)

