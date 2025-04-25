import os
import multiprocessing as mp
from communotron import *

def det_sex(strain):
    if strain == 'C':
        if np.random.random() < 0.5:
            return 'M'
        else:
            return 'F'
    else:
        return 'H'

def eco_dynamic(directory, EnA, EnB, EnC, grid_size, pPred, mStep, period, Tsim, Tlim, equal_dauer_exit):
    rnd.seed()
    fec_df = pd.read_csv('./summ_fec.csv', index_col=0)
    C_fec_raw = [12.53, 22.8, 24.39, 22.32, 25.78, 22.7, 20.87, 21.36, 17.75, 10.44, 10.04, 10.81, 12.82, np.mean([12.82, 4.78]), 4.78, np.mean([4.78, 2.22]),  2.22]
    C_fec = [int(round(x)) for x in C_fec_raw]
    A_fec = list(fec_df[(fec_df['Species']=='P. pacificus') & (fec_df['Strain'] == 'RSO011')]['mean'])
    B_fec = list(fec_df[(fec_df['Species']=='P. mayeri') & (fec_df['Strain'] == 'RSO012')]['mean'])

    source_center = (grid_size//2, grid_size//2)  # Center of the source
    source_diameter = grid_size//5  # Diameter of the source
    decline_rate = 0.3  # Rate of decline
    comm = Community(grid_size, source_center, source_diameter, decline_rate, resource_cycle=period, pred_par_b=pPred, time_lim=Tlim)
    mf_probs = {'A': 1.0, 'B': 0.0, 'C': 0.0}
    if not equal_dauer_exit:
        dev_pars = {'A': {0: 48, 1:72, 4:72, 5:24}, 'B':{0: 48, 1:96, 4:96, 5:24}, 'C':{0: 48, 1:120, 4:24, 5:24}}
    else:
        dev_pars = {'A': {0: 48, 1:72, 4:24, 5:24}, 'B':{0: 48, 1:96, 4:24, 5:24}, 'C':{0: 48, 1:120, 4:24, 5:24}}
    surv_prs = {2: 500, 4: 1000, 5: 1000}
    fec_data = {'A': A_fec, 'B':B_fec, 'C':C_fec}
    age_lim = {'A' : 120, 'B': 120, 'C': 408}
    comm.add_features(mf_probs, dev_pars, surv_prs, fec_data, age_lim)
    nA = np.random.poisson(EnA)
    nB = np.random.poisson(EnB)
    nC = np.random.poisson(EnC)
    strains = np.array(['A' for i in range(nA)] + ['B' for i in range(nB)] + ['C' for i in range(nC)])
    age = np.zeros(len(strains)) 
    dev_state = np.ones(len(strains)) * 4 
    mf_state = np.array([np.random.binomial(1, mf_probs[x]) for x in strains])
    sex = np.array([det_sex(x) for x in strains])
    comm.set_max_steps(mStep)
    comm.add_strains(strains, dev_state, mf_state, age, sex)
    comm.simulate(Tsim)
    if not os.path.exists(directory):
        os.makedirs(directory)
    comm.save_state(directory)

def main(EnA, EnB, EnC, grid_size=101, pPred=0.005, mStep=10, period=300, Tsim=1000, Tlim=300, equal_dauer_exit=False):
    folder_name = 'EnA' + str(EnA) + '_EnB' + str(EnB) + '_EnC' + str(EnC) + '_size' + str(grid_size) + '_mStep' + str(mStep) + '_pPred' + str(pPred) + '_period' + str(period) + '_Tsim' + str(Tsim) + '__Tlim' + str(Tlim) + '_equal_dauer_exit' + str(equal_dauer_exit)
    directory = '/home/akalirad/pripop2/stats/april2025/' +  folder_name + '/'
    if not os.path.exists(directory):
            os.makedirs(directory)
    num_process = 5
    pool = mp.Pool(processes=num_process)
    results = [pool.apply_async(eco_dynamic, args=(directory, EnA, EnB, EnC, grid_size, pPred, mStep, period, Tsim, Tlim, equal_dauer_exit)) for x in range(num_process)]
    output = [result.get() for result in results]
main()

