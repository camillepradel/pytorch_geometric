import pandas as pd
import os
import plotly.express as px
import numpy as np
from os import listdir
import re

SUPPORTED_SETS = {
    'rgcn':'ogbn-mag',
    'gat':'Reddit',
    'gcn':'Reddit'
}

def analyse(platform) -> None:
    
    results = []
    keys = ["MODEL", "DATASET", "HYPERTHREADING", "AFFINITY", "NR_WORKERS", "TIME(s)"]
    for file in listdir(LOGS):
        test_result = [None]*len(keys)
        if ".log" in file:
            filedir=f"{LOGS}/{file}"
            model = file.split('_')[0].strip()
            #config=file.replace('_','')[:-4]
            config = file.split('_')[1][:-4].strip()
            test_result[0] = model
            test_result[1] = SUPPORTED_SETS.get(model, None)
            test_result[2] = int(re.search('HT(.*)A', config).group(1))
            test_result[3] = int(re.search('A(.*)', config).group(1))
            test_result[4] = int(re.search('W(.*)HT', config).group(1))
            test_result[5] = float(next(reversed(list(open(filedir)))).rstrip('s\n').split(":")[1].lstrip())
            results.append(test_result)
            
    table = pd.DataFrame(results, columns=keys)
    table.sort_values(by=['MODEL','NR_WORKERS','HYPERTHREADING','AFFINITY'], inplace=True)
    table.to_csv(SUMMARY, na_rep='FAILED', index_label="TEST_ID", header=True)

def plot(platform):
    
    os.makedirs(PLOTS, exist_ok=True)
    data = pd.read_csv(SUMMARY)
    data['setup'] = np.nan 
    models = ['gcn','gat','rgcn']
    datasets = ['Reddit', 'Reddit', 'ogbn-mag']
    configs = {'init':[3,512,2,16,0],
                'fin':[5,512,3,256,0]}
    cfg_items = ["<br>num_neighbors=","batch_size=","num_layers=","hidden_channels","warmup="]    
    machines = {'SPR':"2xSPR + 256GB RAM",
                'ICX':"2xICX + 512GB RAM",
                'CSX':"2xCSX + 256GB RAM"}
    
    version = platform.split('-')[1]
    platform = platform.split('-')[0]
    
    cfg_val= configs.get(version, None)
    
    for i, model in enumerate(models):
        # str formatting
        title = machines.get(platform, None)
        title += f"<br>{model}+{datasets[i]}<br>" 
        cfg = f"num_neighbors={[-1] * cfg_val[2]}, " if model !='rgcn' else f"num_neighbors={[cfg_val[0]] * cfg_val[2]}, "
        cfg += ''.join([f"{k}{val}, " for k, val in zip(cfg_items[1:], cfg_val[1:])])
        title += cfg[:-2]
        # plot data
        model_data = model_mask(data, model)
        fig = px.line(model_data, x = "NR_WORKERS", y = "TIME(s)", color = 'setup', 
                        height = 500, width = 1000,
                        labels={"Time":"TIME(s)",
                                "NR_WORKERS":"NR_WORKERS",
                                "setup":'Hyperthreading, CPU Affinity'},
                        title = title).update_traces(mode="lines+markers")
        
        fig.update_xaxes(type = 'category', categoryarray=np.unique(model_data["NR_WORKERS"]))
        avg_time_aff = model_data[(model_data['setup'] == 'NO_HT+AFF') | (model_data['setup'] == 'HT+AFF')].mean()
        avg_time_aff = round(avg_time_aff['TIME(s)'],2)
        avg_time_noaff = model_data[(model_data['setup'] == 'NO_HT+NO_AFF') | (model_data['setup'] == 'HT+NO_AFF')].iloc[1:].mean()
        avg_time_noaff = round(avg_time_noaff['TIME(s)'],2)
        fig.add_annotation(text=f"Avg. time NO_AFF*: {avg_time_noaff}s<br>Avg. time AFF: {avg_time_aff}s<br><br>* excl. nr_workers = 0", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1.25,
                    y=0.5,
                    bordercolor='white',
                    borderwidth=1)
        fig.write_image(f"{PLOTS}/{platform}-{version}-{model}.png")
         
def model_mask(data, model):
    
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 0) & (data['AFFINITY'] == 0), 'NO_HT+NO_AFF', data['setup'])) 
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 0) & (data['AFFINITY'] == 1), 'NO_HT+AFF', data['setup']))
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 1) & (data['AFFINITY'] == 0), 'HT+NO_AFF', data['setup']))
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 1) & (data['AFFINITY'] == 1), 'HT+AFF', data['setup']))
    
    return data.loc[(data['MODEL']==model)]
    
    
    
if __name__ == '__main__':
    
    platform = "SPR-init"
    
    CWD=f'pytorch_geometric/benchmark/inference/logs/{platform}'
    LOGS=f"{CWD}/logs"
    SUMMARY=f"{CWD}/summary_{platform}.csv"
    PLOTS=f'{CWD}/plots'
        
    analyse(platform)
    plot(platform)
