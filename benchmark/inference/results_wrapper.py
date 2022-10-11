import pandas as pd
import os
import plotly.express as px
import numpy as np


def run(platform) -> None:
    
    file = f"/home/jpietrak/Code/projects/results_{platform}.log"

    results = []
    keys = ["MODEL", "DATASET", "HYPERTHREADING", "AFFINITY", "GOMP", "NR_WORKERS", "OMP_NUM_THREADS", "Time"]
    test_result = [None]*len(keys)

    with open(file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines() 
        for line in lines:
            
            if 'Dataset' in line:
                dataset = line.split(':')[1].strip()
            elif "Evaluation" in line:
                model = line.split(' ')[-1][:-1].strip()
            
            for i, key in enumerate(keys):
                if key in line:
                    test_result[i] = line.split(':')[1].strip()
                    if key == "Time":
                        test_result[i] = test_result[i][:-1]
                        test_result[0] = model
                        test_result[1] = dataset    
                        results.append(test_result)
                        test_result = [None]*len(keys)
    
    
    table = pd.DataFrame(results, columns=keys)
    path = f"{os.path.split(file)[0]}/summary_{platform}.csv"
    table.to_csv(path, na_rep='FAILED', index_label="TEST_ID", header=True)
    
    return path

def bar(file, platform):
    
    data = pd.read_csv(file)
    data['setup'] = np.nan 
    models = ['gcn','gat','rgcn']
    datasets = ['Reddit', 'Reddit', 'ogbn-mag']
    for i, model in enumerate(models):
        dataset = datasets[i]
        title = "2xSPR + 256GB RAM" if platform == 'SPR' else "2xICX + 512GB RAM"
        title = title + f"<br>{model}+{dataset}"
        cfg = "<br>num_neighbors=[-1], batch=256, num_layers=2, hidden_channels=64" if model!='rgcn' else "<br>num_neighbors=[5,5], batch=128, num_layers=2, hidden_channels=64"
        title = title + cfg
        model_data = model_mask(data, model)
        fig = px.bar(model_data, x = "NR_WORKERS", y = "Time", color = 'setup', 
                        barmode='group', height = 500, width = 1000, 
                        labels={"Time":"TIME(s)",
                                "NR_WORKERS":"NR_WORKERS",
                                "setup":''},
                        title = title)
        fig.update_xaxes(type = 'category', categoryarray=np.unique(model_data["NR_WORKERS"]))
        fig.write_image(f"{platform}-{model}.png")

        
                        
    
def model_mask(data, model):
    
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 'off') & (data['AFFINITY'] == 'False'), 'NO_HT+NO_AFF', data['setup'])) 
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 'off') & (data['AFFINITY'] != 'False'), 'NO_HT+AFF', data['setup']))
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 'on') & (data['AFFINITY'] == 'False'), 'HT+NO_AFF', data['setup']))
    data = data.assign(setup=np.where((data['HYPERTHREADING'] == 'on') & (data['AFFINITY'] != 'False'), 'HT+AFF', data['setup']))
    
    return data.loc[(data['MODEL']==model)]
    
    
    
if __name__ == '__main__':
    
    platform = "SPR"
    summary = run(platform)
    bar(summary, platform)
