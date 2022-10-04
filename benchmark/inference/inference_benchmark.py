import argparse
import torch
from utils import get_dataset, get_model

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile
import os

supported_sets = {
    'ogbn-mag': ['rgat', 'rgcn'],
    'ogbn-products': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
    'Reddit': ['edge_cnn', 'gat', 'gcn', 'pna', 'sage'],
}

SOCKETS = 2
CORES = 24 #! Change this to 56??
TOTAL_CORES = SOCKETS*CORES
MODELS = {'gcn':'Reddit', 'gat':'Reddit', 'rgcn':'ogbn-mag'}
HT = ['off', 'on']

def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('BENCHMARK STARTS')
    for hyperthreading in HT:
        print(f'Setting hyperthreading: {hyperthreading.upper()}')
        ht_cmd = f'echo {hyperthreading} > /sys/devices/system/cpu/smt/control'
        test_ht_cmd = "cat /sys/devices/system/cpu/smt/active"
        os.system(ht_cmd)
        os.system(test_ht_cmd)
        
        for model_name, dataset_name in MODELS.items():
            print(f'Evaluation bench for {model_name}:')
            assert dataset_name in supported_sets.keys(
            ), f"Dataset {dataset_name} isn't supported."
            print(f'Dataset: {dataset_name}')
            dataset, num_classes = get_dataset(dataset_name, args.root,
                                            args.use_sparse_tensor, args.bf16)
            data = dataset.to(device)
            hetero = True if dataset_name == 'ogbn-mag' else False
            mask = ('paper', None) if dataset_name == 'ogbn-mag' else None
            degree = None

            inputs_channels = data[
                'paper'].num_features if dataset_name == 'ogbn-mag' \
                else dataset.num_features
            if model_name not in supported_sets[dataset_name]:
                print(f'Configuration of {dataset_name} + {model_name} '
                        f'not supported. Skipping.')
                continue
            
            for num_workers in [1,2,3]: #! [0,1,2,3,4,8,12,16,20,24,30,36,44,56]:
                for cpu_affinity in [True, False]:
                    if num_workers == 0 and cpu_affinity:
                        continue
                    cpu_aff_cores= list(range(num_workers))
                    
                    cmds = []
                    cmds.append(f"echo HYPERTHREADING: {HT}")
                    cmds.append(f"echo NR_WORKERS: {num_workers}")
                    omp_num_threads=TOTAL_CORES-num_workers
                    os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)
                    cmds.append(r"echo OMP_NUM_THREADS: $OMP_NUM_THREADS")
                    cmds.append(f"echo CPU AFFINITY: {cpu_affinity}")

                    if cpu_affinity and cpu_aff_cores:
                        if HT:
                            gomp_cpu_affinity=list(range(cpu_aff_cores[-1]+1, CORES)) + list(range(TOTAL_CORES+1,TOTAL_CORES+CORES+1))
                        else:
                            gomp_cpu_affinity=list(range(cpu_aff_cores[-1]+1, TOTAL_CORES))
                        gomp_cpu_affinity = ''.join([f"{i}, " for i in gomp_cpu_affinity])
                        gomp_cpu_affinity = gomp_cpu_affinity[:-2]
                        os.environ["GOMP_CPU_AFFINITY"] = gomp_cpu_affinity
                        cmds.append(r"echo GOMP_CPU_AFFINITY: $GOMP_CPU_AFFINITY")

                    [os.system(cmd) for cmd in cmds]

                    if torch.cuda.is_available():
                        amp = torch.cuda.amp.autocast(enabled=False)
                    else:
                        amp = torch.cpu.amp.autocast(enabled=args.bf16)
                    if not hetero:
                        subgraph_loader = NeighborLoader(
                            data,
                            num_neighbors=[-1],  # layer-wise inference
                            input_nodes=mask,
                            batch_size=args.eval_batch_sizes[0],
                            shuffle=False,
                            num_workers=num_workers,
                            use_cpu_worker_affinity=cpu_affinity,
                            cpu_worker_affinity_cores=cpu_aff_cores
                        )
                    

                    for layers in args.num_layers:
                        # limit number of neighs
                        num_neighbors = [3, 5]
                        if hetero:
                            # batch-wise inference
                            subgraph_loader = NeighborLoader(
                                data,
                                num_neighbors=num_neighbors,
                                input_nodes=mask,
                                batch_size=args.eval_batch_sizes[0],
                                shuffle=False,
                                num_workers=args.num_workers,
                            )


                        for hidden_channels in args.num_hidden_channels:
                            print('----------------------------------------------')
                            print(f'Batch size={args.eval_batch_sizes[0]}, '
                                f'Layers amount={layers}, '
                                f'Num_neighbors={num_neighbors}, '
                                f'Hidden features size={hidden_channels}, '
                                f'Sparse tensor={args.use_sparse_tensor}')
                            params = {
                                'inputs_channels': inputs_channels,
                                'hidden_channels': hidden_channels,
                                'output_channels': num_classes,
                                'num_heads': args.num_heads,
                                'num_layers': layers,
                            }

                            if model_name == 'pna':
                                if degree is None:
                                    degree = PNAConv.get_degree_histogram(
                                        subgraph_loader)
                                    print(f'Calculated degree for {dataset_name}.')
                                params['degree'] = degree

                            model = get_model(
                                model_name, params,
                                metadata=data.metadata() if hetero else None)
                            model = model.to(device)
                            model.eval()

                            with amp:
                                for _ in range(args.warmup):
                                    model.inference(subgraph_loader, device,
                                                    progress_bar=True)
                                with timeit():
                                    model.inference(subgraph_loader, device,
                                                    progress_bar=True)

                                if args.profile:
                                    with torch_profile():
                                        model.inference(subgraph_loader, device,
                                                        progress_bar=True)
                                    rename_profile_file(model_name, dataset_name,
                                                        str(batch_size),
                                                        str(layers),
                                                        str(hidden_channels),
                                                        str(num_neighbors))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GNN inference benchmark')
    argparser.add_argument('--datasets', nargs='+',
                           default=['ogbn-mag', 'ogbn-products',
                                    'Reddit'], type=str)
    argparser.add_argument(
        '--use-sparse-tensor', action='store_true',
        help='use torch_sparse.SparseTensor as graph storage format')
    argparser.add_argument(
        '--models', nargs='+',
        default=['edge_cnn', 'gat', 'gcn', 'pna', 'rgat', 'rgcn'], type=str)
    argparser.add_argument('--root', default='../../data', type=str,
                           help='relative path to look for the datasets')
    argparser.add_argument('--eval-batch-sizes', nargs='+',
                           default=[512, 1024, 2048, 4096, 8192], type=int)
    argparser.add_argument('--num-layers', nargs='+', default=[2, 3], type=int)
    argparser.add_argument('--num-hidden-channels', nargs='+',
                           default=[64, 128, 256], type=int)
    argparser.add_argument(
        '--num-heads', default=2, type=int,
        help='number of hidden attention heads, applies only for gat and rgat')
    argparser.add_argument(
        '--hetero-num-neighbors', default=10, type=int,
        help='number of neighbors to sample per layer for hetero workloads')
    argparser.add_argument('--num-workers', default=0, type=int)
    argparser.add_argument('--warmup', default=1, type=int)
    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--bf16', action='store_true')

    args = argparser.parse_args()

    run(args)
