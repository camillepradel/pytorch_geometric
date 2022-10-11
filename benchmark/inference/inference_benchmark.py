import argparse
import torch
from utils import get_dataset, get_model

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import PNAConv
from torch_geometric.profile import rename_profile_file, timeit, torch_profile
import os

supported_sets = {
    'rgcn':'ogbn-mag',
    'gat':'Reddit',
    'gcn':'Reddit'
}

class MemOverflow(Exception):
    pass

def run(args: argparse.ArgumentParser) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('BENCHMARK STARTS')
   
    for model_name in args.models:
        print(f'Evaluation bench for {model_name}:')
        dataset_name = supported_sets.get(model_name, None)
        dataset, num_classes = get_dataset(dataset_name, args.root,
                                        args.use_sparse_tensor, args.bf16)
        data = dataset.to(device)
        hetero = True if dataset_name == 'ogbn-mag' else False
        mask = ('paper', None) if dataset_name == 'ogbn-mag' else None

        inputs_channels = data[
            'paper'].num_features if dataset_name == 'ogbn-mag' \
            else dataset.num_features

        try:
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
                    num_workers=args.num_workers[0],
                    use_cpu_worker_affinity = (True if args.cpu_affinity == 0 else False),
                    cpu_worker_affinity_cores=list(range(args.num_workers[0]))
                )
            

            for layers in args.num_layers:
                num_neighbors = [args.hetero_num_neighbors] * layers
                if hetero:
                    # batch-wise inference
                    subgraph_loader = NeighborLoader(
                        data,
                        num_neighbors=num_neighbors,
                        input_nodes=mask,
                        batch_size=args.eval_batch_sizes[0],
                        shuffle=False,
                        num_workers=args.num_workers[0],
                        use_cpu_worker_affinity = (True if args.cpu_affinity == 0 else False),
                        cpu_worker_affinity_cores=list(range(args.num_workers[0]))
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

                    model = get_model(
                        model_name, params,
                        metadata=data.metadata() if hetero else None)
                    model = model.to(device)
                    model.eval()

                    with amp:
                        for _ in range(1):
                            try:
                                model.inference(subgraph_loader, device,
                                                progress_bar=True)
                            except RuntimeError:
                                raise MemOverflow
                            
                        with timeit():
                            try:
                                model.inference(subgraph_loader, device,
                                                progress_bar=True)
                            except RuntimeError:
                                raise MemOverflow
        except MemOverflow:
            print('RuntimeError: received 0 items of ancdata')
            print('Skip the rest of the range of nr_workers - results may be invalid.')
            continue


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
