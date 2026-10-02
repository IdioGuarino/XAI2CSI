
import time
import argparse
import os
import json
from tqdm import tqdm
from pathlib import Path
import importlib
from box import Box
from networks.sm_network import SMNet as Model 
import torch
from torch.nn.functional import cross_entropy
from libs.data_loader import get_data_processor
from libs.utils import Logger, make_args_dict, seed_everything, format_inputs, get_elapsed, wrapperScheduler, plot_loss
from sklearn.metrics import accuracy_score, f1_score
from libs.data_formatter import dataset_config, get_dataset_setup
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
def main(ts_campaing, network, dataset, exp_name, results_path, device, **kwargs):


    args_path=os.path.join(exp_main_path, 'args-'+str(campaign_ts)+'.txt')
    args= make_args_dict(
        device=device,
        network=network,
        dataset=dataset,
        exp=exp_name,
        output=results_path,
        **kwargs
    )

    


def parse_arguments():
    parser = argparse.ArgumentParser(description="CSI-HAR Training")

    # Argument definitions
    parser.add_argument('-o', '--output', required=True, help="Output directory for results")

    #Exp Settings
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda', help="Device to run on (default: cuda)")
    parser.add_argument('-N', '--network', choices=['CNN2D', 'INET'], default='CNN2D', help="Network type (default: CNN2D)")
    parser.add_argument('-D', '--dataset', choices=['cominelli23'], default='cominelli23', help="Dataset name")
    parser.add_argument('-E', '--early', action='store_true', help="To enable loss-based early stopping on validation set")
    parser.add_argument('-p', '--patience', type=int, default=10, help="Patience for early stopping (default: 10)")
    # parser.add_argument('-l', '--lr', type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument('-e', '--epochs', type=int, default=200, help="Number of epochs (default: 200)")
    parser.add_argument('-n', '--exp', default='XAI4CSI', help="Experiment name (default: XAI4CSI)")
    parser.add_argument('-x', '--scheduler', choices=['RoP', 'Fixed', 'Cosine'], default='RoP', help="Scheduler name (default: RoP)")
    parser.add_argument('-r', '--seed', type=int, default=46, help="Random seed (default: 46)")
    parser.add_argument('-z', '--out_features_size', type=int, default=128, help="Output features size (default: 128)")

    parser.add_argument('-O', '--target_fold', type=int, default=0, help="Target fold (default: 0)")
    parser.add_argument('-F', '--num_folds', type=int, default=10, help="Number of folds (default: 10)")
    parser.add_argument('-S', '--all_folds', action='store_true', help="Experiment on all folds")
    parser.add_argument('-c', '--cross', type=str, default=None, choices=['users', 'environments', 'rx', 'day'], help="Cross-evaluation (Scenario) strategy")
    parser.add_argument('-t', '--target_scenario', type=str, default=None, help="Target scenario for cross-evaluation")
    parser.add_argument('-T', '--temporal', action='store_true', help="Temporal splitting strategy [default: False, it disables k-folds]")
    parser.add_argument('-vs', '--val_size', type=float, default=0.2, help="Validation size for both Temporal and k-folds splitting [default: 0.2]")
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, help="Test size for Temporal splitting [default: 0.2]")
    parser.add_argument('-ps', '--post_scaling', action='store_true', help="Apply post-scaling to data after filtering on scenarios in cross-evaluation mode [default: False]")
    parser.add_argument('-f', '--class_filter', type=str, default=None, nargs='*', help="Class to filter the dataset (e.g., 'Walking, Sitting')")


    parser.add_argument('-op', '--optimizer', type=str, default='SGD', help="Optimizer to use [default: SGD]", choices=['SGD', 'Adam'])
    parser.add_argument('-sc', '--scaler', type=str, default='MMS', help="Scaler to use [default: MMS]", choices=['MMS', 'SS'])
    # Data settings
    parser.add_argument('-A', '--amplitude', action='store_true', help="Use amplitude data")
    parser.add_argument('-P', '--phase', action='store_true', help="Use phase difference data")
    parser.add_argument('-w', '--window_size', type=int, default=50, help="Window size (default: 50)")
    parser.add_argument('-s', '--stride', type=int, default=25, help="Stride size (default: 25)")

    parser.add_argument('-W', '--wifi', type=str, default='ax', choices=['ax', 'ac'],help="Number of channels (default: ax)")
    parser.add_argument('-C', '--bandwidth', type=int, default=80, help="Channel bandwidth (default: 80)")
    
    return parser.parse_args()



if __name__ == '__main__':
    
    t_args = parse_arguments()

    if t_args.cross is not None:
        if t_args.target_scenario is None:
            if t_args.cross == 'users':
                scenarios = ['S1', 'S2', 'S3']
            elif t_args.cross == 'environments':
                scenarios = ['S5', 'S6', 'S7']
            elif t_args.cross == 'rx':
                scenarios = ['Rx1', 'Rx2', 'Rx3']
            elif t_args.cross == 'day':
                scenarios = ['S1', 'S5']
        else:
            scenarios = [t_args.target_scenario]
    else:
        scenarios = [None]
    # Handle device choice
    if 'cuda' in t_args.device and not torch.cuda.is_available():
        print('CUDA is not available, falling back to CPU.')
        device = torch.device('cpu')
    else:
        device = torch.device(t_args.device)
        # torch.cuda.empty_cache()
        # gc.collect()
    print(f'Running on: {device}')
    verbose = False
    print("filtering classes: ", t_args.class_filter)
    train_eval=False
    network = t_args.network
    dataset = t_args.dataset
    exp_name = t_args.exp
    results_path = t_args.output

    s_exp_name=f"{dataset}_scratch_{exp_name}"
    exp_main_path=os.path.join(results_path, s_exp_name)
    Path(exp_main_path).mkdir(parents=True, exist_ok=True)

    t_args.split_strategy = 'temporal' if t_args.temporal or t_args.cross is not None else 'stratified'
    if t_args.temporal:
        t_args.num_folds = 1
        t_args.target_fold = 0

    t_args.antennas=dataset_config[dataset]['antennas']
    t_args.subcarriers=get_dataset_setup(dataset, t_args.wifi, t_args.bandwidth)['n_subcarriers']-len(get_dataset_setup(dataset, t_args.wifi, t_args.bandwidth)['subcarriers_filter'])

    if t_args.optimizer=='SGD':
        t_args.lr=0.1

        t_args.mode="min"
        t_args.patience=10
        t_args.lr_factor=0.3
        t_args.lr_min=0.0001
        t_args.monitor="val_loss"

    elif t_args.optimizer=='Adam':
        t_args.lr = 1e-3

        t_args.mode="min"
        t_args.patience=10
        t_args.lr_factor=0.1
        t_args.lr_min=1e-6
        t_args.monitor="val_loss"

    for scenario in scenarios:
        # if scenario in ['S5', 'S6']:
        #     continue
        t_args.target_scenario = scenario
        campaign_ts=int(time.time()) 
        stdout_path=os.path.join(exp_main_path, 'stdout-'+str(campaign_ts)+'.txt')
        # sys.stdout = Logger(stdout_path)



        with Logger(stdout_path):
            models_path=os.path.join(exp_main_path, 'models')
            Path(models_path).mkdir(parents=True, exist_ok=True)

            t_epochs_path=os.path.join(exp_main_path, 'epochs_training')
            Path(t_epochs_path).mkdir(parents=True, exist_ok=True)

            v_epochs_path=os.path.join(exp_main_path, 'epochs_validation')
            Path(v_epochs_path).mkdir(parents=True, exist_ok=True)

            model_path = os.path.join(models_path, "task0-%d.ckpt"%(campaign_ts))  



            results_path=os.path.join(exp_main_path, 'results')
            Path(results_path).mkdir(parents=True, exist_ok=True)
            args_path=os.path.join(exp_main_path, 'args-'+str(campaign_ts)+'.txt')
            with open(args_path, 'w') as f:
                json.dump(vars(t_args), f, indent=4)

            args=Box(vars(t_args))  
            seed_everything(seed=args.seed)

            print("filtering classes: ", args.class_filter)
            trn_loader, val_loader, tst_loader = None, None, None
            trn_dset, val_dset, tst_dset, class_order = get_data_processor(
                dataset_name=dataset,
                wifi=args.wifi, channel_bandwidth=args.bandwidth,
                window_size=args.window_size, stride=args.stride,
                split_strategy=args.split_strategy, val_size=args.val_size, test_size=args.test_size, post_scaling=args.post_scaling,
                seed=args.seed,
                num_folds=args.num_folds, target_fold=args.target_fold,
                target_scenario=args.target_scenario, cross=args.cross,
                amplitude=args.amplitude, phase=args.phase, scaler=args.scaler, target_classes=args.class_filter
            )


            trn_loader = DataLoader(trn_dset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False)
            val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False) if val_dset is not None else None
            tst_loader = DataLoader(tst_dset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=False)

            Net = getattr(importlib.import_module(name='networks'), network)
            model = Net(antennas=args.antennas, subcarriers=args.subcarriers, window_size=args.window_size,
                            out_features_size=args.out_features_size, num_classes=len(class_order),
                            dataset=args.dataset, amplitude=args.amplitude, phase=args.phase)
            
            model.to(device)

            kwargs = {
                        "mode": args.mode,
                        "patience": args.patience,
                        "lr_factor": args.lr_factor,
                        "lr_min": args.lr_min,
                        "monitor": args.monitor,
                        "verbose": True
                    }
            
            if args.optimizer=='SGD':
                optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
            elif args.optimizer=='Adam':
                optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

            if verbose:
                if args.phase and args.amplitude:
                    print("Using only phase data, adjusting input size for summary.")
                    nchannels= args.antennas + args.antennas-1
                elif args.amplitude :
                    print("Using only amplitude data, adjusting input size for summary.")
                    nchannels=args.antennas
                else:
                    print("Using only phase data, adjusting input size for summary.")
                    nchannels=args.antennas-1
                summary(model, input_size=(nchannels, args.subcarriers, args.window_size))

            seed_everything(seed=args.seed)
            print(f"EXP: {campaign_ts}")

            t_start = time.time()
            class_loss = cross_entropy


            scheduler = wrapperScheduler(model=model, optimizer=optimizer, scheduler_name=args.scheduler, **kwargs)
            
            do_stop, is_best= False, False

            for epoch in range(args.epochs):
                model.train()
                total_loss, val_loss, trn_samples, val_samples=0, 0, 0, 0
                
                for batch, labels in tqdm(trn_loader, total=len(trn_loader), unit='batch', colour='green'):
                    
                    batch, labels = format_inputs(device, batch, labels)

                    scheduler.optimizer.zero_grad()

                    logits = model(batch)  # Recupero logits della prima testa
                        
                    fine_tuning_loss = class_loss(logits, labels)
                    loss = fine_tuning_loss

                    loss.backward()  # Computes gradients
                    scheduler.optimizer.step()  # Progates gradients
                    total_loss += loss.item()*len(labels)
                    trn_samples+=len(labels)

                if train_eval:
                    model.eval()
                    y_trues, y_preds = [], []
                    with torch.no_grad(): 
                        for batch, labels in trn_loader:       
                            batch, labels = format_inputs(device, batch, labels)
                            
                            logits = model(batch)  # Recupero logits della prima testa
                            y_trues.extend(labels.cpu().numpy())
                            y_preds.append(logits)
                    y_preds = torch.cat(y_preds)
                    y_preds = torch.argmax(y_preds, dim=1).cpu().numpy()
                    trn_acc = accuracy_score(y_trues, y_preds)*100
                else:
                    trn_acc = 0.0
                model_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": scheduler.optimizer.state_dict(), 
                    "loss": loss, 
                    "total_loss": total_loss,
                    "total_loss_bathes": trn_samples,
                    "trn_accuracy": trn_acc,
                    }

                e_model_path = os.path.join(t_epochs_path, "task0-%d-%d.tar"%(epoch,campaign_ts))
                torch.save(model_dict, e_model_path)

                model.eval()
                y_trues, y_preds = [], []
                with torch.no_grad(): 
                    for batch, labels in val_loader:       
                        batch, labels = format_inputs(device, batch, labels)
                        
                        logits = model(batch)  # Recupero logits della prima testa
                            
                        loss = class_loss(logits, labels)
                        val_loss += loss.item()*len(labels)
                        val_samples+=len(labels)
                        
                        y_trues.extend(labels.cpu().numpy())
                        y_preds.append(logits)
                y_preds = torch.cat(y_preds)
                y_preds = torch.argmax(y_preds, dim=1).cpu().numpy()
                val_acc = accuracy_score(y_trues, y_preds)*100

                do_stop, is_best=scheduler.update(val_loss/val_samples)
                
                char_el='*' if is_best else ' '

                print(f'Epoch [{epoch+1}/{args.epochs}], Train loss: {total_loss/trn_samples} Train.Acc: {trn_acc:.2f} - Val. loss: {val_loss/val_samples} Val.Acc: {val_acc:.2f} {char_el}')

                model_dict = {
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": scheduler.optimizer.state_dict(), 
                    "loss": loss, 
                    "val_loss": val_loss,
                    "val_loss_bathes": val_samples,
                    "val_accuracy": val_acc,
                    "do_stop": do_stop,
                    "is_best": is_best
                    }

                e_model_path = os.path.join(v_epochs_path, "task0-%d-%d.tar"%(epoch,campaign_ts))
                torch.save(model_dict, e_model_path)

                e_model_path = os.path.join(t_epochs_path, "task0-%d-%d.pkt"%(epoch,campaign_ts))
                torch.save(model, e_model_path)

                if do_stop:
                    model.load_state_dict(scheduler.best_model)
                    break
                if epoch==args.epochs-1:
                    model.load_state_dict(scheduler.best_model)
                    break
                # model.train()    


            torch.save(model,model_path) 

            h, mn, s=get_elapsed(t_start)
            print(f"Experiments ended in {h} hours and {mn} minutes.")

            model.eval()
            y_trues, y_preds = [], []
            with torch.no_grad(): 
                for batch, labels in trn_loader:       
                    batch, labels = format_inputs(device, batch, labels)
                    
                    logits = model(batch)  # Recupero logits della prima testa
                    y_trues.extend(labels.cpu().numpy())
                    y_preds.append(logits)
            y_preds = torch.cat(y_preds)
            y_preds = torch.argmax(y_preds, dim=1).cpu().numpy()
            trn_acc = accuracy_score(y_trues, y_preds)*100
            trn_fsc = f1_score(y_trues, y_preds, average='macro')*100
            print(f'[{campaign_ts}] Train accuracy: {trn_acc:.2f}| Train F1 Score: {trn_fsc:.2f}.\n')

            y_trues, y_preds = [], []
            with torch.no_grad(): 
                for batch, labels in tst_loader:       
                    batch, labels = format_inputs(device, batch, labels)
                    logits = model(batch)  # Recupero logits della prima testa
                    y_trues.extend(labels.cpu().numpy())
                    y_preds.append(logits)
            y_preds = torch.cat(y_preds)
            y_preds = torch.argmax(y_preds, dim=1).cpu().numpy()
            tst_acc = accuracy_score(y_trues, y_preds)*100
            tst_fsc = f1_score(y_trues, y_preds, average='macro')*100
            print(f'[{campaign_ts}] Test accuracy: {tst_acc:.2f}| Test F1 Score: {tst_fsc:.2f}.\n')

            trn_dset.close()
            val_dset.close()
            tst_dset.close()

            plot_loss(exp_main_path, str(campaign_ts), normalize=True, task=0)
