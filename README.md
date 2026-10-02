# XAI2CSI

This repository contains code for **XAI2CSI: Interpreting CSI with eXplainable AI for Human Activity Recognition**.

`main.py` trains a CSI-based activity-recognition model, evaluates it on the training and test splits, and writes model checkpoints, logs, arguments, and loss plots to the requested output directory.

## Setup

Use Python 3 and install the packages listed in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

The default dataset is `cominelli23`. The loader expects its data under `data/cominelli23/`, including `ground_truth.json`. Ensure the dataset files are in place and formatted as expected by the data loader before starting an experiment.

## Run and evaluation modes

The output directory is required. Replace each `<PLACEHOLDER>` in the examples with a real path or name before running the command. Keep paths containing spaces quoted. The class labels passed to `-f` must match labels in the selected dataset; the labels shown below illustrate the argument format.

### Upper-bound (no cross-evaluation)

When `-c` / `--cross` is omitted, the framework uses all available users, environments, receivers, and days; it does not hold out one of these domains for cross-evaluation. This is the upper-bound run:

```bash
python3 main.py \
  -o "<OUTPUT_DIRECTORY>" \
  -W ax -C 80 \
  -w 50 -s 25 \
  -T \
  -n "<EXPERIMENT_NAME>" \
  -d cuda:1 \
  -N CNN2D \
  -op SGD \
  -A -P \
  -f Walk Run Jump Sitting Empty
```

### Cross-evaluation

Set `-c` / `--cross` to select the domain for cross-evaluation. If `-t` / `--target_scenario` is omitted, the script automatically performs leave-one-scenario-out cross-evaluation: it holds out one target scenario at a time from training and creates one model per target scenario. Thus the number of models is the number of scenarios in the selected type: three for users, environments, and receivers, and two for days. Add `-t` to run just one specified target scenario.

**By user** (`S1`, `S2`, `S3`):

```bash
python3 main.py -o "<OUTPUT_DIRECTORY>" -c users \
  -n "<EXPERIMENT_NAME>" -d cuda:1 -N CNN2D -A -P
```

**By environment** (`S5`, `S6`, `S7`):

```bash
python3 main.py -o "<OUTPUT_DIRECTORY>" -c environments \
  -n "<EXPERIMENT_NAME>" -d cuda:1 -N CNN2D -A -P
```

**By receiver** (`Rx1`, `Rx2`, `Rx3`):

```bash
python3 main.py -o "<OUTPUT_DIRECTORY>" -c rx \
  -n "<EXPERIMENT_NAME>" -d cuda:1 -N CNN2D -A -P
```

**By day** (`S1`, `S5`):

```bash
python3 main.py -o "<OUTPUT_DIRECTORY>" -c day \
  -n "<EXPERIMENT_NAME>" -d cuda:1 -N CNN2D -A -P
```

To run only environment scenario `S5`, for example, specify it with `-t`:

```bash
python3 main.py -o "<OUTPUT_DIRECTORY>" -c environments -t S5 \
  -n "<EXPERIMENT_NAME>" -d cuda:1 -N CNN2D -A -P
```

## `main.py` parameters

| Option | Meaning | Default / accepted values |
|---|---|---|
| `-o`, `--output` | **Required.** Base directory for experiment output. | No default |
| `-d`, `--device` | Compute device. | `cuda`; choices: `cpu`, `cuda`, `cuda:0`, `cuda:1`. If CUDA is unavailable, the script falls back to CPU. |
| `-N`, `--network` | Model architecture. | `CNN2D`; choices: `CNN2D`, `INET` |
| `-D`, `--dataset` | Dataset identifier. | `cominelli23`; this is the only choice currently accepted by the CLI. |
| `-E`, `--early` | Early-stopping flag. | Off unless supplied. **Currently parsed but not consulted by the training loop**, so supplying it does not toggle early stopping. |
| `-p`, `--patience` | Early-stopping / scheduler patience. | `10`. The script currently resets this value to `10` for either optimizer, so a different CLI value is not effective. |
| `-b`, `--batch_size` | Number of samples per training batch. | `64` |
| `-e`, `--epochs` | Maximum number of training epochs. | `200` |
| `-n`, `--exp` | Experiment name, used to name the output subdirectory. | `XAI4CSI` |
| `-x`, `--scheduler` | Learning-rate scheduler. | `RoP`; choices: `RoP`, `Fixed`, `Cosine` |
| `-r`, `--seed` | Random seed. | `46` |
| `-z`, `--out_features_size` | Model output-feature size. | `128` |
| `-O`, `--target_fold` | Fold index to use. | `0` |
| `-F`, `--num_folds` | Number of folds for fold-based splitting. | `10` |
| `-S`, `--all_folds` | Request experiments on all folds. | Off unless supplied. **Currently parsed but not used by the execution logic.** |
| `-c`, `--cross` | Cross-evaluation strategy. When omitted, all available users, environments, receivers, and days are included (upper-bound, no cross-evaluation). | Not set; choices: `users`, `environments`, `rx`, `day`. |
| `-t`, `--target_scenario` | Select one target scenario for cross-evaluation. When omitted with `-c`, automatically performs leave-one-scenario-out evaluation, training one model per target scenario: `S1`–`S3` for users, `S5`–`S7` for environments, `Rx1`–`Rx3` for receivers, and `S1`, `S5` for days. | Not set |
| `-T`, `--temporal` | Use temporal splitting instead of stratified splitting; sets the fold count to one. | Off unless supplied. Cross-evaluation also selects temporal splitting. |
| `-vs`, `--val_size` | Validation proportion. | `0.2` |
| `-ts`, `--test_size` | Test proportion for temporal splitting. | `0.2` |
| `-ps`, `--post_scaling` | Apply post-scaling after scenario filtering during cross-evaluation. | Off unless supplied |
| `-f`, `--class_filter` | Restrict the dataset to one or more activity classes. Supply class labels as separate values after the option. | No filter |
| `-op`, `--optimizer` | Optimizer. | `SGD`; choices: `SGD`, `Adam`. The script sets the learning rate internally (`0.1` for SGD, `0.001` for Adam). |
| `-sc`, `--scaler` | Data scaling method. | `MMS` (MinMax scaling); `SS` (StandardScaler) |
| `-A`, `--amplitude` | Include amplitude CSI data. | Off unless supplied |
| `-P`, `--phase` | Include phase-difference CSI data. | Off unless supplied |
| `-w`, `--window_size` | CSI window length. | `50` |
| `-s`, `--stride` | Window stride. | `25` |
| `-W`, `--wifi` | Wi-Fi type used to select CSI configuration. | `ax`; choices: `ax`, `ac` |
| `-C`, `--bandwidth` | Channel bandwidth in MHz. | `80` |

## Output

Results are written under `<OUTPUT_DIRECTORY>/<DATASET>_scratch_<EXPERIMENT_NAME>/`. The run creates subdirectories for saved models, training and validation epoch checkpoints, and results. It also writes a timestamped stdout log and a JSON file containing the run arguments.
