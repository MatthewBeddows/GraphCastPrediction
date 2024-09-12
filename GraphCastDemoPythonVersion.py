# Import necessary libraries
import dataclasses
import functools

from typing import Optional
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import matplotlib
import numpy as np
import xarray



# Define utility functions
def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))

def select(data: xarray.Dataset, variable: str, level: Optional[int] = None, max_steps: Optional[int] = None) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(data: xarray.Dataset, center: Optional[float] = None, robust: bool = False) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (data, matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis"))

def save_data_to_csv(data: dict[str, tuple], file_prefix: str = "output_data"):
    """
    Save data from xarray.Dataset or xarray.DataArray contained in tuples to CSV files.

    Args:
    - data (dict[str, tuple]): Dictionary where each value is a tuple containing xarray Datasets or DataArrays.
    - file_prefix (str): Prefix for the CSV file names.

    Returns:
    - None
    """
    for title, dataset_tuple in data.items():
        # Handle tuples, assuming they contain xarray.Dataset or xarray.DataArray objects
        if not isinstance(dataset_tuple, tuple):
            print(f"Error: {title} is not a tuple. Skipping...")
            continue
        
        for i, dataset in enumerate(dataset_tuple):
            if hasattr(dataset, 'to_dataframe'):
                # Flatten the dataset for saving to CSV
                df = dataset.to_dataframe().reset_index()

                # Generate a filename based on the title and index in the tuple
                filename = f"{file_prefix}_{title.replace(' ', '_')}_{i}.csv"

                # Save to CSV
                df.to_csv(filename, index=False)
                print(f"Saved {title} data (tuple index {i}) to {filename}")
            else:
                print(f"Warning: Object at index {i} in tuple '{title}' is not an xarray object. Skipping...")






# Load the Data and initialize the model
params_file = "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
#dataset_file = "source-era5_date-2024-01-01_res-0.25_levels-13_steps-12.nc"
dataset_file = "source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"

# Load the model params
with open(f"../model/params/{params_file}", "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
print("Model license:\n", ckpt.license, "\n")

# Load the example data
with open(f"../model/dataset/{dataset_file}", "rb") as f:
    example_batch = xarray.load_dataset(f).compute()


assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets

print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()]))

# Extract training and eval data
train_steps = 1
eval_steps = example_batch.sizes["time"] - 2

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"), **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

# Load normalization data
with open("../model/stats/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open("../model/stats/mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()
with open("../model/stats/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

# Build jitted functions, and possibly initialize random weights
def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    loss, diagnostics = predictor.loss(inputs, targets, forcings)
    return xarray_tree.map_structure(lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True), (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(params, state, jax.random.PRNGKey(0), model_config, task_config, i, t, f)
        return loss, (diagnostics, next_state)
    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(params, state, inputs, targets, forcings)
    return loss, diagnostics, next_state, grads

def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)

def with_params(fn):
    return functools.partial(fn, params=params, state=state)

def drop_state(fn):
    return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

if params is None:
    params, state = init_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings
    )

loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))



# Run the model
print("Inputs:  ", eval_inputs.dims.mapping)
print("Targets: ", eval_targets.dims.mapping)
print("Forcings:", eval_forcings.dims.mapping)

predictions = rollout.chunked_prediction(
    run_forward_jitted,
    rng=jax.random.PRNGKey(0),
    inputs=eval_inputs,
    targets_template=eval_targets * np.nan,
    forcings=eval_forcings
)

# Plot predictions
plot_pred_variable = "2m_temperature"
plot_pred_level = 500
plot_pred_robust = True
plot_pred_max_steps = predictions.dims["time"]

plot_size = 5
plot_max_steps = min(predictions.dims["time"], plot_pred_max_steps)

data = {
    "Targets": scale(select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
    "Predictions": scale(select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps), robust=plot_pred_robust),
    "Diff": scale((select(eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps) -
                   select(predictions, plot_pred_variable, plot_pred_level, plot_max_steps)),
                  robust=plot_pred_robust, center=0),
}
fig_title = plot_pred_variable
if "level" in predictions[plot_pred_variable].coords:
    fig_title += f" at {plot_pred_level} hPa"

#plot_data(data, fig_title, plot_size, plot_pred_robust)
save_data_to_csv(data, file_prefix="weather_data")

# Loss computation
loss, diagnostics = loss_fn_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings
)
print("Loss:", float(loss))

# Gradient computation
loss, diagnostics, next_state, grads = grads_fn_jitted(
    inputs=train_inputs,
    targets=train_targets,
    forcings=train_forcings
)
mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

# Autoregressive rollout
print("Inputs:  ", train_inputs.dims.mapping)
print("Targets: ", train_targets.dims.mapping)
print("Forcings:", train_forcings.dims.mapping)

predictions = run_forward_jitted(
    rng=jax.random.PRNGKey(0),
    inputs=train_inputs,
    targets_template=train_targets * np.nan,
    forcings=train_forcings
)
