# Toy Pixel Transformer

## Train model

Let's say I want to run experiment `configs/experiment/toy_mirror-pit_grid_small.yaml`:

```sh
python -m src.train experiment=toy_mirror-pit_grid_small
```

You can also override lightning parameters like so:

```sh
python -m src.train experiment=toy_mirror-pit_grid_small trainer.accelerator=cuda
```