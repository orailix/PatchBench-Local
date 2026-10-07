# Benchmark Configs

There is one config file per method, shared by all models. Each file contains:

- `training_fixed`: which parameters are fixed during training
- `validation_fixed`: which parameters are fixed during the validation search
- `validation_selected`: which parameters are chosen by validation
- `inference_fixed`: which parameters are fixed during benchmark inference

A parameter in `validation_selected` can specify candidate `values` and a fallback `default`. For example, a strength search can cover several orders of magnitude:
```json
"strength": {
  "default": 0.1,
  "values": [0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0]
}
```
In this example, validation tests these strengths in order on the validation prompts and saves the best one in that model’s report under `analysis/validation/`. If several parameters have `values`, validation tests their combinations. Different models may need strengths of very different magnitudes.

Parameters searched by the validation grid can specify `fixed_by_model`, a mapping from model names to fixed parameter values. For a matching model, validation uses that value as its sole candidate. Other models use the existing `values` and `default`. The AST and CAST configs use this mapping to reproduce the strengths reported in the paper. CAST's separate gate search is unaffected.

Benchmark inference loads the selected parameters from the model’s report. If no usable selection is available, it uses `default`. The `--no-validation-selection` option forces config defaults even when a report exists. A `default` is tested during validation only if it also appears in `values`.

