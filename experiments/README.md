# Experiments

This directory keeps experiment planning and status files separate from raw
data, generated outputs, and model checkpoints.

```text
status/       Implementation status and run-state notes.
runbooks/     Sanitized commands without API keys or passwords.
results/      Paper-facing summaries traced to raw outputs.
```

Raw datasets stay in `data/`, generated files stay in `outputs/`, and model
weights stay in `checkpoints/`. Those directories are ignored unless a file has
already been intentionally versioned.
