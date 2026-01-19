#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Running steady-state model: peak..."
python model.py run_ss peak

echo "Running steady-state model: night..."
python model.py run_ss night

echo "Running steady-state model: peak (two-layer pane)..."
python model.py run_ss peak --two-layer-pane

echo "Running steady-state model: night (two-layer pane)..."
python model.py run_ss night --two-layer-pane

# Counterfactuals (peak, single layer)
echo "Running counterfactual: equal_solar (peak)..."
python model.py run_ss peak --counterfactuals=equal_solar

echo "Running counterfactual: equal_solar_pane_abs (peak)..."
python model.py run_ss peak --counterfactuals=equal_solar_pane_abs

echo "Running counterfactual: equal_ir (peak)..."
python model.py run_ss peak --counterfactuals=equal_ir

echo "Running counterfactual: uncoated_boro (peak)..."
python model.py run_ss peak --counterfactuals=uncoated_boro

# All radiative properties counterfactual (night, single layer)
echo "Running counterfactual: equal_solar,equal_ir (night)..."
python model.py run_ss night --counterfactuals=equal_solar,equal_ir

echo "All models complete."
