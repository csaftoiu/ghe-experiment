#!/bin/bash
set -e

cd "$(dirname "$0")"

# Function to process data only if output doesn't exist
process_if_needed() {
    local input="$1"
    local mat_a="$2"
    local mat_b="$3"
    local mat_c="$4"

    local basename=$(basename "$input" .csv)
    local output="data/main_exp/processed/${basename}.csv"

    if [ ! -f "$output" ]; then
        echo "Processing $input..."
        python main_exp_plots.py process_data "$input" "$mat_a" "$mat_b" "$mat_c"
    fi
}

# Process main experiment data (creates data/main_exp/processed/)
echo "Checking/processing main experiment data..."
process_if_needed data/main_exp/June2021_SBC_full.csv Sapph Boro CaF2
process_if_needed data/main_exp/June2122_CBS_full.csv CaF2 Boro Sapph
process_if_needed data/main_exp/June2223_BCS_final.csv Boro CaF2 Sapph
process_if_needed data/main_exp/June2324_CSB_final.csv CaF2 Sapph Boro
process_if_needed data/main_exp/June2425_SCB_final.csv Sapph CaF2 Boro
process_if_needed data/main_exp/Jul67_SCB_sunny.csv Sapph CaF2 Boro
process_if_needed data/main_exp/Jul78_BSC_sunny.csv Boro Sapph CaF2
process_if_needed data/main_exp/Jul810_BCS_sunny.csv Boro CaF2 Sapph
process_if_needed data/main_exp/Jul1011_CSB_cloudy.csv CaF2 Sapph Boro
process_if_needed data/main_exp/Jul1112_CSB_cloudy.csv CaF2 Sapph Boro
process_if_needed data/main_exp/Jul1213_CSB_sunny.csv CaF2 Sapph Boro

# Generate transmission plot
echo "Generating transmission plot..."
python pane_properties.py transmission

# Generate clear-sky (sunny) figure
echo "Generating clear-sky figure..."
python main_exp_plots.py plot_paper_figures \
	data/main_exp/processed/June2021_SBC_full.csv \
	data/main_exp/processed/June2122_CBS_full.csv \
	data/main_exp/processed/Jul67_SCB_sunny.csv \
	data/main_exp/processed/Jul78_BSC_sunny.csv \
	data/main_exp/processed/Jul810_BCS_sunny.csv \
	data/main_exp/processed/Jul1213_CSB_sunny.csv \
	--wrap=1 --opacity=0.3 --average --opacity-avg=1.0 --lw-avg=5 \
	--title="Clear-sky Temperature Differences"

# Generate cloudy figure
echo "Generating cloudy figure..."
python main_exp_plots.py plot_paper_figures \
	data/main_exp/processed/June2223_BCS_final.csv \
	data/main_exp/processed/June2324_CSB_final.csv \
	data/main_exp/processed/June2425_SCB_final.csv \
	data/main_exp/processed/Jul1011_CSB_cloudy.csv \
	data/main_exp/processed/Jul1112_CSB_cloudy.csv \
	--wrap=1 --opacity=0.3 --average --opacity-avg=1.0 --lw-avg=5 \
	--title="Cloudy Sky Temperature Differences" \
	--output="cloudy_sky_combrel.pdf"

# Generate simple variant figures
echo "Generating simple variant main figure..."
python simple_variant_plots.py main_exp

echo "Generating simple variant glass figure..."
python simple_variant_plots.py glass_exp

echo "All figures generated in figures/"
