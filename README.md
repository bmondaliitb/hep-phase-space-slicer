# HEP Phase Space Slicer

Interactive Python GUI for exploring correlations in numpy feature arrays.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
hep-phase-space-slicer clustercalib_nn_500k_batch001.npy
```

or:

```bash
python -m phase_space_slicer clustercalib_nn_500k_batch001.npy other_batch.npy
```

## Use

- Add as many 2D correlation panels or 1D distribution panels as needed.
- Pick the X and Y variables for each panel.
- Color depth shows the number of selected rows in each X/Y bin.
- Set X range, Y range, and bin resolution independently for every plot.
- Use **Auto Range** on a plot to reset its X/Y limits to the selected columns.
- 1D distributions use the same active phase-space slice as the 2D plots.
- Use `_` and `□` on any plot to minimize or maximize that plot.
- Load multiple files at once; files are concatenated row-wise and must have the same column count.
- Use the **Spreadsheet** tab to preview data, rename columns, and create derived columns.
- Derived columns use Python/numpy expressions such as `np.log(col0) + sqrt(col1)`.
- Enable the slice tool to show highlighted slice regions on plots.
- On 2D plots, set the highlighted box with `x1`, `y1`, `x2`, `y2`, and `Angle`; the box updates immediately.
- Applying a 2D slice updates the entries while preserving that plot's current range and binning.
- On 1D plots, drag the two vertical lines to select a distribution range.
- Click **Apply Slice** on any plot to update all 1D and 2D plots.
- Use **Clear Slice** to show all events again.

For plain 2D arrays without column names, columns are named `feature_00`, `feature_01`, and so on.
Renamed labels, derived-column formulas, and the active GUI state are saved in `.phase_space_slicer_state.json` beside the data file; the source `.npy` file is not modified.
# hep-phase-space-slicer
