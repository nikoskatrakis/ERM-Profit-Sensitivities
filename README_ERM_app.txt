ERM Sensitivity Explorer

Files:
- erm_sensitivity_app.py

Run:
python3 erm_sensitivity_app.py

What it does:
- Recreates the ERM profitability logic from the spreadsheet without reading the spreadsheet at runtime.
- Lets you choose one or two input variables.
- Lets you choose the output variable: Day1Gain or Profit.
- Lets you set min and max values for each selected input.
- Validates min/max against the hard-coded ranges taken from the spreadsheet.
- Uses 30-step sliders for each selected variable.
- Draws either:
  - a 2D line chart for one input variable, or
  - a rotatable 3D surface + clickable grid markers for two input variables.
- Clicking a plotted point shows the nearest X / Y / Z combination.

Notes:
- Loan amount keeps the spreadsheet-style lower bound and uses the current house value as its effective upper bound.
- The code is standalone and does not depend on the spreadsheet file.
