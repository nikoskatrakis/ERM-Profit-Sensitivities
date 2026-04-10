# ERM-Profit Sensitivities

This is a visual tool, driven primarily from the definition of Day 1 Gain as it was defined in PRA's SS3/17, in 2025.
It allows the user to see how one of the two metrics (day 1 gain and Profit, as defined further down) vary when one or two parameters vary. 

A number of other parameters remain constant. 

After each change, the user needs to press the update button for the chart/surface to update.

The user can choose as variable(s) any of the following: 
- Risk free rate to time of sale
- Time of sale
- Loan accumulation rate
- Deferment rate
- House price at inception
- House price volatility (assumed the same for pricing and for capital calculations)
- Funding cost (assumed to be a % of the loan amount at outset).
- Loan to Value ratio
- SCR percentile used for calculating Cost of Capital (see below)
- SCR exponential decay factor $λ$
- Real-world House price inflation rate (used in SCR calculations only)

The code uses a Black-scholes formula and lognormal distribution for the house price over time to evaluate the NNEG.

In the above, it is assumed that the house price inflation is equal to the difference between the risk free rate $r$ and the deferment rate $q$.

The property is assumed to be sold at a certain future time, $T$. 

All above variables are continuous where relevant, except from the loan accumulation rate that is annually compounding. 

The user can choose to vary one or two parameters and see how the resulting Day-1 gain and Profit behave. 
Profit is defined as Day-1 Gain less PV of cost of capital. This PV is defined as follows: 
1. We calculate the SCR as the PV of the shortfall of the property proceeds over the accumulated loan amount on the date of reversion discounted to today; for that calculation, a RW HPI value is used; the user can choose the percentile of the property distribution to use; that gives us SCR_0
2. We assume an exponential decay for the calculation of SCR at future years, compared to the SCR_0 at a rate λ.
3. We therefore calculate a PV, by allowing for the above decay at the risk free rate, and using the CoC rate supplied by the user. 

So we get that 

```math
\text{Profit}
= \text{Day1Gain}
- R_{\text{CoC}} \cdot \text{SCR}_0 \cdot
\left[
\frac{e^{-\lambda}}{1 + r_{\text{fr}}}
\cdot
\frac{1 - \left(\frac{e^{-\lambda}}{1 + r_{\text{fr}}}\right)^T}
{1 - \frac{e^{-\lambda}}{1 + r_{\text{fr}}}}
\right]
```

The ranges for the variables are reasonably broad I hope, maybe too broad to be realistic. But the user can modify the ranges through the API to see the behaviour in more tight areas if they wish.
I hope this is a useful tool for people who would like to get an insight to the dynamics of Equity Release Mortgages.

You may notice that there is no sensitivity to prepayment risk or mortality risk (although the time to sale can be thought as a proxy for these). 

A spreadsheet that shows the calculations performed by the tool is also on the repository. The tool is validated against the spreadsheet using all extreme scenarios and around 180 multi-variate random scenarios. But the spreadsheet itself is not validated against anything else.

Any comments, please email nikoskatrakis@gmail.com
