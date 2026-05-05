
## $\chi^2$ meaning and code

`chi2_contingency` builds this "perfectly independent" expected distribution table, it compares it against your actual, observed data. It calculates the Chi-Square ($\chi^2$) statistic by summing the squared differences between the observed ($O$) and expected ($E$) values, normalized by the expected values:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

The larger the $\chi^2$ value, the more your actual data deviates from the expected independent distribution, indicating that the variables are likely related.
## $p_{value}$
$p_{value}$: it is the probability of observing a **Chi-Square statistic ($X^2$)** as large as (or larger than) the one you just calculated from your entire contingency table, assuming that $H_0$ (perfect independence) is true. 


> Area under the $\chi^2$ distribution
> To find the probability of a particular value, we find the area under the curve before the value. The area that's after the value is called the p-value

Before running an experiment, scientists set a "threshold for guilt," known as the **Significance Level ($\alpha$)**. The most common threshold is **0.05** (or 5%).

- **Low p-value ($\le$ 0.05):** Your data is highly unlikely to have occurred by random chance alone. You **reject the null hypothesis** and conclude your results are "statistically significant." (The defendant is <span style="color:rgb(255, 0, 0)">guilty</span>).
- **High p-value ($>$ 0.05):** Your data is fairly likely to happen by random chance. You **fail to reject the null hypothesis**. You don't have enough evidence to prove there is a real effect. (<span style="color:rgb(0, 176, 80)">Not enough evidence</span> to convict).
