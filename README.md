# EAP_EF
Code and papers writting for courses Empirical Asset Pricing and Economic Forecasting.
Both papers utilize sentiment extracted from Reddit comments collected from google big query. 

# Empirical Asset Pricing: 
This paper examines the cryptocurrency market with an outset in traditional empirical asset pricing, to assess potential
similarities with the asset classes for which our theory is based, namely equities. More specifically, we investigate the
potential role sentiment plays in this market by analyzing the significance of the excess returns on a zero-investment
strategy based on daily sentiment sorted portfolios. We then examine if this cross-sectional cryptocurrency return predictor
can be explained using a low number of common factors, utilizing the Fama-Macbeth regression setup and related factors
[Fama & MacBeth, 1973], [Fama & French, 1992]. Finally, we construct a sentiment factor and
test its relevance in risk pricing when controlling for established factors. Portfolios and common factors are constructed
using data from 01.01.14-29.30.19, consisting of twelve cryptocurrencies which historically have constituted at least 80%
of the entire market capitalization. Computational restrictions have resulted in the use of a relatively low amount of coins,
which entails that the constructed portfolios likely suffer from a bit of noise and idiosyncratic risk. We have opted to
use quartile portfolios to balance this noise while still obtaining a proper amount of cross-sectional variability in portfolio
returns.

# Economic Forecasting: 
This paper seeks to determine the performance of the most prevalent forecasting models when
incorporating social media sentiment within the field of Bitcoin. We further conduct forecast
combinations, in order to examine whether optimal performance could be obtained by utilizing
both the econometric models and the machine learning techniques with and without sentiment.
