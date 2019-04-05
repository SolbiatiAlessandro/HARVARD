Greeks
======

> DELTA derivatives respect to stock price
> VEGA  derivative respect of volatility

recall **put call parity**: Call(t,T) - Put(t,T) = V(t,T)
(DELTA of the call) - 1 = (Delta of the Put) 
VEGA of the call = VEGA of the put

Vega of call, put > 0
Vega of forward contract = 0
Vega of a future contract > 0if correlation with underlying asset and money market account is positive

- example: Call Spread (long with strike K1 and short with strike K2)
You want stock to move a lot if you are at price < K1, and you want not to move a lot if you are > K2
Vega of a Call Spread : indeterminate
Delta of Call Spread : always positive (you always want to go right)

- example: Butterfly (1 long and 2 shorts)
Vega indeterminate
Delta indeterminate

> Gamme of call option: second derivative of stock price
> Theta: derivatives over time


Black Scholes Formula
=====================

Expected value of the option payout

Call(strike k)(t, T) = Z(t, T)(Forward(t, T)Phi(d1) - K * Phi(d2))

Z is the zero-coupon bond


Options on Forward Contracts
============================

- Allows us to 
- Underpins financial system (Swaptions)

(Going long a forward contract is an agreement to buy the stock at a time T)
Consider Eurpoean Option at time T to go long a forward contract, with delivery price K and maturity T
I would exercise the option at time T if the option has positive value
if S(T) >= K: exercise option
Payout function is (S(t) - K)+, that is the same payout as European Call Option







