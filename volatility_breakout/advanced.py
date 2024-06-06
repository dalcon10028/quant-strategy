import numpy as np
import pandas as pd
import dataclasses as dc
import json
from datetime import datetime


@dc.dataclass
class VolatilityBreakoutResult:
    """ Result of volatility breakout strategy.

    Attributes:
        cagr (float): Compound annual growth rate.
        mdd (float): Maximum drawdown.
        simple_hpr (float): Simple holding period return.
        hpr (float): Holding period return.
    """
    cagr: float
    mdd: float
    simple_hpr: float
    hpr: float


def calculate_hpr(first_close: float, last_close: float) -> float:
    """ Calculate holding period return.

    Args:
        first_close (float): First close price.
        last_close (float): Last close price.

    Returns:
        float: Holding period return.
    """
    return (last_close - first_close) / first_close


def calculate_noise_ratio(open: float, high: float, low: float, close: float) -> float:
    """ Calculate noise ratio.

    Args:
        open (float): Open price.
        high (float): High price.
        low (float): Low price.
        close (float): Close price.

    Returns:
        float: Noise ratio.
    """

    return 1 - abs(open - close) / (high - low) if high - low != 0 else 0


def volatility_breakout(df: pd.DataFrame, noise_ratio_period: int = 13, fee: float = 0.1) -> VolatilityBreakoutResult:
    """ Volatility breakout strategy.

    Args:
        df (pd.DataFrame): DataFrame with open, high, low, close, and volume data.
        noise_ratio_period (int): Noise ratio period days. (Default is 13 days).
        fee: Slippage and commission fee. (Default is 0.1%).

    Returns:
        VolatilityBreakoutResult: Result of volatility breakout strategy.
    """

    # Calculate holding period returns
    simple_hpr = calculate_hpr(df['close'].iloc[0], df['close'].iloc[-1])

    # Add Noise Ratio and Average Noise Ratio to DataFrame
    df['noise_ratio'] = df.apply(lambda x: calculate_noise_ratio(x['open'], x['high'], x['low'], x['close']), axis=1)
    df['average_noise_ratio'] = df['noise_ratio'].rolling(window=noise_ratio_period).mean()

    # Add Target Price to DataFrame
    df['range'] = (df['high'] - df['low']).shift(1)
    df['next_open'] = df['open'].shift(-1)
    df['target_price'] = df['open'] + df['range'] * df['average_noise_ratio']
    df['buy_signal'] = np.where(df['high'] > df['target_price'], True, False)

    # Calculate Rate of Return
    df['ror'] = np.where(df['buy_signal'], df['next_open'] / df['target_price'] - fee / 100, 1)

    # Holding period return
    df['hpr'] = df['ror'].cumprod()

    # Maximum drawdown
    # (high - low) / high
    df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax()

    df.dropna(inplace=True)

    # Calculate CAGR and MDD
    # CAGR = (Ending Value / Beginning Value) ^ (1 / Number of Years) - 1
    cagr = df['hpr'][-1] ** (252 / len(df)) - 1
    mdd = df['dd'].max()
    hpr = df['hpr'][-1]

    return VolatilityBreakoutResult(
        cagr=cagr,
        mdd=mdd,
        simple_hpr=simple_hpr,
        hpr=hpr
    )


def main():
    # Load data
    with open('time_series_233740_1day.json') as f:
        data = json.load(f)

    # Convert data to DataFrame
    twelve_data_types = {'datetime': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64',
                         'close': 'float64', 'volume': 'float64'}
    df = (pd.DataFrame(data['values'])
          .drop_duplicates()
          .astype(twelve_data_types)
          .sort_values('datetime')
          .set_index('datetime'))

    # Calculate volatility breakout strategy
    result = volatility_breakout(df, noise_ratio_period=20, fee=0.1)

    # dataframes to excel
    df.to_excel(f'volatility_breakout_{datetime.now().strftime("%Y%m%d%H%M%S")}.xlsx')

    print(result)


if __name__ == '__main__':
    main()
