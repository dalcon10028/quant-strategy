import json
import logging
import pandas as pd
import numpy as np
import dataclasses as dc

from backtesting import Backtest, Strategy

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


def volatility_breakout(
        df: pd.DataFrame,
        noise_ratio_period: int = 13,
        fee: float = 0.1,
        moving_average_period: int = 5,
) -> VolatilityBreakoutResult:
    """ Volatility breakout strategy.

    Args:
        df (pd.DataFrame): DataFrame with open, high, low, close, and volume data.
        noise_ratio_period (int): Noise ratio period days. (Default is 13 days).
        fee: Slippage and commission fee. (Default is 0.1%).
        trailing_stop: Trailing stop loss. (Default is 5%).

    Returns:
        VolatilityBreakoutResult: Result of volatility breakout strategy.
    """

    # Calculate holding period returns
    simple_hpr = calculate_hpr(df['close'].iloc[0], df['close'].iloc[-1])

    # Add Noise Ratio and Average Noise Ratio to DataFrame
    df['noise_ratio'] = df.apply(lambda x: calculate_noise_ratio(x['open'], x['high'], x['low'], x['close']), axis=1)
    df['average_noise_ratio'] = df['noise_ratio'].rolling(window=noise_ratio_period).mean()

    # Add target price
    df['range'] = (df['high'] - df['low']).shift(1)
    df['next_open'] = df['open'].shift(-1)
    df['target_price'] = df['open'] + df['range'] * df['average_noise_ratio']

    # Add market timing
    df['md'] = df['close'].rolling(window=moving_average_period).mean()

    # Add buy signal
    df['buy_signal'] = np.where((df['high'] > df['target_price']) & (df['high'] > df['md']), True, False)

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler() ## 스트림 핸들러 생성
logger.addHandler(stream_handler) ## 핸들러 등록

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

def main():

    class VolatilityBreakoutStrategy(Strategy):
        noise_ratio_period = 20
        moving_average_period = 5
        # buy_amount = 90_000_000

        def init(self):
            logger.info('init')
            # 이동평균선 추가
            pass

        def next(self):
            # 날짜와 데이터 출력
            if not self.position and self.data['buy_signal'][-1]:
                logger.info(self.position)
                logger.info(f'buy {self.data.index[-1]}: {self.data["Close"][-1]} > {self.data["target_price"][-1]}')

                # buy_amount로 가능한만큼 매수

                # if self.buy_amount > self.equity:
                #     self.buy(limit=self.data['target_price'][-1], size=self.equity // self.data['Close'][-1])
                # else:
                #     self.buy(limit=self.data['target_price'][-1], size=self.buy_amount // self.data['Close'][-1])

                self.buy(limit=self.data['target_price'][-1])

            # 다음날 시가에 매도
            elif self.position:
                logger.info(self.position)
                logger.info(f'sell {self.data.index[-1]}: {self.data["Open"][-1]}')
                self.position.close()
                logger.info(self.position)

    # Load data
    with open('../../data/ohlcv/twelvedata/time_series_233740_1day.json') as f:
        data = json.load(f)

    # Convert data to DataFrame
    twelve_data_types = {'datetime': 'datetime64[ns]', 'open': 'float64', 'high': 'float64', 'low': 'float64',
                         'close': 'float64'}
    df = (pd.DataFrame(data['values'])
          .drop_duplicates()
          .astype(twelve_data_types)
          .sort_values('datetime')
          .set_index('datetime'))

    # Calculate volatility breakout strategy
    result = volatility_breakout(df, noise_ratio_period=20, fee=0.0)

    # convert columes for backtesting
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'noise_ratio', 'average_noise_ratio', 'range', 'next_open', 'target_price', 'md', 'buy_signal', 'ror', 'hpr', 'dd']

    # Run backtest
    bt = Backtest(df, VolatilityBreakoutStrategy, cash=100_000_000, commission=.0015)
    # bt = Backtest(df, VolatilityBreakoutStrategy, cash=100_000_000, commission=.001)
    stats = bt.run()
    bt.plot()

    print(stats)

    # pretty print stats._trades


if __name__ == '__main__':
    main()