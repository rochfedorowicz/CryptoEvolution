# scripts/create_dataset.py

# global imports
import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

# local imports
from source.data_handling import DataHandler, CoinbaseApiDataCollector, YahooFinanceApiDataCollector
from source.indicators import AverageTrueRangeIndicatorHandler, BollingerBandsDeviatorIndicatorHandler, \
    DonchainChannelsIndicatorHandler, ExponentialMovingAverageIndicatorHandler, IndicatorHandlerBase, \
    MovingAverageConvergenceDivergenceIndicatorHandler, MoneyFlowIndexIndicatorHandler, \
    MovingVolumeProfileIndicatorHandler, OnBalanceVolumeIndicatorHandler, \
    RelativeStrengthIndexIndicatorHandler, StochasticOscillatorIndicatorHandler, \
    VolatilityIndicatorHandler
from source.utils import AWSHandler, Granularity

def str_to_list_of_indicators(list_of_indicators_str: Optional[str] = None) -> tuple[list[IndicatorHandlerBase], str]:
    if not list_of_indicators_str:
        return [], list_of_indicators_str

    indicators_map = {
        'atr': AverageTrueRangeIndicatorHandler(),
        'bb': BollingerBandsDeviatorIndicatorHandler(),
        'dc': DonchainChannelsIndicatorHandler(),
        'ema': ExponentialMovingAverageIndicatorHandler(),
        'macd': MovingAverageConvergenceDivergenceIndicatorHandler(),
        'mfi': MoneyFlowIndexIndicatorHandler(),
        'mvp': MovingVolumeProfileIndicatorHandler(),
        'obv': OnBalanceVolumeIndicatorHandler(),
        'rsi': RelativeStrengthIndexIndicatorHandler(),
        'so': StochasticOscillatorIndicatorHandler()
    }

    if list_of_indicators_str == 'all':
        return list(indicators_map.values()), ','.join(indicators_map.keys())

    list_of_indicators = []
    for indicator_str in list_of_indicators_str.split(','):
        indicator_str = indicator_str.strip()
        indicator = indicators_map.get(indicator_str)
        if indicator is None:
            logging.error(f'Found invalid indicator name!')
            raise ValueError(f"Invalid indicator name: '{indicator_str}'. Valid options: {list(indicators_map.keys())}")
        list_of_indicators.append(indicator)

    return list_of_indicators, list_of_indicators_str

async def main(start_date: str, end_date: str, granularity_str: str, ticker: Optional[str] = None,
    file_path: Optional[str] = None, list_of_indicators_str: Optional[str] = None) -> bool:
    try:
        aws_handler = AWSHandler()
        data_handler = DataHandler()
        data_handler.register_api_data_collectors([CoinbaseApiDataCollector(), YahooFinanceApiDataCollector()])
        list_of_indicators, list_of_indicators_str = str_to_list_of_indicators(list_of_indicators_str)
        list_of_indicators.append(VolatilityIndicatorHandler())

        data, meta_data = await data_handler.prepare_data(ticker or file_path, start_date, end_date,
                                                          Granularity.from_string(granularity_str),
                                                          list_of_indicators)
        csv_data_buffer = data_handler.save_extended_data_into_csv_formatted_string_buffer(data, meta_data)

        if not ticker:
            ticker = os.path.splitext(os.path.basename(file_path))[0]
        file_name = f'DS_{ticker}_{start_date}_{end_date}_{granularity_str}_{list_of_indicators_str}.csv'
        for char_to_replace in [':', ' ', ',']:
            file_name = file_name.replace(char_to_replace, '_')

        aws_handler.upload_buffer_to_s3(os.getenv('BUCKET_NAME'), csv_data_buffer, file_name)
        logging.info('Successfully uploaded data to S3 bucket! File name: %s', file_name)
        return True

    except Exception:
        logging.error('Encounter problem during script execution!', exc_info=True)
        logging.error('Script execution failed!')
        return False

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "{asctime} | {levelname} | {funcName}:{lineno} | {message}",
                        style = "{", datefmt = "%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = 'Prepare data with given parameters and save it into AWS S3 bucket.')

    input_group = parser.add_mutually_exclusive_group(required = True)
    input_group.add_argument('--ticker', type = str, help = 'Asset unique identifier.')
    input_group.add_argument('--file_path', type = str, help = 'Path to the file to be preprocessed.')

    parser.add_argument('--start_date', type = str, required = True, help = 'Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type = str, required = True, help = 'End date in YYYY-MM-DD format.')
    parser.add_argument('--granularity', type = str, required = True, choices = ['1m', '5m', '15m', '30m', '1h', '6h', '1d'],
                               help = 'Granularity of the fetched data.')
    parser.add_argument('--add_indicators', type = str, required = False,
                               nargs = '?', const = 'all', default = None,
                               help = '''List of indicators to be added, that looks like: indicator_1,indicator_2,...,indicator_N.
                               Possible indicators are: average_true_range=atr, bollinger_bands=bb, donchain_channels=dc,
                               exponential_moving_average=ema, moving_average_convergence_divergence=macd, money_flow_index=mfi,
                               moving_volume_profile=mvp, on_balance_volume=obv, relative_strength_index=rsi, stochastic_oscillator=so.
                               If not explicitly specified, all indicators will be used.''')

    if sys.platform.startswith('win'):
        policy = asyncio.WindowsSelectorEventLoopPolicy()
    else:
        policy = asyncio.DefaultEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

    args = parser.parse_args()
    success = asyncio.run(main(args.start_date, args.end_date, args.granularity, args.ticker,
                               args.file_path, args.add_indicators))

    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
