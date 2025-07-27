# scripts/create_data.py

# global imports
import argparse
import asyncio
import logging
import os
import sys

# local imports
from source.data_handling import DataHandler
from source.indicators import AverageTrueRangeIndicatorHandler, BollingerBandsDeviatorIndicatorHandler, \
    DonchainChannelsIndicatorHandler, ExponentialMovingAverageIndicatorHandler, \
    MovingAverageConvergenceDivergenceIndicatorHandler, MoneyFlowIndexIndicatorHandler, \
    MovingVolumeProfileIndicatorHandler, OnBalanceVolumeIndicatorHandler, \
    RelativeStrengthIndexIndicatorHandler, StochasticOscillatorIndicatorHandler, \
    VolatilityIndicatorHandler
from source.utils import AWSHandler, Granularity

def str_to_granularity(granularity_str):
    granularity_map = {
        '1m': Granularity.ONE_MINUTE,
        '5m': Granularity.FIVE_MINUTES,
        '15m': Granularity.FIFTEEN_MINUTES,
        '30m': Granularity.THIRTY_MINUTES,
        '1h': Granularity.ONE_HOUR,
        '6h': Granularity.SIX_HOURS,
        '1d': Granularity.ONE_DAY
    }

    return granularity_map.get(granularity_str)

def str_to_list_of_indicators(list_of_indicators_str):
    if not list_of_indicators_str:
        return []

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
    list_of_indicators = []
    for indicator_str in list_of_indicators_str.split(','):
        list_of_indicators.append(indicators_map.get(indicator_str))

    return list_of_indicators

async def main(trading_pair, start_date, end_date, granularity_str, list_of_indicators_str) -> bool:
    try:
        data_handler = DataHandler()
        list_of_indicators = str_to_list_of_indicators(list_of_indicators_str) + [VolatilityIndicatorHandler()]
        if None in list_of_indicators:
            index_of_none = list_of_indicators.index(None)
            invalid_name = list_of_indicators_str.split(',')[index_of_none]
            logging.error(f'Invalid indicator name: {invalid_name}')
            raise ValueError(f"Invalid indicator in list!")

        data, meta_data = await data_handler.prepare_data(trading_pair, start_date, end_date,
                                                          str_to_granularity(granularity_str),
                                                          list_of_indicators)
        csv_data_buffer = data_handler.save_extended_data_into_csv_formatted_string_buffer(data, meta_data)

        file_name = f'DS_{trading_pair}_{start_date}_{end_date}_{granularity_str}_{list_of_indicators_str}.csv'
        for char_to_replace in [':', ' ', ',']:
            file_name = file_name.replace(char_to_replace, '_')

        aws_handler = AWSHandler()
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
    parser.add_argument('--trading_pair', type = str, required = True, help = 'Trading pair symbol.')
    parser.add_argument('--start_date', type = str, required = True, help = 'Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type = str, required = True, help = 'End date in YYYY-MM-DD format.')
    parser.add_argument('--granularity', type = str, required = True, choices = ['1m', '5m', '15m', '30m', '1h', '6h', '1d'],
                        help = 'Granularity of the fetched data.')
    parser.add_argument('--list_of_indicators', type = str, required = False,
                        help = '''List of indicators, that looks like: indicator_1,indicator_2,...,indicator_N.
                        Possible indicators are: average_true_range=atr, bollinger_bands=bb, donchain_channels=dc,
                        exponential_moving_average=ema, macd=macd, money_flow_index=mfi, moving_volume_profile=mvp,
                        on_balance_volume=obv, relative_strength_index=rsi, stochastic_oscillator=so.''')

    if sys.platform.startswith('win'):
        policy = asyncio.WindowsSelectorEventLoopPolicy()
    else:
        policy = asyncio.DefaultEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

    args = parser.parse_args()
    success = asyncio.run(main(args.trading_pair, args.start_date, args.end_date, args.granularity, args.list_of_indicators))

    if not success:
        sys.exit(1)
    else:
        sys.exit(0)
