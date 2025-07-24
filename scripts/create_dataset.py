# scripts/create_data.py

# global imports
import argparse
import asyncio
import io
import logging
import os
import sys

# local imports
from source.data_handling import DataHandler
from source.indicators import DonchainChannelsIndicatorHandler, \
    MovingVolumeProfileIndicatorHandler, StochasticOscillatorIndicatorHandler
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
        'donchain_channels': DonchainChannelsIndicatorHandler(),
        'moving_volume_profile': MovingVolumeProfileIndicatorHandler(),
        'stochastic_oscillator': StochasticOscillatorIndicatorHandler()
    }
    list_of_indicators = []
    for indicator_str in list_of_indicators_str.split(','):
        list_of_indicators.append(indicators_map.get(indicator_str))

    return list_of_indicators

async def main(trading_pair, start_date, end_date, granularity_str, list_of_indicators_str) -> bool:
    try:
        data_handler = DataHandler(str_to_list_of_indicators(list_of_indicators_str))
        data = await data_handler.prepare_data(trading_pair, start_date, end_date, str_to_granularity(granularity_str))
        csv_data_buffer = io.StringIO()
        data.to_csv(csv_data_buffer, index = True)

        file_name = f'DS_{trading_pair}_{start_date}_{end_date}_{granularity_str}_{list_of_indicators_str}.csv'
        for char_to_replace in [':', ' ', ',']:
            file_name = file_name.replace(char_to_replace, '_')
        aws_handler = AWSHandler(os.getenv('ROLE_NAME'))
        aws_handler.upload_buffer_to_s3(os.getenv('BUCKET_NAME'), csv_data_buffer, file_name)
        logging.info('Successfully uploaded data to S3 bucket! File name: %s', file_name)
        return True

    except Exception as e:
        logging.error('Encounter problem during script execution!')
        logging.error(e)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Prepare data with given parameters and save it into AWS S3 bucket.')
    parser.add_argument('--trading_pair', type = str, required = True, help = 'Trading pair symbol.')
    parser.add_argument('--start_date', type = str, required = True, help = 'Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', type = str, required = True, help = 'End date in YYYY-MM-DD format.')
    parser.add_argument('--granularity', type = str, required = True, choices = ['1m', '5m', '15m', '30m', '1h', '6h', '1d'],
                        help = 'Granularity of the fetched data.')
    parser.add_argument('--list_of_indicators', type = str, required = False,
                        help = '''List of indicators, that looks like: indicator_1,indicator_2,...,indicator_N.
                        Possible indicators are: donchain_channels, moving_volume_profile, stochastic_oscillator.''')

    if sys.platform.startswith('win'):
        policy = asyncio.WindowsSelectorEventLoopPolicy()
    else:
        policy = asyncio.DefaultEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)

    args = parser.parse_args()
    success = asyncio.run(main(args.trading_pair, args.start_date, args.end_date, args.granularity, args.list_of_indicators))

    if not success:
        logging.error('Script execution failed!')
        sys.exit(1)
