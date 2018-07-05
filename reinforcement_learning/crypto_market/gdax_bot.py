from __future__ import print_function

import requests
import datetime
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json, hmac, hashlib, time, base64
from requests.auth import AuthBase
import os
import sys

np.warnings.filterwarnings('ignore')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_setting(setting, file):
    try:
        with open(file, 'r') as fd:
            for line in fd.readlines():
                setting_name, setting_value = line.split(':')
                if setting_name == setting:
                    return setting_value.strip()
    except:
        return None


WINDOW = 30
MAX_TRIES = 24
HOLD = 0
SELL = 1
BUY = 2
MARKET_PERIOD = 300
TIMEOUT = 5


# Create custom authentication for Exchange
class CoinbaseExchangeAuth(AuthBase):
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + request.path_url + (request.body or '')
        message = message.encode('utf-8')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message, hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')

        '''

            CB-ACCESS-KEY The api key as a string.
            CB-ACCESS-SIGN The base64-encoded signature (see Signing a Message).
            CB-ACCESS-TIMESTAMP A timestamp for your request.
            CB-ACCESS-PASSPHRASE The passphrase you specified when creating the API key.

        '''
        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request


def do_sell_action(shares, price, auth):
    shares_to_sell = np.round(0.5 * shares, ROUNDING)

    if shares_to_sell == 0.:
        return None

    order = {
        'size': str(shares_to_sell),
        'price': str(price),
        'side': 'sell',
        'product_id': PAIR,
    }

    print(order)

    r = requests.post(GDAX_URL + '/orders', data=json.dumps(order), auth=auth, timeout=TIMEOUT)

    if r.status_code == 200:
        j = r.json()
        return j['id']
    else:
        eprint('Sell order failed:', r.status_code)

    return None


def do_buy_action(cash, price, auth):
    shares_to_buy = np.round((0.5 * cash) / price, ROUNDING)

    if shares_to_buy == 0.:
        return None

    order = {
        'size': str(shares_to_buy),
        'price': str(price),
        'side': 'buy',
        'product_id': PAIR,
    }

    print(order)

    r = requests.post(GDAX_URL + '/orders', data=json.dumps(order), auth=auth, timeout=TIMEOUT)
    print(r.url)
    if r.status_code == 200:
        j = r.json()
        return j['id']
    else:
        eprint('Buy order failed: ', r.status_code)

    return None


def has_standing_order(order_id, auth):
    if order_id is None:
        return fetch_order_id(auth)

    r = requests.get(GDAX_URL + '/orders/' + str(order_id), auth=auth, timeout=TIMEOUT)
    print(r.url)
    if r.status_code == 200:
        j = r.json()
        print(j)
        if j['status'] == 'done':
            return None, False

        return order_id, True

    elif r.status_code == 404:
        # order cancelled
        return None, False
    else:
        eprint('Order status request failed:', r.status_code)
        return order_id, False


def fetch_order_id(auth):
    params = {'status': 'open', 'product_id': PAIR}
    r = requests.get(GDAX_URL + '/orders', params=params, auth=auth, timeout=TIMEOUT)

    if r.status_code == 200:
        j = r.json()
        order_id = next((d['id'] for d in j if d['status'] == 'open'), None)
        if order_id is not None:
            return order_id, True
    else:
        eprint('Failed to fetch orders:', r.status_code)
    return None, False


def cancel_order(order_id, auth):
    r = requests.delete(GDAX_URL + '/orders/' + str(order_id), auth=auth, timeout=TIMEOUT)
    if r.status_code != 200:
        raise Exception('Failed to delete order: ', order_id)


def get_account_balance(auth):
    r = requests.get(GDAX_URL + '/accounts', auth=auth, timeout=TIMEOUT)
    j = r.json()

    shares, cash = None, None

    parts = PAIR.split('-')
    for d in j:
        if d['currency'] == parts[0]:
            shares = float(d['available'])
        if d['currency'] == parts[1]:
            cash = float(d['available'])
    return shares, cash


def load_history_data():
    params = {
        'start': (datetime.datetime.now() - datetime.timedelta(minutes=int(5 * 30))).isoformat(),
        'end': datetime.datetime.now().isoformat(),
        'granularity': 300
    }
    request = requests.get(GDAX_HISTORY_URL, params=params, timeout=TIMEOUT)
    history_price_data = None
    if request.status_code == 200:
        j = request.json()
        history_price_data = pd.DataFrame(j, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

    return history_price_data


def load_price_data():
    r = requests.get(GDAX_PRODUCT_URL, timeout=TIMEOUT)
    if r.status_code == 200:
        j = r.json()
        return float(j['price'])

    return None


def compute_input(history_data):
    pct = history_data.pct_change()
    mean = pct.rolling(window=WINDOW).mean().as_matrix()
    median = pct.rolling(window=WINDOW).median().as_matrix()
    std = pct.rolling(window=WINDOW).std().as_matrix()
    upperbb = mean + (2 * std)
    lowerbb = mean - (2 * std)

    return np.array([[pct.as_matrix()[-1],
                      lowerbb[-1],
                      mean[-1],
                      median[-1],
                      upperbb[-1]]])


def compute_action_single_model(input, model='models/decision_tree_4.pkl'):
    clf = joblib.load(model)
    return clf.predict(input)


def main_loop(order=None):
    auth = CoinbaseExchangeAuth(API_KEY, API_SECRET, API_PASS)
    order, has_order = has_standing_order(order, auth)
    tries_counter = 0
    while has_order and tries_counter < MAX_TRIES:
        print(datetime.datetime.now(), ' - has standing order [', str(order), ']')
        time.sleep(300)
        order, has_order = has_standing_order(order, auth)
        tries_counter += 1

    try:
        if has_order:
            cancel_order(order, auth)
            order = None
    except Exception as e:
        eprint(e)
        return order

    shares, cash = get_account_balance(auth)

    # print('Current state(%s):%.4f,%.4f' % (PAIR, shares, cash))

    history_data = load_history_data()

    if history_data is None:
        eprint('Failed to load history data')
        return order

    history_data.sort_values(by=['time'], inplace=True)
    history_data.reset_index(drop=True, inplace=True)

    history_data = history_data['close']

    price = load_price_data()

    if price is None:
        eprint('Failed to load price data...')
        return order

    with open(LOG_FILE, 'a+') as fd:
        fd.write('%d;%.4f;%.4f;%.4f\n' % (round(time.time()), price, shares, cash))

    history_data.set_value(len(history_data), price)
    action = compute_action_single_model(compute_input(history_data), model=MODEL)

    if action == SELL:
        # print('Chosen action is: SELL')
        order = do_sell_action(shares, price, auth)
    elif action == BUY:
        # print('Chosen action is: BUY')
        order = do_buy_action(cash, price, auth)

    # print('Waiting %d seconds' % MARKET_PERIOD)
    time.sleep(MARKET_PERIOD)
    print('-' * 80)
    return order


def main(settings_file):
    global SETTINGS_FILE, API_KEY, API_PASS, API_SECRET, PAIR
    global LOG_FILE, ROUNDING, MODEL, GDAX_URL
    global GDAX_HISTORY_URL, GDAX_PRODUCT_URL

    if '~' in settings_file:
        userdir = os.path.expanduser('~')
        settings_file = settings_file.replace('~', userdir)

    SETTINGS_FILE = settings_file

    API_KEY = load_setting('API_KEY', SETTINGS_FILE)
    API_SECRET = load_setting('API_SECRET', SETTINGS_FILE)
    API_PASS = load_setting('API_PASS', SETTINGS_FILE)

    PAIR = load_setting('PAIR', SETTINGS_FILE)
    LOG_FILE = load_setting('LOG_FILE', SETTINGS_FILE)
    ROUNDING = int(load_setting('ROUNDING', SETTINGS_FILE))
    MODEL = load_setting('MODEL', SETTINGS_FILE)

    gdax_prod = load_setting('GDAX_PROD_URL', SETTINGS_FILE)

    GDAX_URL = 'https://' + load_setting('GDAX_URL', SETTINGS_FILE)

    if gdax_prod is not None:
        gdax_prod = 'https://' + load_setting('GDAX_PROD_URL', SETTINGS_FILE)
        GDAX_HISTORY_URL = gdax_prod + '/products/{}/candles'.format(PAIR)
        GDAX_PRODUCT_URL = gdax_prod + '/products/{}/ticker'.format(PAIR)
    else:
        GDAX_HISTORY_URL = GDAX_URL + '/products/{}/candles'.format(PAIR)
        GDAX_PRODUCT_URL = GDAX_URL + '/products/{}/ticker'.format(PAIR)

    order_id = None
    print('Starting bot on %s for %s' % (GDAX_URL, PAIR))
    while True:
        try:
            order_id = main_loop(order_id)
        except Exception as e:
            eprint(e)
            time.sleep(MARKET_PERIOD)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file')
    args = parser.parse_args()
    main(args.settings_file)
